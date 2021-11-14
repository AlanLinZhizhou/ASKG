"""
The entrance for sentiment classifcation and emotion detection.
"""
import sys
import os
import random
import argparse
import torch
import torch.nn as nn

from autosenti.config import PAD_TOKEN
from autosenti.knowgraph import KnowledgeGraph

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

import numpy as np

from skgframework.layers import *
from skgframework.encoders import *
from skgframework.utils.vocab import Vocab
from skgframework.utils.constants import *
from skgframework.utils import *
from skgframework.utils.optimizers import *
from skgframework.utils.config import load_hyperparam
from skgframework.utils.seed import set_seed
from skgframework.model_saver import save_model
from skgframework.opts import finetune_opts
# from multiprocessing import Process, Pool
import multiprocessing as mp
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score,f1_score

g_k = 3
g_lambda = 0.10
class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.mylambda = args.mylambda
        self.k = args.k
        self.use_vm = False if args.no_vm else True


    def forward(self, src, tgt, seg, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, seg)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        u, s, v = torch.svd(output)
        l1 = s.size(0)
        reg_loss = 0
        global g_k, g_lambda
        get_k = g_k
        get_mylambda = g_lambda
        for i in range(get_k):
            reg_loss = reg_loss + torch.pow(s[l1 - 1 - i], 2)
        loss_pt2 = reg_loss * (get_mylambda / 39.000)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and pos is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, pos) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1)) + loss_pt2
            else:
                # we run this loss function
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1)) + loss_pt2
            return loss, logits
        else:
            return None, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    get_task = path.split('/')
    if get_task[2] in ['imdb', 'sst3', 'sst2', 'sst5'] or get_task[3] in ['MR', 'kitchen']:
        enc = 'ISO-8859-1'
    else:
        enc = 'utf-8'
    with open(path, mode="r", encoding=enc) as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
        print('load pre-trained model:', args.pretrained_model_path)
    else:
        print('random initialization')
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer, scheduler


# Datset loader.
def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
    instances_num = input_ids.size()[0]
    for i in range(instances_num // batch_size):
        input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
        label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
        mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
        pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
        vms_batch = vms[i * batch_size: (i + 1) * batch_size]
        yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
    if instances_num > instances_num // batch_size * batch_size:
        input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
        label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
        mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
        pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:, :]
        vms_batch = vms[instances_num // batch_size * batch_size:]

        yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch


def add_knowledge_worker(params):
    p_id, sentences, columns, kg, vocab, args = params

    sentences_num = len(sentences)
    dataset = []

    for line_id, line in enumerate(sentences):

        if line_id % 10000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')
        if 1:
            # try:
            if len(line) == 2:
                # label = int(line[columns["label"]])

                label = int(line[0])
                text = line[1]

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length,
                                                              vocab=vocab, em_weight=args.em_weight)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                len1 = 0
                # wordpiece  for English words
                token_ids = [vocab.get(t) for t in tokens]
                for j in range(0, len(token_ids)):

                    if token_ids[j] == 0:
                        len1 = j
                        break
                vm1 = vm[0:len1, 0:len1]

                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                dataset.append((token_ids, label, mask, pos, vm))

            elif len(line) == 3:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]] + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length,
                                                              vocab=vocab, em_weight=args.em_weight)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm))

            elif len(line) == 4:  # for dbqa
                qid = int(line[columns["qid"]])
                label = int(line[columns["label"]])
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                text = CLS_TOKEN + text_a + SEP_TOKEN + text_b + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length,
                                                              vocab=vocab, em_weight=args.em_weight)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm, qid))
            else:
                pass

        # except:
        #     print("Error line: ", line, '   //', sys.exc_info())
    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None,
                vms_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    pos_ids_batch = soft_tgt_batch.to(args.device)
    vms_batch = vms_batch.to(args.device)
    loss, _ = model(src_batch, tgt_batch, seg_batch, pos=pos_ids_batch, vm=vms_batch)
    # if soft_tgt_batch is not None:
    #     soft_tgt_batch = soft_tgt_batch.to(args.device)
    #
    # loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def micro_f1(preds, labels):
    preds_set = []
    label_set = []

    corr = 0
    tot = 0

    for i in range(len(labels)):
        if preds[i] != 3:
            tot += 1
        if int(labels[i]) != 3:
            preds_set.append(int(preds[i]))
            label_set.append(int(labels[i]))
            if labels[i] == preds[i]:
                corr += 1

    p = corr * 1. / len(preds_set)
    r = corr * 1. / tot

    f1 = 2 * p * r / (p + r)
    return f1


def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    pos_ids = torch.LongTensor([example[3] for example in dataset])
    vms = [example[4] for example in dataset]
    # devp = dataset.split('/')
    # is_dev = True if devp[-1] == 'dev.tsv' else False
    batch_size = args.batch_size
    instances_num = src.size()[0]
    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()
    # self.
    pre_all = []

    for i, (src_batch, tgt_batch, seg_batch, pos_ids_batch, vms_batch) in enumerate(
            batch_loader(batch_size, src, tgt, seg, pos_ids, vms)):

        vms_batch = torch.LongTensor(vms_batch)
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        pos_ids_batch = pos_ids_batch.to(args.device)
        vms_batch = vms_batch.to(args.device)
        with torch.no_grad():
            loss, logits = args.model(src_batch, tgt_batch, seg_batch, pos_ids_batch, vms_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        pre_all += pred.cpu().numpy().tolist()
        # pre_all += pred
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()
    # print('pre_all\t', pre_all)
    # mf1 = micro_f1(pre_all, tgt.numpy().tolist())
    if not print_confusion_matrix:
        print('pre_all\t', pre_all)
        print('tgt\t', tgt.numpy().tolist())
        coorvalue = pearsonr(pre_all, tgt.numpy().tolist())[0]
        if args.train_path.split('/')[2] == 'emoint':
            print('Pearson correlation:\t', coorvalue)
        if args.train_path.split('/')[2] == 'emocontext':
            print('Micro F1 value:\t', micro_f1(pre_all, tgt.numpy().tolist()))
        if args.train_path.split('/')[2] in ['isear','AMAN','emosti','ALM']:
            print('Marco F1 value:\t', f1_score(tgt.numpy().tolist(), pre_all, average='macro'))
        if args.train_path.split('/')[2] == 'emolines':
            total = [0, 0, 0, 0, 0, 0, 0, 0]
            right = [0, 0, 0, 0, 0, 0, 0, 0]
            new_tgt = tgt.numpy().tolist()
            for i in range(0, len(pre_all)):
                total[new_tgt[i]] += 1
                if new_tgt[i] == pre_all[i]:
                    right[new_tgt[i]] += 1
            avg = 0.0
            weighted_score = 0.0
            if args.train_path.split('/')[3] == 'emopush':
                weighted_acc = [0.0095, 0.0072, 0.0028, 0.1425, 0.6685, 0.0962, 0.0349, 0.0385]
            if args.train_path.split('/')[3] == 'friends':
                weighted_acc = [0.0523, 0.0228, 0.0170, 0.1179, 0.4503, 0.1911, 0.0343, 0.1143]
            print('Report acc per label:')
            for j in range(0, len(total)):
                avg += right[j] / total[j]
                acc_perlabel = right[j] / total[j]
                weighted_score = weighted_score + weighted_acc[j] * acc_perlabel
                print("Label {}: {:.3f}".format(j, right[j] / total[j]))
            print('unweighted acc:\t', avg / 8)
            args.weighted_acc = weighted_score
            print('weighted acc:\t', weighted_score)
    # weighted_score = 0.0
    if print_confusion_matrix:
        # print("Confusion matrix:")
        # print(confusion)
        global g_k, g_lambda
        get_task = args.train_path.split('/')
        # print('Test set label:\t',pre_all)
        # print('Writing to test files')
        filename = 'EI-oc_en_' + get_task[3] + '_pred?k_' + str(g_k) + '-lambda_' + str(g_lambda) + '.txt'
        if get_task[2] in ['emoint'] and print_confusion_matrix:
            print('Test set label:\t', pre_all)
            print('Writing to test files')
            with open('./results/' + filename, mode='w', encoding='utf-8') as f1:
                with open(args.test_path, mode='r', encoding='utf-8') as f:
                    for line_id, line in enumerate(f):
                        if line_id == 0:
                            new_line = line.strip()
                        else:
                            content = line.split('\t')
                            content[3] = str(pre_all[line_id - 1])
                            new_line = '\t'.join(content)
                        # with open('./results/'+filename,mode='w',encoding='utf-8') as f1:
                        f1.write(new_line + '\n')
            return 0, 0
        else:
            print('pre_all\t', pre_all)
            print('tgt\t', tgt.numpy().tolist())
            print("Report precision, recall, and f1:")
            eps = 1e-9
            for i in range(confusion.size()[0]):
                p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
                r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
                f1 = 2 * p * r / (p + r + eps)
                print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
            if get_task[2] in ['isear','AMAN','emosti','ALM']:
                print('Marco F1 value:\t', f1_score(tgt.numpy().tolist(), pre_all, average='macro'))
            if get_task[2] in ['emolines']:
                total = [0, 0, 0, 0, 0, 0, 0,0]
                right = [0, 0, 0, 0, 0, 0, 0,0]
                new_tgt = tgt.numpy().tolist()
                for i in range(0, len(pre_all)):
                    total[new_tgt[i]] += 1
                    if new_tgt[i] == pre_all[i]:
                        right[new_tgt[i]] += 1
                avg = 0.0
                # weighted_score = 0.0
                if get_task[3] == 'emopush':
                    weighted_acc = [0.0095, 0.0072, 0.0028, 0.1425, 0.6685, 0.0962, 0.0349, 0.0385]
                if get_task[3] == 'friends':
                    weighted_acc = [0.0523, 0.0228, 0.0170, 0.1179, 0.4503, 0.1911, 0.0343, 0.1143]
                print('Report acc per label:')
                for j in range(0, len(total)):
                    avg += right[j] / total[j]
                    acc_perlabel = right[j] / total[j]
                    weighted_score = weighted_score + weighted_acc[j] * acc_perlabel
                    print("Label {}: {:.3f}".format(j, right[j] / total[j]))
                print('unweighted acc:\t', avg / 8)
                args.weighted_acc=weighted_score
                print('weighted acc:\t', weighted_score)
            # args.model.embedding

    # for layer in args.model.lay:

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    if args.train_path.split('/')[2] == 'emoint':
        print('Return Pearson Correlation')
        return coorvalue, confusion
    if args.train_path.split('/')[2] == 'emocontext':
        print('Return emocontext F1')
        return micro_f1(pre_all, tgt.numpy().tolist()), confusion
    if args.train_path.split('/')[2] in ['isear','AMAN','emosti','ALM']:
        print('Return marco F1')
        return f1_score(tgt.numpy().tolist(), pre_all, average='macro'), confusion
    if args.train_path.split('/')[2] in ['emolines']:
        print('Return weighted acc:\t',weighted_score)
        return args.weighted_acc, confusion

    return correct / len(dataset), confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    # knowledge graph
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")
    parser.add_argument("--em_weight", type=float, default=1.0,
                        help="Weight of the knowledge em.")
    parser.add_argument("--mylambda", type=float, default=1.0,
                        help="Weight of the knowledge em.")
    parser.add_argument("--k", type=int, default=1,
                        help="k feature to constraint")
    parser.add_argument("--l_ra", type=int, default=1,
                        help="range of lambda to constraint")
    parser.add_argument("--l_ra0", type=int, default=1,
                        help="lower bounds of lambda to constraint")
    parser.add_argument("--step", type=float, default=0.1,
                        help="step of lambda to constraint")
    parser.add_argument("--k0", type=int, default=0,
                        help="lower bounds of k")
    parser.add_argument("--weighted_acc", type=float, default=0.0,
                        help="weighted acc of emolines")
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Load vocab
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(args.device)

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=True)

    def read_dataset(path, workers_num=1, is_dev=True):
        print("Loading sentences from {}".format(path))
        dataset, columns = [], {}
        sentences = []
        # set encoding
        get_task = args.train_path.split('/')
        if get_task[2] in ['imdb', 'sst3','sst2'] or get_task[3] in ['MR']:
            enc = 'ISO-8859-1'
        else:
            enc = 'utf-8'
        if get_task[2] in ['emoint'] and not is_dev:
            with open(path, mode='r', encoding=enc) as f1:
                for line_id, line in enumerate(f1):
                    if line_id == 0:
                        continue
                    sentences.append('1\t' + line.split('\t')[1])
        else:
            with open(path, mode='r', encoding=enc) as f:
                for line_id, line in enumerate(f):
                    if line_id == 0:
                        continue
                    sentences.append(line)
        sentence_num = len(sentences)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(
            sentence_num, workers_num))
        if workers_num > 1 and is_dev:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append(
                    (i, sentences[i * sentence_per_block: (i + 1) * sentence_per_block], columns, kg, vocab, args))
            pool = mp.Pool(workers_num)
            # mp.set_start_method('spawn')
            res = pool.map(add_knowledge_worker, params)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, columns, kg, vocab, args)
            dataset = add_knowledge_worker(params)

        return dataset

    # Training phase.
    for k in range(args.k0, args.k):
        for lam in range(args.l_ra0, args.l_ra):
            load_or_initialize_parameters(args, model)

            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.device_count() > 1:
                print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
                model = nn.DataParallel(model)

            model = model.to(args.device)
            global g_k, g_lambda
            g_lambda = lam * args.step
            g_k = k
            print('k =', g_k, '\tlambda = ', g_lambda)
            if k == 0:
                lam += 14
            trainset = read_dataset(args.train_path, args.workers_num)
            print("Shuffling dataset")
            random.shuffle(trainset)
            instances_num = len(trainset)
            batch_size = args.batch_size

            src = torch.LongTensor([example[0] for example in trainset])
            tgt = torch.LongTensor([example[1] for example in trainset])
            seg = torch.LongTensor([example[2] for example in trainset])

            soft_tgt = torch.FloatTensor([example[3] for example in trainset])

            vms = [example[4] for example in trainset]

            args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

            print("Batch size: ", batch_size)
            print("The number of training instances:", instances_num)

            optimizer, scheduler = build_optimizer(args, model)

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
                args.amp = amp

            if torch.cuda.device_count() > 1:
                print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
                model = torch.nn.DataParallel(model)
            args.model = model

            total_loss, result, best_result = 0.0, 0.0, 0.0

            print("Start training.")

            for epoch in range(1, args.epochs_num + 1):
                model.train()
                for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch, vms_batch) in enumerate(
                        batch_loader(batch_size, src, tgt, seg, soft_tgt, vms)):
                    vms_batch = torch.LongTensor(vms_batch)
                    loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch,
                                       soft_tgt_batch,
                                       vms_batch)
                    total_loss += loss.item()
                    if (i + 1) % args.report_steps == 0:
                        print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                          total_loss / args.report_steps))
                        total_loss = 0.0

                result = evaluate(args, read_dataset(args.dev_path))
                if result[0] >= best_result:
                    best_result = result[0]
                    save_model(model, args.output_model_path)
                    # model.output_layer_2

            # Evaluation phase.
            if args.test_path is not None:
                print("Test set evaluation.")
                if torch.cuda.device_count() > 1:
                    model.module.load_state_dict(torch.load(args.output_model_path))
                else:
                    model.load_state_dict(torch.load(args.output_model_path))
                res, _ = evaluate(args, read_dataset(args.test_path, 1, False), True)
                get_task = args.train_path.split('/')
                if get_task[2] == 'emocontext':
                    print('final F1 value:\t', res)
                if get_task[2] == 'emoint':
                    print('final pearson corealation:\t', res)
            if k == 0:
                break


if __name__ == "__main__":
    main()
