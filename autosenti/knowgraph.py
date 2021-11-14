# coding: utf-8
"""
KnowledgeGraph
"""
import os
import autosenti.config as config

import numpy as np
import nltk
import autosenti.wordpiece as wp
from skgframework.utils import Vocab, math, random

from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format("./models/GoogleNews-vectors-negative300.bin",binary=True)


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, predicate=False):
        self.predicate = predicate
        self.task = ''
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.special_tags = set(config.NEVER_SPLIT_TAG)
        self.sum_em = [1.0]
        self.kg_form = ['good', 'bad']


        # math.pow(float(g_num) / float(fr) - float(b_num) / float(fr), 2)
        for i in range(1, 251):
            sum = 0
            for j in range(0, i + 1):
                sum = sum + math.pow(j / i - (i - j) / i, 2)
            self.sum_em.append(sum)

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            print('spo_path: ',spo_path)
            task_name = spo_path.split('/')[8][0:4]

            self.task = task_name

            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        if task_name == 'sst5':
                            subj, pred, obje, frequency, conflict, good_num, bad_num, won_num, ter_num, ext_sent = line.strip().split(
                                "\t")
                        elif task_name in ['sad_','joy_','fear','ange']:
                            subj, pred, obje, frequency, conflict, not_j, low_j, mid_j, high_j, ext_sent = line.strip().split(
                                "\t")
                        elif task_name in ['isea']:
                            subj, pred, obje, frequency, conflict, anger, disgust, fear, joy, sad = line.strip().split(
                                "\t")
                        elif task_name in ['AMAN']:
                            subj, pred, obje, frequency, conflict, anger, disgust, fear, joy, sad, surprise = line.strip().split(
                                "\t")
                        elif task_name in ['emos']:
                            subj, pred, obje, frequency, conflict, anger, disgust, fear, joy, sad, surprise, shame = line.strip().split(
                                "\t")
                        elif task_name in ['alm.']:
                            subj, pred, obje, frequency, conflict, anger, fear, joy, sad,  surprise = line.strip().split(
                                "\t")
                        else:
                            subj, pred, obje, frequency, conflict, good_num, bad_num, ext_sent = line.strip().split(
                                "\t")
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        # slight changes
                        if task_name == 'sst5':
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + good_num + '\t' + bad_num + '\t' + won_num + '\t' + ter_num + '\t' + ext_sent
                        elif task_name in ['sad_', 'joy_', 'fear', 'ange']:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + not_j + '\t' + low_j + '\t' + mid_j + '\t' + high_j + '\t' + ext_sent
                        elif task_name in ['isea']:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + anger + '\t' + disgust + '\t' + fear + '\t' + joy + '\t' + sad
                        elif task_name in ['AMAN']:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + anger + '\t' + disgust + '\t' + fear + '\t' + joy + '\t' + sad + '\t' +surprise
                        elif task_name in ['emos']:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + anger + '\t' + disgust + '\t' + fear + '\t' + joy + '\t' + sad + '\t' +surprise+ '\t' +shame
                        elif task_name in ['alm.']:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + anger + '\t' + fear + '\t' + joy + '\t' + sad + '\t' +surprise
                        else:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + good_num + '\t' + bad_num + '\t' + ext_sent
                    else:
                        if task_name == 'sst5':
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + good_num + '\t' + bad_num + '\t' + won_num + '\t' + ter_num + '\t' + ext_sent
                        elif task_name in ['sad_', 'joy_', 'fear', 'ange']:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + not_j + '\t' + low_j + '\t' + mid_j + '\t' + high_j + '\t' + ext_sent
                        elif task_name in ['isea']:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + anger + '\t' + disgust + '\t' + fear + '\t' + joy + '\t' + sad
                        elif task_name in ['AMAN']:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + anger + '\t' + disgust + '\t' + fear + '\t' + joy + '\t' + sad + '\t' +surprise
                        elif task_name in ['emos']:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + anger + '\t' + disgust + '\t' + fear + '\t' + joy + '\t' + sad + '\t' +surprise+ '\t' +shame
                        elif task_name in ['alm.']:
                            value = pred + '\t' + obje + '\t' + frequency + '\t' + conflict + '\t' + anger + '\t' + fear + '\t' + joy + '\t' + sad + '\t' +surprise
                        else:
                            value = obje + '\t' + frequency + '\t' + conflict + '\t' + good_num + '\t' + bad_num + '\t' + ext_sent
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128,
                              vocab=None, em_weight=1.0):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """

        # add_cls=[C]
        # split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        # split_sent_batch = [nltk.word_tokenize(sent) for sent in sent_batch]
        # vocab = Vocab()
        # vocab.load('../models/google_uncased_en_vocab.txt')
        wptoken = wp.WordpieceTokenizer(vocab.i2w)
        split_sent_batch = [wptoken.tokenize(sent) for sent in sent_batch]

        # INSERT CLS_TOKEN
        split_sent_batch[0].insert(0, config.CLS_TOKEN)
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []

        for split_sent in split_sent_batch:

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            # traverse each word in the sentence
            word_index = 0
            for token in split_sent:
                # entities = self.lookup_table.get(token, [])[:max_entities]
                # entities = self.lookup_table.get(token, [])
                if token in self.lookup_table.keys():
                    entities_set = self.lookup_table[token]
                else:
                    entities_set = []
                # start to design strategy
                em = 0.0
                threshold = 0.11
                entities_sele = []
                entities_inject = []
                single_set = []
                single_set_inject = []
                final_sets = []
                ext_senti = 0
                # expectation filter and knowledge attention mechenism
                for e in entities_set:
                    if self.task == 'sst5':
                        opp, jud, fr, con, g_num, b_num, won_num, ter_num, ext_senti = e.split('\t')
                    elif self.task in ['sad_', 'joy_', 'fear', 'ange']:
                        opp, jud, fr, con, not_j, low_j, mid_j, high_j, ext_senti = e.split('\t')
                    elif self.task in ['isea']:
                        opp, jud, fr, con, anger, disgust, fear, joy, sad = e.split('\t')
                    elif self.task in ['AMAN']:
                        opp, jud, fr, con, anger, disgust, fear, joy, sad, surprise = e.split('\t')
                    elif self.task in ['emos']:
                        opp, jud, fr, con, anger, disgust, fear, joy, sad, surprise, shame = e.split('\t')
                    elif self.task in ['alm.']:
                        opp, jud, fr, con, anger, fear, joy,  sad, surprise = e.split('\t')
                    else:
                        opp, jud, fr, con, g_num, b_num, ext_senti = e.split('\t')
                    if float(ext_senti) > 0 and float(ext_senti) < 0.10:
                        continue
                    if float(ext_senti) < 0 and float(ext_senti) > -0.10:
                        continue
                    # we shall not wordpiece opp here
                    # reduce noise for fr larger than 1
                    if int(fr) > 1:
                        if self.task in ['sst5'] :
                            total_good, total_bad=float(g_num)+float(won_num),float(b_num)+float(ter_num)
                        elif self.task in ['sad_', 'joy_', 'fear', 'ange'] :
                            total_good, total_bad=float(mid_j)+float(high_j),float(not_j)+float(low_j)
                        elif self.task in ['isea']:
                            total_good, total_bad=float(joy),float(anger)+float(disgust)+float(sad)+float(fear)
                        elif self.task in ['AMAN']:
                            total_good, total_bad=float(joy)+float(surprise),float(anger)+float(disgust)+float(sad)+float(fear)
                        elif self.task in ['emos']:
                            total_good, total_bad=float(joy)+float(surprise),float(anger)+float(disgust)+float(sad)+float(fear)+float(shame)
                        elif self.task in ['alm.']:
                            total_good, total_bad=float(joy)+float(surprise),float(anger)+float(sad)+float(fear)
                        else:
                            total_good, total_bad=g_num,b_num
                        em = math.pow(float(total_good) / float(fr) - float(total_bad) / float(fr), 2)
                        if em > threshold:
                            # entities_sele.append(opp + ' ' + jud + ' ' + fr + ' ' + con + ' ' + g_num + ' ' + b_num)
                            # find the most matching word
                            sim_list = []

                            for sop in split_sent:
                                # calculate similarity
                                if sop not in model.vocab or opp not in model.vocab:
                                    sim_list.append(0)
                                else:
                                    # s_list=[]
                                    try:
                                        sim_list.append(model.similarity(sop, opp))
                                    except:
                                        sim_list.append(0)
                                    # print('sop: ', sop, 'opp: ', opp, ' ;word not found, initialize it with 0')
                                    # sim_index.append(sop_index)
                                # sop_index += 1
                            # if len(sim_list) == 0:
                            #     simi = 0
                            #     dis = 0
                            #     weight = em / self.sum_em[int(fr)]
                            #     entities_sele.append([opp, jud, fr, con, g_num, b_num, simi, dis, weight])
                            #     # entities_sele.append(opp + ' ' + jud + ' ' + fr + ' ' + con + ' ' + g_num + ' ' + b_num + ' ' + str(simi) + ' ' + str(dis)+' '+str(em))
                            #
                            #     opp_pieced = ' '.join(wptoken.tokenize(opp))
                            #     entities_inject.append(opp_pieced + ' ' + jud)
                            # else:
                            # max similarity

                            simi = max(sim_list)
                            if simi > 1:
                                simi = 0.999
                            simi_index = sim_list.index(max(sim_list))
                            dis = abs(simi_index - word_index)
                            # w1 = dis / len(split_sent)
                            w1 = 0.5
                            weight = math.asin(float(simi)) / (1 + math.e ** w1) + em_weight * math.asin(
                                em / self.sum_em[int(fr)]) / (
                                             1 + math.e ** 0.5)
                            if self.task == 'sst5':
                                # for opinion absent in lexicon, jud remain unchanged
                                if float(ext_senti) > 0:
                                    jud = 'wonderful' if float(ext_senti) >= 0.35 else 'good'
                                if float(ext_senti) < 0:
                                    jud = 'terrible' if float(ext_senti) <= -0.35 else 'bad'
                            elif self.task in ['sad_', 'joy_', 'fear', 'ange','isea','AMAN','emos','alm.']:
                                # keep original judge
                                do_nothing = 1
                            else:
                                if float(ext_senti) == 0:
                                    jud = self.kg_form[0] if g_num >= b_num else self.kg_form[1]
                                else:
                                    jud = self.kg_form[0] if float(ext_senti) > 0 else self.kg_form[1]
                            if simi > 0.75:
                                if self.task in ['sad_', 'joy_', 'fear', 'ange']:
                                    entities_sele.append([opp, jud, fr, con, not_j, low_j, mid_j, high_j, simi, dis, weight])
                                elif self.task in ['isea']:
                                    entities_sele.append(
                                        [opp, jud, fr, con, anger, disgust, fear, joy, sad , simi, dis, weight])
                                elif self.task in ['AMAN']:
                                    entities_sele.append(
                                        [opp, jud, fr, con, anger, disgust, fear, joy, sad ,surprise, simi, dis, weight])
                                elif self.task in ['emos']:
                                    entities_sele.append(
                                        [opp, jud, fr, con, anger, disgust, fear, joy, sad ,surprise,shame, simi, dis, weight])
                                elif self.task in ['alm.']:
                                    entities_sele.append(
                                        [opp, jud, fr, con, anger,  fear, joy, sad ,surprise, simi, dis, weight])
                                elif self.task in ['sst5']:
                                    entities_sele.append([opp, jud, fr, con, g_num, b_num, won_num,ter_num,simi, dis, weight])
                                else:
                                    entities_sele.append([opp, jud, fr, con, g_num, b_num, simi, dis, weight])
                                # entities_sele.append(opp + ' ' + jud + ' ' + fr + ' ' + con + ' ' + g_num + ' ' + b_num + ' ' + str(simi) + ' ' + str(dis)+' '+str(em))
                                opp_pieced = ' '.join(wptoken.tokenize(opp))
                                entities_inject.append(opp_pieced + ' ' + jud)
                    # for fr==1, we don't leverage expectation filter
                    else:

                        sim_list = []
                        sim_index = []
                        # em = math.pow(float(g_num) / float(fr) - float(b_num) / float(fr), 2)

                        # if no adj. can be found in the sentence
                        for sop in split_sent:
                            # calculate similarity of words
                            if sop not in model.vocab or opp not in model.vocab:
                                sim_list.append(0)
                            else:
                                # s_list=[]
                                try:
                                    sim_list.append(model.similarity(sop, opp))
                                except:
                                    sim_list.append(0)

                        simi = max(sim_list)
                        if simi > 1:
                            simi = 0.999
                        simi_index = sim_list.index(max(sim_list))
                        dis = abs(simi_index - word_index)
                        # w1 = dis / len(split_sent)
                        w1 = 0.5
                        weight = math.asin(float(simi)) / (1 + math.e ** w1) + em_weight * math.asin(
                            em / self.sum_em[int(fr)]) / (
                                         1 + math.e ** 0.5)
                        # weight = math.asin(float(simi)) / (1 + math.e ** w1)
                        if self.task == 'sst5':
                            # for opinion absent in lexicon, jud remain unchanged
                            if float(ext_senti) > 0:
                                jud = 'wonderful' if float(ext_senti) >= 0.35 else 'good'
                            if float(ext_senti) < 0:
                                jud = 'terrible' if float(ext_senti) <= -0.35 else 'bad'
                        elif self.task in ['sad_', 'joy_', 'fear', 'ange','isea','AMAN','emos','alm.']:
                            # keep original judge
                            do_nothing = 1
                        else:
                            if float(ext_senti) == 0:
                                jud = self.kg_form[0] if g_num >= b_num else self.kg_form[1]
                            else:
                                jud = self.kg_form[0] if float(ext_senti) > 0 else self.kg_form[1]
                        if simi > 0.75:
                            if self.task in ['sad_', 'joy_', 'fear', 'ange']:
                                single_set.append([opp, jud, fr, con, not_j, low_j, mid_j, high_j, simi, dis, weight])
                            elif self.task in ['isea']:
                                single_set.append([opp, jud, fr, con, anger, disgust, fear, joy, sad , simi, dis, weight])
                            elif self.task in ['AMAN']:
                                single_set.append([opp, jud, fr, con, anger, disgust, fear, joy, sad ,surprise, simi, dis, weight])
                            elif self.task in ['emos']:
                                single_set.append([opp, jud, fr, con, anger, disgust, fear, joy, sad ,surprise,shame, simi, dis, weight])
                            elif self.task in ['alm.']:
                                single_set.append([opp, jud, fr, con, anger,  fear, joy, sad ,surprise, simi, dis, weight])
                            elif self.task in ['sst5']:
                                single_set.append([opp, jud, fr, con, g_num, b_num, won_num, ter_num, simi, dis, weight])
                            else:
                                single_set.append([opp, jud, fr, con, g_num, b_num, simi, dis, weight])
                            # single_set.append(opp + ' ' + jud + ' ' + fr + ' ' + con + ' ' + g_num + ' ' + b_num + ' ' + str(simi) + ' ' + str(dis)+' '+str(0))
                            opp_pieced = ' '.join(wptoken.tokenize(opp))
                            single_set_inject.append(opp_pieced + ' ' + jud)
                    word_index += 1



                # sort the final knowledge in c_set
                entities_inject2 = []
                single_set_inject2 = []

                flag = 0
                if len(single_set_inject) + len(entities_inject) <= 3:
                    sent_tree.append((token, single_set_inject + entities_inject))
                    final_sets = single_set_inject + entities_inject
                    flag = 1
                else:
                    # sort candidate sentiment knowledge
                    if len(entities_sele) > 0:
                        if self.task in ['sad_', 'joy_', 'fear', 'ange','sst5']:
                            entities_sele.sort(key=lambda x: (x[10]), reverse=True)
                        elif self.task in ['isea']:
                            entities_sele.sort(key=lambda x: (x[11]), reverse=True)
                        elif self.task in ['AMAN']:
                            entities_sele.sort(key=lambda x: (x[12]), reverse=True)
                        elif self.task in ['emos']:
                            entities_sele.sort(key=lambda x: (x[13]), reverse=True)
                        elif self.task in ['alm.']:
                            entities_sele.sort(key=lambda x: (x[11]), reverse=True)
                        else:
                            entities_sele.sort(key=lambda x: (x[8]), reverse=True)
                    if len(single_set) > 0:
                        if self.task in ['sad_', 'joy_', 'fear', 'ange', 'sst5']:
                            single_set.sort(key=lambda x: (x[10]), reverse=True)
                        elif self.task in ['isea']:
                            single_set.sort(key=lambda x: (x[11]), reverse=True)
                        elif self.task in ['AMAN']:
                            single_set.sort(key=lambda x: (x[12]), reverse=True)
                        elif self.task in ['emos']:
                            single_set.sort(key=lambda x: (x[13]), reverse=True)
                        elif self.task in ['alm.']:
                            single_set.sort(key=lambda x: (x[11]), reverse=True)
                        else:
                            single_set.sort(key=lambda x: (x[8]), reverse=True)
                    # 7:3  for fr > 1 and fr = 1; Here, the evaluation knowledge is wordpieced
                    if len(entities_sele) >= 3:
                        for i in range(0, 3):
                            opp_pieced = ' '.join(wptoken.tokenize(entities_sele[i][0]))
                            entities_inject2.append(opp_pieced + ' ' + entities_sele[i][1])
                    else:
                        for i in range(0, len(entities_sele)):
                            opp_pieced = ' '.join(wptoken.tokenize(entities_sele[i][0]))
                            entities_inject2.append(opp_pieced + ' ' + entities_sele[i][1])
                    if len(single_set) >= 3:
                        for i in range(0, 3):
                            opp_pieced = ' '.join(wptoken.tokenize(single_set[i][0]))
                            single_set_inject2.append(opp_pieced + ' ' + single_set[i][1])
                    else:
                        for i in range(0, len(single_set)):
                            opp_pieced = ' '.join(wptoken.tokenize(single_set[i][0]))
                            single_set_inject2.append(opp_pieced + ' ' + single_set[i][1])
                    final_sets = entities_inject2 + single_set_inject2

                if flag == 0:
                    sent_tree.append((token, entities_inject2 + single_set_inject2))

                # another sorting strategy
                # flag = 0
                # sort_together_inject = []
                # if len(single_set_inject) + len(entities_inject) <= 3:
                #     sent_tree.append((token, single_set_inject + entities_inject))
                #     final_sets = single_set_inject + entities_inject
                #     flag = 1
                # else:
                #
                #     sort_together = entities_sele + single_set
                #     if len(sort_together) > 0:
                #         sort_together.sort(key=lambda x: (x[8]), reverse=True)
                #     if len(sort_together)>=3:
                #         for i in range(0,3):
                #             opp_pieced = ' '.join(wptoken.tokenize(sort_together[i][0]))
                #             sort_together_inject.append(opp_pieced + ' ' + sort_together[i][1])
                #     else:
                #         for i in range(0,len(sort_together)):
                #             opp_pieced = ' '.join(wptoken.tokenize(sort_together[i][0]))
                #             sort_together_inject.append(opp_pieced + ' ' + sort_together[i][1])
                #
                # if flag == 0:
                #     sent_tree.append((token, sort_together_inject))

                if token in self.special_tags:
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                else:
                    # token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    # token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                # the place to fill the inject strategy
                for ent in final_sets:
                    length_temp = ent.split(' ')
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(length_temp) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(length_temp) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    # add_word = list(word)
                    add_word = word
                    # know_sent += add_word
                    know_sent.append(add_word)
                    seg += [0]
                pos += pos_idx_tree[i][0]

                s = len(sent_tree[i][1])
                s1 = sent_tree[i][1]
                j = 0
                for tuple in sent_tree[i][1]:
                    if len(tuple) != 0:
                        add_word = tuple.split(' ')
                        know_sent += add_word
                        seg += [1] * len(add_word)
                        if len(pos_idx_tree[i][1]) > 0:
                            temp1 = pos_idx_tree[i][1]
                            temp2 = pos_idx_tree[i][1][j]
                            pos += list(pos_idx_tree[i][1][j])
                        j += 1
                        # judge, po = tuple.split(' ')
                        # know_sent += add_word
                        # seg += [1]*len(add_word)
                        # add_word = [judge.strip(), po.strip()]
                        # know_sent += add_word
                        # pos += pos_idx_tree[i][1][j]
                        # j += 1

                # for j in range(len(sent_tree[i][1])):
                #     a = sent_tree[i][1]
                #     b = 0
                #     add_word = list(sent_tree[i][1][j])
                #     know_sent += add_word
                #     seg += [1]*2
                #     pos += pos_idx_tree[i][1][j]

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch