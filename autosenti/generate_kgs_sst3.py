# -*- coding: utf-8 -*-
"""
generate sst3 kg

"""

from stanfordcorenlp import StanfordCoreNLP
import autosenti.wordpiece as wp
from skgframework.utils import Vocab

path = './stanford-corenlp-full-2018-10-05'
nlp = StanfordCoreNLP(path, lang='en')

# 读取句子

with open('../datasets/sst3/train.tsv', 'r', encoding='ISO-8859-1') as f:
    l_dict = f.readlines()

# with open('G:/post_pro/K-BERT/datasets/kitchen_housewares/train.tsv', 'r', encoding='utf-8') as f:
#     l_dict = f.readlines()

judges = ''
triples = ''
op_dict = {}
vocab = Vocab()
vocab.load('../models/google_uncased_en_vocab.txt')
count_wp = 0
op_skip_list = ['good', 'bad', 'great', 'Bad', 'other', 'such', 'second', 'third', 'fourth', 'fifth', 'sixth',
                'seventh', 'better', 'best', 'worse', 'worst',
                'everyday', '3rd', '2nd', '1st','previous']
ent_skip_list = ['year', 'can', 'other', 'others', 'with', 'lots', 'ones', 'way', 'uses', 'use', 'time', 'kinds',
                 'kind', 'times', 'way', 'brother', 'sister', 'bit', 'days', 'seasons', 'things', 'thing', 'boy',
                 'girl', 'something', 'hate', 'ages', 'words', 'me']

for i in range(1, len(l_dict)):
    # for i in range(1, 3):

    s_label = l_dict[i].split('\t')[0]
    s_content = l_dict[i].split('\t')[1]
    if s_label == '0':
        judges = 'bad'
    if s_label == '1':
        continue
    if s_label == '2':
        judges = 'good'
    token = nlp.word_tokenize(s_content)
    if i % 500 == 0:
        print(i, ' examples,  ', len(l_dict) - i - 1, 'to go')
    dependencyParse = nlp.dependency_parse(s_content)
    pos = nlp.pos_tag(s_content)
    # a1=nlp.pos_tag('apple')[0][1]
    wptoken = wp.WordpieceTokenizer(vocab.i2w)
    for i, begin, end in dependencyParse:
        # print (i, '-'.join([str(begin), token[begin-1]]), '-'.join([str(end),token[end-1]]))
        if i == 'amod' or i == 'nsubj':
            if nlp.pos_tag(token[begin - 1])[0][1] in ['NN', 'NNS', 'NNP', 'NNPS'] and \
                    nlp.pos_tag(token[end - 1])[0][1] in ['JJ', 'JJR', 'JJS']:
                entity1 = token[begin - 1]
                op = token[end - 1]
                entity2 = wptoken.tokenize(entity1)[0]
                entity2 = entity2.strip()
                if len(entity2) < 12 or op.lower().strip() in op_skip_list or entity2.lower().strip() in ent_skip_list:
                    continue
                if op[-2:-1] in ['st', 'th']:
                    continue
                if len(op.split('-')) >= 3:
                    continue
                if op.lower() in ['true', 'right']:
                    judges = 'good'
                if op.lower() in ['false', 'wrong']:
                    judges = 'bad'
                if len(entity2) > 1:
                    count_wp += 1

                if entity2 + '\t' + op not in op_dict.keys():
                    good_num, bad_num = 0, 0
                    if judges == 'good':
                        good_num = 1
                    else:
                        bad_num = 1
                    op_dict[entity2 + '\t' + op] = judges + '\t' + '1' + '\t' + '0' + '\t' + str(good_num) + '\t' + str(
                        bad_num)

                else:
                    ac_judges, count_tr, conf, good_num, bad_num = op_dict[entity2 + '\t' + op].split('\t')
                    count_tr = int(count_tr) + 1
                    if judges == 'good':
                        good_num = int(good_num) + 1
                    if judges == 'bad':
                        bad_num = int(bad_num) + 1
                    # to tell if there are conflicts
                    if conf == '1' or judges != ac_judges:
                        op_dict[entity2 + '\t' + op] = judges + '\t' + str(count_tr) + '\t' + '1' + '\t' + str(
                            good_num) + '\t' + str(bad_num)
                    else:
                        op_dict[entity2 + '\t' + op] = judges + '\t' + str(count_tr) + '\t' + '0' + '\t' + str(
                            good_num) + '\t' + str(bad_num)

    with open('./kgs/sst3.spo', 'w',encoding='utf-8') as f1:
        for (k, v) in op_dict.items():
            # triples = triples + k + '\t' + v + '\n'
            f1.write(k + '\t' + v + '\n')

