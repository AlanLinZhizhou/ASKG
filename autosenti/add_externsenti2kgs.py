"""

add_externsenti2kgs
"""
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_spo_path", type=str, required=True, help="the path of spo files.")
    parser.add_argument("--output_spo_path", type=str, required=True, help="the path of spo files.")
    parser.add_argument("--domain1", type=str, required=False, default='mr' ,help="the path of spo files.")
    args = parser.parse_args()
    with open('./lexicon/senti_score.txt', 'r') as f1:
        score_dict = f1.readlines()

    domains=[args.domain1]
    temp_dict = {}
    for domain in domains:

        enc = 'utf-8'
        with open(args.input_spo_path, 'r',encoding=enc) as f2:
            senti_dict = f2.readlines()

        for score_item in score_dict:
            k, v = score_item.split('\t')
            if k in temp_dict:
                origin_v, cnt = temp_dict[k].split('?')
                new_cnt = str(int(cnt) + 1)
                new_v = float(origin_v.strip()) + float(v.strip())
                temp_dict[k] = str(new_v) + '?' + new_cnt
            else:
                temp_dict[k] = str(v.strip()) + '?' + str(1)
        with open(args.output_spo_path+'_addsenti','w',encoding='utf-8') as f3:
            for senti_item in senti_dict:
                key = senti_item.split('\t')[1].strip()
                key = key + '#a'
                if key in temp_dict:
                    total_num, cnt2 = temp_dict[key].split('?')
                    add_num = float(total_num) / float(cnt2)
                else:
                    add_num = 0
                f3.write(senti_item.strip() + '\t' + str(add_num) + '\n')

if __name__ == "__main__":
    main()