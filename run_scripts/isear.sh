#!/bin/bash
#SBATCH -J isea-r-grid # 作业名是xxx
#SBATCH -p defq # 提交到xx队列
#SBATCH -N 1 # 使用 1 个节点
#SBATCH --ntasks-per-node=1 # 每个节点开启 1 个进程
#SBATCH --cpus-per-task=6 # 每个进程占用 6 个 CPU 核心
#SBATCH --gres=gpu:1 # 如果是 GPU 任务需要在此行定义 GPU 数量
#SBATCH -t 120:00:00 # 任务最大运行时间是 2 day
#SBATCH -o ./logs/isear/731-isear-grid-l01-10-review.log

CUDA_VISIBLE_DEVICES=1 nohup python3 -u run_classifier.py \
    --pretrained_model_path ./models/78bert-review-uer.bin \
    --vocab_path  ./models/review-vocab.txt \
    --train_path  ./datasets/isear/train.tsv \
    --dev_path  ./datasets/isear/dev.tsv \
    --test_path  ./datasets/isear/test.tsv \
    --output_model_path  ./models/730-isear-grid-l01-10-review.bin \
    --config_path ./models/bert/base_config.json \
    --epochs_num 5 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder transformer \
    --mask fully_visible \
    --kg_name isear \
    --workers_num 1 \
    --em_weight 0.6 \
    --mylambda 0.6 \
    --k0 0\
    --k 2 \
    --l_ra0 1 \
    --l_ra 11 \
    --step 0.1 \
    
