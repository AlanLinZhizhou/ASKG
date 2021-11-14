#!/bin/bash
#SBATCH -J AK3L05t2 # 作业名是xxx
#SBATCH -p defq # 提交到xx队列
#SBATCH -N 1 # 使用 1 个节点
#SBATCH --ntasks-per-node=1 # 每个节点开启 1 个进程
#SBATCH --cpus-per-task=6 # 每个进程占用 6 个 CPU 核心
#SBATCH --gres=gpu:1 # 如果是 GPU 任务需要在此行定义 GPU 数量
#SBATCH -t 48:00:00 # 任务最大运行时间是 2 day
#SBATCH -o ./logs/ablation/recheck/length3-k7-lambda0-05-time2.log

CUDA_VISIBLE_DEVICES=1 nohup python3 -u run_classifier.py \
    --pretrained_model_path ./models/78bert-review-uer.bin \
    --vocab_path  ./models/review-vocab.txt \
    --train_path  ./datasets/ALM/train.tsv \
    --dev_path  ./datasets/ALM/dev.tsv \
    --test_path  ./datasets/ALM/test.tsv \
    --output_model_path  ./models/1110-length3-k7-lambda0-05-time2.bin \
    --config_path ./models/bert/base_config.json \
    --epochs_num 10 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder transformer \
    --mask fully_visible \
    --kg_name alml3 \
    --workers_num 1 \
    --em_weight 0.6 \
    --mylambda 0.6 \
    --k0 1 \
    --k 2 \
    --l_ra0 5 \
    --l_ra 6 \
    --step 0.01 \
    --report_steps 20 \


    
