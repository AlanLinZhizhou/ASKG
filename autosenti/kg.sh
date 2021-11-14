#!/bin/bash
#SBATCH -J gene-mr-kg # 作业名是xxx
#SBATCH -p defq # 提交到xx队列
#SBATCH -N 1 # 使用 1 个节点
#SBATCH --ntasks-per-node=1 # 每个节点开启 1 个进程
#SBATCH --cpus-per-task=6 # 每个进程占用 6 个 CPU 核心
#SBATCH --gres=gpu:1 # 如果是 GPU 任务需要在此行定义 GPU 数量
#SBATCH -t 48:00:00 # 任务最大运行时间是 2 day
#SBATCH -o ./logs/619-mr-generate_kg.log


CUDA_VISIBLE_DEVICES=0 nohup python3 -u generate_kgs.py 
    
