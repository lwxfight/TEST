#!/bin/bash

#SBATCH --job-name uniformer         # 作业名
#SBATCH --partition compute
#SBATCH --nodes 1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem 26720
#SBATCH --output uniformer.log

# uniformer.log      # 保存输出结果

# running

echo Start slurm job at `date`

module load anaconda/3
module load cuda/10.2
module add gcc/5.4.0

source activate uniformer
export PYTHONPATH=/public/home/liuwx/perl5/UniFormer/video_classification/slowfast:$PYTHONPATH

python tools/run_net.py \
  --cfg /public/home/liuwx/perl5/UniFormer/video_classification/exp/uniformer_small_k400_8x8/config.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/Infrared \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.BATCH_SIZE 32 \
  NUM_GPUS 1 \
  UNIFORMER.DROP_DEPTH_RATE 0.1 \
  SOLVER.MAX_EPOCH 80 \
  SOLVER.BASE_LR 4e-5 \
  SOLVER.WARMUP_EPOCHS 10.0 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  RNG_SEED 6666 \
  OUTPUT_DIR ./logs/base1
