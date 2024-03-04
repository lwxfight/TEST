work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
  --cfg /public/home/liuwx/perl5/UniFormer/video_classification/exp/uniformer_small_k400_8x8/config.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/Infrared \
  #DATA.PATH_PREFIX  ./ \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.BATCH_SIZE 32 \
  NUM_GPUS 1 \
  UNIFORMER.DROP_DEPTH_RATE 0.1 \
  SOLVER.MAX_EPOCH 40 \
  SOLVER.BASE_LR 4e-4 \
  SOLVER.WARMUP_EPOCHS 10.0 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  RNG_SEED 6666 \
  OUTPUT_DIR ./logs/base
