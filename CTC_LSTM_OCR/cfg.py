# BaseSolver config file

import numpy as np

# TRAINING SETTINGS

MAX_ITERATION = 1200000
CLASS = '0123456789*-'

RND_SEED = 7777

TRAIN_LEARNING_RATE = 0.01
TRAIN_MOMENTUM = 0.9

TRAIN_SUMMARY_INTERVAL = 100
TRAIN_DISPLAY = 100
TRAIN_WEIGHT_DECAY = 0.0001
TRAIN_SNAPSHOT_PREFIX = 'vgg16_faster_rcnn_no_resize'

TRAIN_SNAPSHOT_ITER = 1000
TRAIN_SNAPSHOT_KEPT = 3

TRAIN_LR_REDUCTION = 10000
TRAIN_LR_GAMMA = 0.9

IMAGE_WIDTH = 512
SEQ_LEN = 30
