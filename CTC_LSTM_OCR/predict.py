from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import cfg

import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
from net.cnn_lstm_ctc_ocr import CNN_LSTM_CTC_OCR
import argparse
from utils.textgen import Textgen


def parse_args():
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('--model', dest='model',
                        help='model to predict')
    args = parser.parse_args()
    return args


def _resize_and_pad(filename, height, width):
    _npimg = 255 * np.ones((height, width, 3), dtype=np.float32)
    _img = Image.open(filename)
    new_size = (int(32 * _img.size[0] / _img.size[1]), 32)
    _img = _img.resize(new_size, Image.ANTIALIAS)
    _img = np.array(_img.convert('RGB'))
    _pad = min(width, _img.shape[1])
    _npimg[:, 0:_pad, :] = _img[:, 0:_pad, :]
    _npimg = _npimg.astype(dtype=np.float32)

    return _npimg, new_size[0]


def _get_data(batch_info):
    _data = dict()
    _label = list()
    img_w = cfg.IMAGE_WIDTH
    _data['img_batch'] = np.zeros((len(batch_info), 32, img_w, 3),
                                  dtype=np.float32)
    _data['widths'] = np.zeros(len(batch_info),
                              dtype=np.int32)
    _iter = 0
    for key, val in batch_info.items():
        _img, _width = _resize_and_pad(key, 32, img_w)
        _data['img_batch'][_iter, :, :, :] = _img
        _data['widths'][_iter] = _width
        _label.append(val['label'])
        _iter += 1
    _data['img_batch'] = (_data['img_batch'] - 127.5) / 127.5

    return _data, _label


def next_val_batch():
    batch_key = imdb.get_next_val_batch()
    batch_info = dict()
    for key in batch_key:
        batch_info[key] = imdb.val_list[key]

    data, label = _get_data(batch_info)

    return data, label


if __name__ == '__main__':
    args = parse_args()
    imdb = Textgen(batch_size=5)
    cls = cfg.CLASS
    num_classes = len(cls)
    net = CNN_LSTM_CTC_OCR(num_classes=num_classes)
    gpu_options = tf.GPUOptions(
        allow_growth=True,
        allocator_type='BFC'
    )
    config = tf.ConfigProto(allow_soft_placement=True,
                            gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        net.create_architecture(is_training=True)
        print('Loading model checkpoint from {:s}'.format(args.model))
        saver = tf.train.Saver()
        saver.restore(sess, args.model)
        print('Loaded')
        data, label = next_val_batch()

        [pr] = net.predict(sess, data)
        print(pr)
        print(label)

