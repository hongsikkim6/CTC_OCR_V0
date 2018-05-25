"""Deep Learning base solver module

This module is a base solver module to train, save and restore DL project

Example:
    To train a net using db,

    from basesolver import BaseSolver
    Solver = BaseSolver(sess, net, tb_dir, log_dir)
    Solver.train_model(sess, max_iters=10000)

"""
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import os
import cfg
import time
import pickle
import numpy as np
import glob
from net.cnn_lstm_ctc_ocr import CNN_LSTM_CTC_OCR
from utils.textgen import Textgen
from PIL import Image


class Solver(object):
    """Base solver class

    Some notes here

    Arguments:
        sess: Tensorflow Session, network will run on this session
        network: Class, net(work) should have the following properties

            net.create_architecture()
            '''create tensorflow graph and returns total loss

            Returns:
                Dictionary which contains at least one element
                layers['total_loss']: tf.float, for SGD training
            '''

            net.get_variables_to_restore(variables, var_keep_dic)
            '''get variables to restore at the pretrained model

            Args:
                variables: List of Variables, tf.global_variables()
                var_keep_dic: List of Variables, from checkpointreader

            Returns:
                Dict of List of Variables, indicating variables to restore
                from pretrained model
            '''

            net.train_step_with_summary(sess, data, train_op)
            '''train network with tensorboard summary

            Args:
                sess: Tensorflow Session
                data:
                train_op:

            net.train_step(sess, data, train_op)

            net.get_summary(sess, val_data)

        tb_dir: String, path to save tensorboard summary
        log_dir: String, path to save model checkpoint files
        pretrained_model: String, path to restore pretrained model
            default None(No pretrained model)
    """
    def __init__(self, sess, network, tb_dir, log_dir,
                 num_classes, pretrained_model=None):
        self.net = network
        self.pretrained_model = pretrained_model
        self.log_dir = log_dir
        self.tb_dir = tb_dir
        self.tb_dir_val = tb_dir + '_val'
        self.num_classes = num_classes
        self.cur = 0
        self.vcur = 0

    def train_model(self, sess, max_iters=cfg.MAX_ITERATION):
        with sess.graph.as_default():
            # EDIT: Delete if don't use fixed random seed
            tf.set_random_seed(cfg.RND_SEED)

            layers = self.net.create_architecture()
            loss = layers['total_loss']

            # EDIT: Define learning rate, optimizer and params
            """
            lr = tf.train.exponential_decay(
                learning_rate=cfg.TRAIN_LEARNING_RATE,
                global_step=tf.train.get_global_step(),
                decay_steps=cfg.TRAIN_LR_REDUCTION,
                decay_rate=cfg.TRAIN_LR_GAMMA,
                staircase=False,
                name='learning rate'
            )
            """
            lr = tf.Variable(cfg.TRAIN_LEARNING_RATE, trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=lr,
                beta1=cfg.TRAIN_MOMENTUM
            )

            gvs = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(gvs)

            self.saver = tf.train.Saver(max_to_keep=100000)

            # EDIT: Delete if don't use tensorboard
            self.writer = tf.summary.FileWriter(self.tb_dir, sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tb_dir_val)

        lsf, nfiles, sfiles = self.find_snapshots()

        if lsf == 0:
            rate, last_snapshot_iter, np_paths, ss_paths = \
                self.initialize(sess)
        else:
            rate, last_snapshot_iter, np_paths, ss_paths = \
                self.restore(sess, str(sfiles[-1]), str(nfiles[-1]))

        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        """
        data = self.next_batch()
        a = self.net.test_var(sess, data, test)
        print(a.indices)
        print(a.values)
        """
        while iter < max_iters + 1:
            # EDIT: Delete if no lr reduction
            if iter % cfg.TRAIN_LR_REDUCTION == 0:
                self.snapshot(sess, iter, rate)
                rate *= cfg.TRAIN_LR_GAMMA
                sess.run(tf.assign(lr, rate))

            data = self.next_batch()
            now = time.time()
            if iter == 1 or iter % cfg.TRAIN_SUMMARY_INTERVAL == 0:
                losses, features, pred, summary = self.net.train_step_with_summary(sess, data,
                                                                   train_op)
                features = np.array(features)
                print(iter, losses)
                self.writer.add_summary(summary, float(iter))

                # EDIT: Delete if no validations
                val_data = self.next_val_batch()
                summary_val = self.net.get_summary(sess, val_data)
                self.valwriter.add_summary(summary_val, float(iter))

                last_summary_time = now
            else:
                losses = self.net.train_step(sess, data, train_op)

            # if iter % (cfg.TRAIN_DISPLAY) == 0:
                # EDIT: Display EDIT
                # print(iter, losses)

            if iter == 1 or iter % (cfg.TRAIN_SNAPSHOT_ITER) == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self.snapshot(sess, iter, rate)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                if len(np_paths) > cfg.TRAIN_SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)
            iter += 1

        self.writer.close()
        self.valwriter.close()

    def initialize(self, sess):
        np_paths = []
        ss_paths = []
        last_snapshot_iter = 0
        rate = cfg.TRAIN_LEARNING_RATE

        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))


        return rate, last_snapshot_iter, np_paths, ss_paths

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:
            print(str(e))

    def restore(self, sess, sfile, nfile):
        np_paths = [nfile]
        ss_paths = [sfile]
        last_snapshot_iter, last_snapshot_rate = self.from_snapshot(
            sess, sfile, nfile)

        return last_snapshot_rate, last_snapshot_iter, np_paths, ss_paths

    def from_snapshot(self, sess, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        print('Restored')

        # Edit: use nfile to save and restore hyperparameters
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            cur = pickle.load(fid)
            vcur = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)
            last_snapshot_rate = pickle.load(fid)

        np.random.set_state(st0)
        self.cur = cur
        self.vcur = vcur

        return last_snapshot_iter, last_snapshot_rate

    def snapshot(self, sess, iter, rate):
        net = self.net
        filename = cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.log_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # EDIT: use nfile to save and restore hyperparameters
        nfilename = cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.log_dir, nfilename)

        st0 = np.random.get_state()
        cur = self.cur
        vcur = self.vcur
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vcur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(rate, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def find_snapshots(self):
        # EDIT: this will automatically find the most recent snapshots.
        sfiles = os.path.join(
            self.log_dir, cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)

        nfiles = [ss.replace('.ckpt.meta', '.pkl') for ss in sfiles]
        sfiles = [ss.replace('.meta', '') for ss in sfiles]

        lsf = len(sfiles)
        assert lsf == len(nfiles)

        return lsf, nfiles, sfiles

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(ss_paths) - cfg.TRAIN_SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            os.remove(str(sfile + '.meta'))
            ss_paths.remove(sfile)

        # EDIT: remove hyperparameter nfiles
        to_remove = len(np_paths) - cfg.TRAIN_SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

    def next_batch(self):
        batch_key = imdb.get_next_train_batch()
        batch_info = dict()
        for key in batch_key:
            batch_info[key] = imdb.train_list[key]

        data = self._get_data(batch_info)

        return data

    def next_val_batch(self):
        batch_key = imdb.get_next_val_batch()
        batch_info = dict()
        for key in batch_key:
            batch_info[key] = imdb.val_list[key]

        data = self._get_data(batch_info)

        return data

    def _get_data(self, batch_info):
        _data = dict()
        img_w = cfg.IMAGE_WIDTH
        _data['img_batch'] = np.zeros((len(batch_info), 32, img_w, 3),
                                      dtype=np.float32)
        _data['widths'] = np.zeros(len(batch_info),
                                  dtype=np.int32)
        _data['label'] = np.ones((len(batch_info), cfg.SEQ_LEN),
                                  dtype=np.int32) * self.num_classes
        _iter = 0
        for key, val in batch_info.items():
            _img, _width = self._resize_and_pad(key, 32, img_w)
            _data['img_batch'][_iter, :, :, :] = _img
            _data['widths'][_iter] = _width
            _data['label'][_iter, 0:len(val['label'])] = val['label']
            _iter += 1

        _data['img_batch'] = (_data['img_batch'] - 127.5) / 127.5

        return _data

    def _resize_and_pad(self, filename, height, width):
        _npimg = 255 * np.ones((height, width, 3), dtype=np.float32)
        _img = Image.open(filename)
        new_size = (int(32 * _img.size[0] / _img.size[1]), 32)
        _img = _img.resize(new_size, Image.ANTIALIAS)
        _img = np.array(_img.convert('RGB'))
        _pad = min(width, _img.shape[1])
        _npimg[:, 0:_pad, :] = _img[:, 0:_pad, :]
        _npimg = _npimg.astype(dtype=np.float32)

        return _npimg, new_size[0]


if __name__ == '__main__':
    imdb = Textgen(batch_size=64)
    cls = cfg.CLASS
    num_classes = len(cls)
    log_dir = os.path.join(os.path.dirname(__file__), 'log')
    now = time.asctime(time.localtime(time.time()))
    tb_dir = os.path.join(log_dir, now)
    log_dir = os.path.join(log_dir, 'snapshot')
    net = CNN_LSTM_CTC_OCR(num_classes=num_classes)
    gpu_options = tf.GPUOptions(
        allow_growth=True,
        allocator_type='BFC'
    )
    config = tf.ConfigProto(allow_soft_placement=True,
                            gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        sw = Solver(
            sess=sess,
            network=net,
            tb_dir=tb_dir,
            log_dir=log_dir,
            num_classes=num_classes
        )
        print('Solver Loaded')
        sw.train_model(sess, max_iters=100000)

    print('Done')
