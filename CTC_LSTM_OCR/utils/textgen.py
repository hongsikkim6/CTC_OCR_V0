import os
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import subprocess
import random
import glob


class Textgen(object):
    def __init__(self, batch_size):
        self._classes = '0123456789*-'
        self._num_classes = len(self._classes)
        self._class_to_ind = dict(
            list(
                zip(self._classes, list(range(self._num_classes)))
            )
        )
        self._path = '/home/hongsikkim/HDD/data/snngen'
        self._wd = os.getcwd()
        self._batch_size = batch_size
        self._image_ext = '.jpg'
        self._train_path = os.path.join(self._path, 'train')
        self._val_path = os.path.join(self._path, 'val')
        self.train_list = self._get_train_list()
        self.val_list = self._get_val_list()
        self._train_queue = list()
        self._val_queue = list()

    def _get_gt_classes(self):
        filename = os.path.join(self._gt_class_path, 'Imagenet_gt_classes.pkl')
        with open(filename, 'rb') as f:
            _classes = pickle.load(f)

        return _classes

    def _get_train_list(self):
        _savefile = os.path.join(self._wd, 'train_list.pkl')
        if os.path.exists(_savefile):
            print('Predefined train_list found. Loading...')
            with open(_savefile, 'rb') as fid:
                _train_list = pickle.load(fid)
            print('Done.')
            return _train_list

        _train_list = dict()
        _files = glob.glob(os.path.join(self._train_path, '*.xml'))
        for _file in _files:
            _filename = _file
            _img_name = os.path.join(_file.split('.')[0] + '.jpg')
            _train_list[_img_name] = self._load_annotation(_filename)
        print('Saving train_list for future use')
        with open(_savefile, 'wb') as fid:
            pickle.dump(_train_list, fid, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done')
        return _train_list

    def _get_val_list(self):
        _savefile = os.path.join(self._wd, 'val_list.pkl')
        if os.path.exists(_savefile):
            print('Predefined val_list found. Loading...')
            with open(_savefile, 'rb') as fid:
                _val_list = pickle.load(fid)
            print('Done.')
            return _val_list

        _val_list = dict()
        _files = glob.glob(os.path.join(self._val_path, '*.xml'))
        for _file in _files:
            _filename = _file
            _img_name = os.path.join(_file.split('.')[0] + '.jpg')
            _val_list[_img_name] = self._load_annotation(_filename)
        print('Saving train_list for future use')
        with open(_savefile, 'wb') as fid:
            pickle.dump(_val_list, fid, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done')
        return _val_list

    def get_next_train_batch(self):
        _batch = list()
        if len(self._train_queue) < self._batch_size:
            self._train_queue.extend(self.train_list.keys())
        random.shuffle(self._train_queue)
        for _ in range(self._batch_size):
            _batch.append(self._train_queue.pop())

        # print(_batch[0])
        # print(self.train_list[_batch[0]])
        return _batch

    def get_next_val_batch(self):
        _batch = list()
        if len(self._val_queue) < self._batch_size:
            self._val_queue.extend(self.val_list.keys())
        random.shuffle(self._val_queue)
        for _ in range(self._batch_size):
            _batch.append(self._val_queue.pop())

        # print(_batch[0])
        # print(self.val_list[_batch[0]])
        return _batch

    def _load_annotation(self, filename):
        tree = ET.parse(filename)
        obj = tree.find('object')
        label_txt = obj.find('contents').text
        label = []
        for _element in label_txt:
            label.append(self._class_to_ind[_element])
        label = np.array(label, dtype=np.int32)

        return {'label_txt': label_txt,
                'label': label
                }


if __name__ == '__main__':
    test = Textgen(batch_size=4)
    for i in range(5):
        test.get_next_train_batch()
