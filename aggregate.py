from argparser import argparser
import cv2
import numpy as np
import os
from random import shuffle
import json
from operator import itemgetter
import collections


def create_working_dir(args, folder_name):
    path = args.base_dir + folder_name
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != 17:
            print('OSError: %s', e)
    else:
        print("Successfully created the directory %s " % path)
    return path


class Aggregator(object):
    def __init__(self):
        self.dag_it_num = 0
        self.base_dir = '/media/localadmin/BigBerta/11Nils/kitti/dataset/Data/'
        self.img_dir = 'images/'
        self.label_dir = 'labels/'
        self.inf_dir = 'inference/'
        self.dag_dir = 'dagger/'
        self.img_name = 'name'
        self.sequence = 'seq'
        self.precision = 'prec'
        self.recall = 'rec'
        self.trained = 'trained'
        self.pr = 'precision_times_recall'
        self.dag_it = 'dagger_iteration'
        self.img_list = []

    def create_working_dir(self, folder_name):
        path = args.base_dir + folder_name
        try:
            os.mkdir(path)
        except OSError as e:
            if e.errno != 17:
                print('OSError: %s', e)
        else:
            print("Successfully created the directory %s " % path)
        return path

    def on_new_iter(self):
        self.create_working_dir(os.path.join('%02d/' % self.dag_it_num))
        self.create_working_dir(os.path.join('%02d/inf/' % self.dag_it_num))
        label_path = self.base_dir + self.label_dir
        # for label_name in os.listdir(label_path):
        self.dag_it += 1


def main(args):
    # image_name = os.path.join("%06d" % 1)  # 000001

    seq_list = [os.path.join('%02d' % i) for i in range(0, 22)]

    images = 'images'
    name = 'name'
    sequence = 'sequence'
    prec_green = 'prec_green'
    rec_green = 'rec_green'
    training = 'trained'
    valid = 'valid'
    pr = 'precision_times_recall'
    dag_it = 'dagger_iteration'

    gt_labels = []

    gt_base_path = args.base_dir + args.gt_labels
    print(gt_base_path)
    for gt_label in os.listdir(gt_base_path):
        gt_labels.append({name: gt_label, prec_green: -1.0, rec_green: -1.0,
                          training: False, valid: False, pr: 2.0, dag_it: -1})

    test = gt_labels[0:10]

    sorted_list = sorted(test, key = itemgetter(name), reverse = True)
    print(sorted_list)

    print('test')
    print(seq_list)
    print('Length of ground truth labeled images %d\n\n' % len(gt_labels))



if __name__ == "__main__":
    try:
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")
