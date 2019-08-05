from argparser import argparser
import cv2
import numpy as np
import os
from random import shuffle

from random import shuffle
def create_working_dir(args):
    path = args.base_dir + args.sequence + args.net_date
    try:
        os.mkdir(path)
    except OSError as e:
        print('OSError: %s', e)
    else:
        print("Successfully created the directory %s " % path)
    return path


def iou_for_semantic_class(target_color_channel, inferenced_color_channel):
    intersection = np.logical_and(target_color_channel, inferenced_color_channel)
    union = np.logical_or(target_color_channel, inferenced_color_channel)
    return np.sum(intersection) / np.sum(union)


def main(args):

    # image_name = os.path.join("%06d" % 1)  # 000001

    seq_list = []
    for i in range(0, 21):
        if not i == 8:
            seq_list.append(os.path.join('%02d' % i))

    gt_labels_list = []
    inf_labels_list = []

    for seq in seq_list:
        gt_base_path = args.base_dir + seq + '/' + args.gt_labels
        for gt_label in os.listdir(gt_base_path):
            gt_labels_list.append((gt_label, seq))
        inf_base_path = args.base_dir + seq + '/' + args.inf_labels
        for inf_label in os.listdir(inf_base_path):
            inf_labels_list.append((inf_label, seq))

    print(seq_list)
    print(len(gt_labels_list))
    print(len(inf_labels_list))

    shuffle(gt_labels_list)
    shuffle(inf_labels_list)

    train_list = gt_labels_list[:int(len(gt_labels_list)*0.1)]
    for i, pair in enumerate(train_list):
        seq = pair.__getitem__(1)
        img_name = pair.__getitem__(0)
        print(str(seq) + ' : ' + str(img_name))




    #
    # with open(args.base_dir + 'train_list.json', "w") as f:
    #     f.write("\n".join(map(str, train_list)))
    #
    # with open(args.base_dir + 'val_list.json', "w") as f:
    #     f.write("\n".join(map(str, val_list)))














if __name__ == "__main__":
    try :
        args = argparser()
        main(args)
    except KeyboardInterrupt:
        print("Finished")