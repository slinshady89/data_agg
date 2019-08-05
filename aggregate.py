from argparser import argparser
import cv2
import numpy as np
import os
from random import shuffle
import json


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

    gt_labels = {}
    gt_labels['images'] = []

    # for seq in seq_list:
    #     gt_base_path = args.base_dir + seq + '/' + args.gt_labels
    #     for gt_label in os.listdir(gt_base_path):
    #         gt_labels['images'].append({'name': gt_label, 'sequence': seq})
    #
    # print(seq_list)
    # print('Length of ground truth labeled images %d\n\n' % len(gt_labels['images']))
    #
    # shuffle(gt_labels['images'])
    #
    # with open(args.base_dir + 'train_list.json', "w") as f:
    #     json.dump(gt_labels, f)
    #
    # with open(args.base_dir + 'train_list.json', 'r') as file:
    #     file_list = json.load(file)
    #
    # for counter, item in enumerate(file_list['images']):
    #     if counter > 10:
    #         break
    #     print(item['name'])
    #     print(item['sequence'])
    #
    # batch1 = file_list['images'][:10]
    # print(batch1)
    #
    # for name in batch1:
    #     print(name['name'])

    # # use only the first 10% of the all images to train the net
    #train_list = gt_labels['images'][:int(len(gt_labels['images'])*0.0003)]

    # print(file_list["images"])


    gt_labels = os.listdir(args.base_dir + '08/' + args.gt_labels)
    inf_labels = os.listdir(args.base_dir + '08/20190730_170755/')

    r = g = b = 0.0

    for name in gt_labels:
        gt_img = cv2.imread(args.base_dir + '08/' + args.gt_labels + name)
        h, w, _ = gt_img.shape
        cropped_img = gt_img[(h - 256) :h, ((w - 1024) // 2) : (w - (w - 1024) // 2)]
        gt_img = cv2.resize(cropped_img, (1024, 256))
        inf_img = cv2.imread(args.base_dir + '08/20190730_170755/' + name)
        r += iou_for_semantic_class(gt_img[:, :, 0], inf_img[:, :, 0])
        g += iou_for_semantic_class(gt_img[:, :, 1], inf_img[:, :, 1])
        b += iou_for_semantic_class(gt_img[:, :, 2], inf_img[:, :, 2])
        print('\r\033[1A\033[0K %d of %d' % (int(name.replace('.png', '')), len(gt_labels)))

    r /= len(gt_labels)
    g /= len(gt_labels)
    b /= len(gt_labels)

    print(r, g, b)







if __name__ == "__main__":
    try :
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")

