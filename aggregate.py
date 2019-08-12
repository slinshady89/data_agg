from argparser import argparser
import cv2
import numpy as np
import os
from random import shuffle
import json
from operator import itemgetter


def create_working_dir(args, folder_name):
    path = args.base_dir + folder_name
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno == 17:
            print(path)
            print('Directory exists already! Everything fine!')
        else:
            print('OSError: %s', e)
    else:
        print("Successfully created the directory %s " % path)
    return path


def main(args):
    # image_name = os.path.join("%06d" % 1)  # 000001

    seq_list = []
    for i in range(0, 21):
        if not i == 8:
            seq_list.append(os.path.join('%02d' % i))

    images = 'images'
    name = 'name'
    sequence = 'sequence'
    iou_green = 'iou_green'

    gt_labels = dict()
    gt_labels[images] = []

    for seq in seq_list:
        gt_base_path = args.base_dir + seq + args.gt_labels
        for gt_label in os.listdir(gt_base_path):
            gt_labels[images].append({name: gt_label, sequence: seq, iou_green: 0.00})

    print(seq_list)
    print('Length of ground truth labeled images %d\n\n' % len(gt_labels[images]))

    shuffle(gt_labels[images])

    test_list = dict()
    test_list[images] = []

    test_list[images] = gt_labels[images][0:20]

    with open(args.base_dir + 'Data/test_list.json', "w") as f:
        json.dump(test_list, f)

    with open(args.base_dir + 'Data/test_list.json', "r") as f:
        test_read = json.load(f)

    shuffle(test_read[images])

    for i, label in enumerate(test_read[images]):
        if i >= 10:
            break
        label[iou_green] = 0.1

    with open(args.base_dir + 'Data/test_read.json', "w") as f:
        json.dump(test_read, f)

    count = 0
    for i in range(0, len(test_list[images]) - 1):
        for j in range(0, len(test_read[images]) - 1):
            # if test_read[images][j][iou_green] > 0:
            if (test_list[images][i][name] == test_read[images][j][name]) and \
                    (test_list[images][i][sequence] == test_read[images][j][sequence]):
                # print('\nFound match at %d and %d:' % (i, j))
                # print(test_list[images][i], test_read[images][j])
                count += 1

    print('\nFound %d matches\n' % count)

    sorted_list = sorted(test_read[images], key = itemgetter(iou_green), reverse = True)

    ix = np.random.choice(np.arange(len(sorted_list)), 8)
    print(len(sorted_list))
    print(np.arange(len(sorted_list)))

    for i in range(0, len(sorted_list) - 1):
        print(sorted_list[i])


if __name__ == "__main__":
    try:
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")
