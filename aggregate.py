from argparser import argparser
import cv2
import numpy as np
import os
from random import shuffle
import json


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
    
    gt_labels = {}
    gt_labels[images] = []

    for seq in seq_list:
        gt_base_path = args.base_dir + seq + args.gt_labels
        for gt_label in os.listdir(gt_base_path):
            gt_labels[images].append({name: gt_label, sequence: seq})

    print(seq_list)
    print('Length of ground truth labeled images %d\n\n' % len(gt_labels[images]))

    shuffle(gt_labels[images])

    with open(args.base_dir + 'train_list.json', "w") as f:
        json.dump(gt_labels, f)

    create_working_dir(args, 'Data/')
    label_path = create_working_dir(args, 'Data/Labels/')
    image_path = create_working_dir(args, 'Data/Images/')

    print('\nSaving labels to:\n' + label_path)
    print('\nSaving rgb images to:\n' + image_path)
    print('\n')

    for i, item in enumerate(gt_labels[images]):
        # if i > 2:
        #     break
        label = cv2.imread(args.base_dir + item[sequence] + args.gt_labels + item[name])
        img = cv2.imread(args.base_dir + item[sequence] + args.images + item[name])
        l_status = False
        while not l_status:
            l_status = cv2.imwrite(label_path + item[sequence] + '_' + item[name], label)
        i_status = False
        while not i_status:
            i_status = cv2.imwrite(image_path + item[sequence] + '_' + item[name], img)

        print('\r\033[1A\033[0K %d of %d' % (i + 1, len(gt_labels[images])))


if __name__ == "__main__":
    try :
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")

