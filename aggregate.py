from argparser import argparser
import os
from random import shuffle
import json
from operator import itemgetter
from aggregator import Aggregator


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



def main(args):
    seq_list = [os.path.join('%02d' % i) for i in range(0, 22)]

    gt_labels = []

    aggregator = Aggregator()
    print('Num of ground truth labeled images %d\n\n' % len(aggregator.img_list))

    train_batch_len = int(len(aggregator.img_list) * 0.1 * 0.8)
    val_batch_len = int(len(aggregator.img_list) * 0.1 * 0.2)

    train_batch = aggregator.img_list[0:train_batch_len]
    val_batch = aggregator.img_list[train_batch_len:train_batch_len + val_batch_len]

    print(len(train_batch))
    print(len(val_batch))
    print(len(aggregator.img_list[train_batch_len + val_batch_len:]))

    len_all = len(train_batch) + len(val_batch) + len(aggregator.img_list[train_batch_len + val_batch_len:])
    print('len all 3 together %d' % len_all)

    test = aggregator.img_list[0:10]

    for i in range(len(test)):
        test[i][aggregator.k_dag_it] = i

    shuffle(test)
    # for i in range(len(test)):
    #     print(test[i])

    test[3:10] = sorted(test[3:10], key = itemgetter(aggregator.k_dag_it), reverse = False)
    #
    # print('\n')
    # for i in range(len(test)):
    #     print(test[i])

    sorted_list = sorted(test, key = itemgetter(aggregator.k_dag_it), reverse = True)



if __name__ == "__main__":
    try:
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")
