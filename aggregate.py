import json
from operator import itemgetter
from argparser import argparser
from aggregator import Aggregator
from random import shuffle, randint


def main(args):
    # seq_list = [os.path.join('%02d' % i) for i in range(0, 22)]

    aggregator = Aggregator()
    print('Num of ground truth labeled images %d\n\n' % len(aggregator.agg_list))

    aggregator.sort_agg_list()

    for i in range(0, 100):
        print('\nDAgger Iteration: %d\n' % aggregator.dag_it_num)
        aggregator.on_new_iter()
        train, val = aggregator.get_training_data()
        # aggregator.save_list(train, 'train')
        # aggregator.save_list(val, 'val')
        print('Length train set %d' % len(train))
        print('Length validation set %d' % len(val))
        if aggregator.images_evaluated == len(aggregator.agg_list):
            print('DAgger stopped!')
            break
        aggregator.aggregate()

    test = aggregator.agg_list[0:100]
    len_train_total = int(len(aggregator.agg_list[0:100]) * 0.1)
    train_batch_len = int(len_train_total * 0.8)
    val_batch_len = len_train_total - train_batch_len

    train_batch = test[0:train_batch_len]
    val_batch = test[train_batch_len:train_batch_len + val_batch_len]

    # shuffle worst images after evaluation that belong to training data in the next iteration
    # so they are added to training and validation with no order

    for j in range(0, 9):
        for i in range(len_train_total * j, len_train_total + len_train_total * j):
            test[i][aggregator.k_dag_it] = j
            test[i][aggregator.k_pr] = randint(0, 100) / 100
            test[i][aggregator.k_quota_g] = randint(5, 12) / 100.0
        aggregator.dag_it_num = j

    shuffle(test)

    test = aggregator.sort_agg_list(test)
    #
    # for i in range(len(test)):
    #         print('%d | %s' % (i, test[i]))


    test[10:100] = sorted(test[10:100], key = itemgetter(aggregator.k_dag_it), reverse = False)

    # aggregator.save_list(test)


if __name__ == "__main__":
    try:
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")
