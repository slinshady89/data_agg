import json
from operator import itemgetter
from argparser import argparser
from aggregator import Aggregator
from random import shuffle, randint


def main(args):
    # seq_list = [os.path.join('%02d' % i) for i in range(0, 22)]

    aggregator = Aggregator()
    print('Num of ground truth labeled images %d\n\n' % len(aggregator.agg_list))

    for i in range(0, 100):
        print('\nDAgger Iteration: %d\n' % aggregator.dag_it_num)
        aggregator.on_new_iter()
        train, val = aggregator.get_training_data()
        # aggregator.save_list(train, 'train')
        # aggregator.save_list(val, 'val')
        print('Length train set %d' % len(train))
        print('Length validation set %d' % len(val))
        if aggregator.dag_done:
            print('DAgger stopped!')
            break
        aggregator.aggregate()


if __name__ == "__main__":
    try:
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")
