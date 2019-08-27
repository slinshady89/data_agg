import multiprocessing
import time
import numpy as np

from argparser import argparser
from aggregator import Aggregator
from trainer import Trainer
from evaluator import Evaluator


def main(args):
    aggregator = Aggregator()
    print('Num of ground truth labeled images %d\n\n' % len(aggregator.agg_list))

    for i in range(0, 100):
        print('\nDAgger Iteration: %d\n' % aggregator.dag_it_num)
        aggregator.on_new_iter()
        train, val, idx_eval = aggregator.get_training_data()

        num_max_threads = 40
        batch_size = (len(aggregator.agg_list) - len(train) - len(val)) // num_max_threads
        print('Evaluating %d images in %d threads' % (batch_size, num_max_threads))

        aggregator.save_list(train, 'train')
        aggregator.save_list(val, 'val')
        aggregator.save_list(aggregator.agg_list[idx_eval:], 'eval')

        if aggregator.dag_done:
            print('DAgger stopped!')
            break

        train_list = [elem[aggregator.k_img_name] for elem in train]
        val_list = [elem[aggregator.k_img_name] for elem in val]
        inf_list = [elem[aggregator.k_img_name] for elem in aggregator.agg_list[idx_eval:]]

        trainer = Trainer(_train_list = train_list, _val_list = val_list, _inf_list = inf_list)
        trainer.epoch_steps = 100
        trainer.val_steps = 25
        trainer.n_epochs = 1
        # trains model for defined number of epochs with the actual dataset
        trainer.train()
        # safes inferences of images that are unseen by the net
        trainer.predict()

        inf_dir = aggregator.dag_dir + '%02d/inf/' % aggregator.dag_it_num
        evaluator = Evaluator(aggregator.base_dir, inf_dir, aggregator.label_dir, _eval_list = inf_list)

        jobs = []
        q = multiprocessing.Queue()

        prq = np.zeros((len(inf_list), 3, 3), dtype = np.float)
        for k in range(0, len(inf_list), batch_size):
            p = multiprocessing.Process(target = evaluator.process_batch, args = (q, k, batch_size))
            # if i == 0:
            #     jobs.append([p, i])
            # else:
            jobs.append([p, i])
            p.start()

        for l, job in enumerate(jobs):
            ret = q.get()
            prq[job[1]:job[1] + len(ret)] = ret

        for job in jobs:
            job[0].join()

        for k in range(len(prq)):
            aggregator.agg_list[idx_eval + k][aggregator.k_precision] = prq[k, 1, 0]
            aggregator.agg_list[idx_eval + k][aggregator.k_recall] = prq[k, 1, 1]
            aggregator.agg_list[idx_eval + k][aggregator.k_quota_g] = prq[k, 1, 2]
            aggregator.agg_list[idx_eval + k][aggregator.k_pr] = prq[k, 1, 0] * prq[k, 1, 1]
            print(aggregator.agg_list[idx_eval + k])

        aggregator.aggregate()
        aggregator.save_list()


if __name__ == "__main__":
    try:
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")
