import multiprocessing
import os
import gc
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

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

        print('len inf list %d' % len(inf_list))
        print('batch * threads %d' % (batch_size * num_max_threads))


        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        with tf.Graph().as_default():
            session = tf.Session('')
            KTF.set_session(session)
            KTF.set_learning_phase(1)

            trainer = Trainer(_train_list = train_list, _val_list = val_list, _inf_list = inf_list)
            trainer.epoch_steps = 100
            trainer.val_steps = 25
            trainer.n_epochs = 10
            trainer.dag_it = aggregator.dag_it_num
            # trains model for defined number of epochs with the actual dataset
            trainer.train()
            # safes inferences of images that are unseen by the net
            trainer.predict()
            session.close()
            del train_list
            del val_list
            del trainer.model
            del trainer.multi_model
            gc.collect()

        print('\nInference done!\n')

        inf_dir = aggregator.dag_dir + '%02d/inf/' % aggregator.dag_it_num

        evaluator = Evaluator(aggregator.base_dir, inf_dir, aggregator.label_dir,
                              _eval_list = inf_list)

        jobs = []
        q = multiprocessing.Queue()

        prq = np.zeros((len(inf_list), 3, 3), dtype = np.float)
        for k in range(0, len(evaluator.eval_list), batch_size):
            p = multiprocessing.Process(target = evaluator.process_batch, args = (q, k, batch_size))
            jobs.append([p, k])
            p.start()

        print('len inf list')
        print(len(inf_list))

        for l, job in enumerate(jobs):
            ret = q.get()
            if prq[job[1]:job[1] + ret.shape[0]].shape[0] is ret.shape[0]:
                prq[job[1]:job[1] + ret.shape[0]] = ret
            else:
                print(job[1])
                for i in range(len(ret)):
                    print(ret[i])


        for job in jobs:
            job[0].join()



        # for k in range(len(inf_list)):
        #     print('%d | %.2f : %.2f : %.2f' % (k, prq[k, 1, 0], prq[k, 1, 1], prq[k, 1, 2]))

        for k in range(len(prq)):
            aggregator.agg_list[idx_eval + k - 1][aggregator.k_precision] = prq[k, 1, 0]
            aggregator.agg_list[idx_eval + k - 1][aggregator.k_recall] = prq[k, 1, 1]
            aggregator.agg_list[idx_eval + k - 1][aggregator.k_quota_g] = prq[k, 1, 2]
            aggregator.agg_list[idx_eval + k - 1][aggregator.k_pr] = prq[k, 1, 0] * prq[k, 1, 1]

        # for k in range(int(len(aggregator.agg_list) * 0.9), len(aggregator.agg_list)):
        #     print('%d | %s' % (k, aggregator.agg_list[k]))

        aggregator.aggregate()
        aggregator.save_list()


if __name__ == "__main__":
    try:
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")
