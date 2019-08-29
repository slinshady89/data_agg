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

        if aggregator.dag_done:
            print('DAgger stopped!')
            break

        train, val, idx_eval = aggregator.get_training_data()

        num_max_threads = 40
        batch_size = (len(aggregator.agg_list[idx_eval:])) // num_max_threads

        # if batch_size == 0 a last learning process could be useful as far as enough data for training
        # is aggregated but unfortunately less than 40 images for evaluation should be considered
        # as a stopping condition
        if batch_size is 0:
            print('\nStopping DAgger in Iteration: %d\n' % aggregator.dag_it_num)
            print('\nOnly %d Images for Evaluation remaining.\n' % (len(aggregator.agg_list) - idx_eval))
            break
        print('Evaluating %d images in %d threads' % (batch_size, num_max_threads))

        aggregator.save_list(train, 'train')
        aggregator.save_list(val, 'val')
        aggregator.save_list(aggregator.agg_list[idx_eval:], 'eval')

        print(len(aggregator.agg_list[idx_eval:]))

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

            trainer = Trainer(_train_list = train, _val_list = val, _inf_list = aggregator.agg_list[idx_eval:])
            trainer.epoch_steps = 250
            trainer.val_steps = 50
            trainer.batch_size = 6
            trainer.n_epochs = 10
            trainer.dag_it = aggregator.dag_it_num
            # trains model for defined number of epochs with the actual dataset
            trainer.train()
            # safes inferences of images that are unseen by the net
            trainer.predict()
            session.close()
            # del train_list
            # del val_list
            # del trainer.model
            # del trainer.multi_model
            # gc.collect()

        print('\nInference done!\n')

        inf_dir = aggregator.dag_dir + '%02d/inf/' % aggregator.dag_it_num

        evaluator = Evaluator(aggregator.base_dir, inf_dir, aggregator.label_dir,
                              _eval_list = aggregator.agg_list)

        jobs = []
        q = multiprocessing.Queue()

        prq = np.zeros((len(inf_list), 3), dtype = np.float)
        l = 0
        for k in range(idx_eval, idx_eval + batch_size * num_max_threads, batch_size):
            print('Starting %d-th Process at Index %d with batch_size %d' % (l, k, batch_size))
            l += 1
            p = multiprocessing.Process(target = evaluator.process_batch, args = (q, k, batch_size))
            jobs.append([p, k])
            p.start()

        # evaluate the first num_max_threads * batch_size images with multiprocessing
        for l, job in enumerate(jobs):
            rets = q.get()
            try:
                for idx, ret in enumerate(rets[:, 1]):
                    aggregator.agg_list[job[1] + idx][aggregator.k_precision] = ret[0]
                    aggregator.agg_list[job[1] + idx][aggregator.k_recall] = ret[1]
                    aggregator.agg_list[job[1] + idx][aggregator.k_pr] = ret[0] * ret[1]
                    aggregator.agg_list[job[1] + idx][aggregator.k_quota_g] = ret[2]
                # prq[job[1]:job[1] + rets.shape[0]] = rets[:, 1, :]
            except ValueError as e:
                print(job[1])
                print(prq.shape[0] - job[1], rets.shape[0])

        for job in jobs:
            job[0].join()

        print('Evaluationg rest of images from %d to %d' % (idx_eval + num_max_threads * batch_size,
                                                            len(aggregator.agg_list)))
        # evaluate images that doesn't fit into multi processing batches
        for k in range(idx_eval + num_max_threads * batch_size, len(aggregator.agg_list)):
            chunk = evaluator.process_image(k)
            aggregator.agg_list[k][aggregator.k_precision] = chunk[1, 0]
            aggregator.agg_list[k][aggregator.k_recall] = chunk[1, 1]
            aggregator.agg_list[k][aggregator.k_pr] = chunk[1, 0] * chunk[1, 1]
            aggregator.agg_list[k][aggregator.k_quota_g] = chunk[1, 2]

        aggregator.save_list()
        aggregator.aggregate()


if __name__ == "__main__":
    try:
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")
