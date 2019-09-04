import os
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
        inf_dir = aggregator.dag_dir + '%02d/inf/' % aggregator.dag_it_num
        evaluator = Evaluator(aggregator.base_dir, inf_dir, aggregator.label_dir,
                              _agg_list = aggregator.agg_list)

        evaluator.estimate_batch_size(len(aggregator.agg_list[idx_eval:]))

        print('Evaluating %d images in %d threads' % (evaluator.batch_size, evaluator.num_max_threads))

        if aggregator.dag_done or evaluator.stop_dagger:
            print('DAgger stopped!')
            break

        aggregator.save_list(train, 'train')
        aggregator.save_list(val, 'val')
        aggregator.save_list(aggregator.agg_list[idx_eval:], 'eval')

        # Training and Prediction

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        with tf.Graph().as_default():
            session = tf.Session('')
            KTF.set_session(session)
            KTF.set_learning_phase(1)

            trainer = Trainer(_train_list = train,
                              _val_list = val,
                              _inf_list = aggregator.agg_list[idx_eval:])
            trainer.epoch_steps = 250
            trainer.val_steps = 50
            trainer.batch_size = 6
            trainer.n_epochs = 25
            trainer.dag_it = aggregator.dag_it_num
            # trains model for defined number of epochs with the actual dataset
            trainer.train()
            # safes inferences of images that are unseen by the net
            trainer.predict()
            session.close()

        print('\nInference done!\n')

        # Training and prediction done

        # Evaluation

        aggregator.agg_list = evaluator.process_prediction(agg_chunk = aggregator.agg_list,
                                                           idx_eval = idx_eval)

        # Evaluation done and saved for next iteration

        aggregator.save_list()
        aggregator.aggregate()





if __name__ == "__main__":
    try:
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\n Cancelled by user. \n\nGoodbye!")
