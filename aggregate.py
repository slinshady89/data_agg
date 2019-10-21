import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from argparser import argparser
from aggregator import Aggregator
from trainer import Trainer
from evaluator import Evaluator


def main(args):
    aggregator = Aggregator(_base_dir = args.base_dir,
                            _img_dir = args.img_dir,
                            _label_dir = args.label_dir,
                            _inf_dir = args.inf_dir,
                            _dag_dir = args.dag_dir,
                            _poses_dir = args.poses_dir)
    print('Num of ground truth labeled images %d\n\n' % len(aggregator.agg_list))

    for i in range(0, 100):
        print('\nDAgger Iteration: %d\n' % aggregator.dag_it_num)
        # creates new directories each iteration
        aggregator.on_new_iter()
        # returns the training, validation lists and knowledge of index at which index evaluation in agg_list starts
        train, val, idx_eval = aggregator.get_training_data()
        # set directory to save predictions of inference
        inf_dir = aggregator.dag_dir + '%02d/inf/' % aggregator.dag_it_num
        # initiates the evaluator
        evaluator = Evaluator(aggregator.base_dir, inf_dir, aggregator.label_dir,
                              _agg_list = aggregator.agg_list)
        # estimating a batch each process should evaluate later to don't exceed a given process number
        evaluator.estimate_batch_size(len(aggregator.agg_list[idx_eval:]))

        print('Evaluating %d images in %d threads' % (evaluator.batch_size, evaluator.num_max_threads))

        if aggregator.dag_done or evaluator.stop_dagger:
            aggregator.save_list()

            print('DAgger stopped!')
            break

        aggregator.save_list(train, 'train')
        aggregator.save_list(val, 'val')

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
            trainer.batch_size = 8
            trainer.epoch_steps = len(train) // trainer.batch_size
            trainer.val_steps = len(val) // trainer.batch_size
            trainer.n_epochs = 25
            trainer.dag_it = aggregator.dag_it_num
            trainer.update_callback()
            # trains model for defined number of epochs with the actual dataset
            trainer.train()
            print('\nTraining done!\nStarting Prediction\n')
            # safes inferences of images that are unseen by the net
            trainer.predict()
            session.close()

        print('\nInference done!\n')
        print('Evaluating %d images' % len(aggregator.agg_list[idx_eval:]))
        # Training and prediction done

        # Evaluation

        aggregator.agg_list = evaluator.process_prediction(agg_chunk = aggregator.agg_list,
                                                           idx_eval = idx_eval)
        print('Evaluation done. Saving evaluated data.')
        aggregator.save_list(aggregator.agg_list[idx_eval:], 'eval')
        # Evaluation done and saved for next iteration

        # save full aggregation list with all information of all iterations until this in iteration's folder
        aggregator.save_list()
        # delete all images of inference step to save space on the drive
        aggregator.delete_inf()
        aggregator.prepare_next_it()


if __name__ == "__main__":
    try:
        args = argparser()
        main(args)
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\nCancelled by user. \n\nGoodbye!")
