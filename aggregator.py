
import os
from random import shuffle



class Aggregator(object):
    def __init__(self):
        self.dag_it_num = 0
        self.base_dir = '/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/'
        self.img_dir = 'images/'
        self.label_dir = 'labels/'
        self.inf_dir = 'inference/'
        self.dag_dir = 'dagger/'
        self.k_img_name = 'name'
        self.k_sequence = 'seq'
        self.k_precision = 'prec'
        self.k_recall = 'rec'
        self.k_trained = 'trained'
        self.k_pr = 'precision_times_recall'
        self.k_dag_it = 'dagger_iteration'
        self.k_training = 'training_data'
        self.k_valid = 'validation_data'
        self.img_list = []

        for gt_label in os.listdir(self.base_dir + self.label_dir):
            self.img_list.append({self.k_img_name: gt_label, self.k_precision: -1.0, self.k_recall: -1.0,
                                 self.k_training: False, self.k_valid: False, self.k_pr: 2.0, self.k_dag_it: -1})
        shuffle(self.img_list)

    def create_working_dir(self, folder_name):
        path = self.base_dir + folder_name
        try:
            os.mkdir(path)
        except OSError as e:
            if e.errno != 17:
                print('OSError: %s', e)
        else:
            print("Successfully created the directory %s " % path)
        return path

    def on_new_iter(self):
        self.create_working_dir(os.path.join('%02d/' % self.dag_it_num))
        self.create_working_dir(os.path.join('%02d/inf/' % self.dag_it_num))
        label_path = self.base_dir + self.label_dir
        # for label_name in os.listdir(label_path):
        self.k_dag_it += 1
