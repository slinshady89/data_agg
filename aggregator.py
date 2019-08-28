import json
import os
from math import inf
from random import shuffle, randint
from operator import itemgetter


class Aggregator(object):
    def __init__(self):
        self.dag_it_num = 0
        self.dag_done = False
        self.base_dir = '/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/'
        self.img_dir = 'images/'
        self.label_dir = 'labels/'
        self.inf_dir = 'inf/'
        self.dag_dir = 'dagger/'
        self.k_img_name = 'name'
        self.k_sequence = 'seq'
        self.k_precision = 'prec'
        self.k_recall = 'rec'
        self.k_quota_g = 'quota_green_gt'
        # evaluation metric
        self.k_pr = 'precision_times_recall'
        self.k_dag_it = 'dagger_iteration'
        # self.k_training = 'training_data'
        # self.k_valid = 'validation_data'
        self.train_perc = 0.8
        self.val_perc = 0.2
        self.train_batch_perc = 0.1  # 0.0005
        self.agg_batch_perc = self.train_batch_perc / 2
        self.agg_list_name = self.base_dir + self.dag_dir + 'agg_list.json'
        self.agg_list = []
        # add a dictionary entry for every label image in label_dir with initial values for evaluation metrics
        # and key values if they belong to training or validation data of which DAgger iteration
        # training and validation data accumulates older iterations
        # images with green quota lower than 0.07 aren't further processed
        for gt_label in os.listdir(self.base_dir + self.label_dir):
            self.agg_list.append({self.k_img_name: gt_label, self.k_precision: -1.0, self.k_recall: -1.0,
                                  self.k_pr: -1.0, self.k_dag_it: 200,
                                  self.k_quota_g: -1.0})

        self.len_agg_list = len(self.agg_list)
        shuffle(self.agg_list)
        self.len_train_batch = int(self.len_agg_list * self.train_batch_perc * self.train_perc)
        self.len_val_batch = int(self.len_agg_list * self.train_batch_perc * self.val_perc)
        self.len_agg_train = self.len_train_batch // 2
        self.len_agg_val = self.len_val_batch // 2
        self.len_train_set = self.len_train_batch + self.len_val_batch
        self.len_agg_set = self.len_agg_train + self.len_agg_val
        self.num_imgs_to_train = self.len_train_set + self.len_agg_set * self.dag_it_num

        with open(self.agg_list_name, "w") as f:
            json.dump(self.agg_list, f)
        f.close()
        for i in range(self.len_train_set):
            self.agg_list[i][self.k_dag_it] = 0

        self.train_data_total = self.len_train_batch + self.len_val_batch
        self.agg_data_total = self.len_agg_train + self.len_agg_val
        self.on_new_iter()
        self.images_evaluated = 0

    def create_working_dir(self, folder_name):
        path = self.base_dir + folder_name
        try:
            os.mkdir(path)
        except OSError as e:
            if e.errno != 17:
                print('OSError: %s', e)
        return path

    def on_new_iter(self):
        self.create_working_dir(os.path.join(self.dag_dir))
        self.create_working_dir(os.path.join(self.dag_dir + '%02d' % self.dag_it_num))
        self.create_working_dir(os.path.join(self.dag_dir + '%02d/inf/' % self.dag_it_num))
        self.agg_list_name = self.base_dir + self.dag_dir + '%02d/' % self.dag_it_num \
                             + 'agg_list_%02d.json' % self.dag_it_num
        self.num_imgs_to_train = self.len_train_batch + self.len_val_batch + \
                                 (self.len_agg_train + self.len_agg_val) * self.dag_it_num

    def aggregate(self):
        self.dag_it_num += 1
        pass

    def sort_agg_list(self, list_to_sort = None):
        if list_to_sort is None:
            # self.agg_list = sorted(sorted(self.agg_list, key = itemgetter(self.k_dag_it), reverse = False),
            #                        key = itemgetter(self.k_pr), reverse = False)
            self.agg_list = sorted(self.agg_list, key = itemgetter(self.k_dag_it, self.k_pr), reverse = False)
            self.save_list()
        else:
            return sorted(list_to_sort, key = itemgetter(self.k_dag_it, self.k_pr), reverse = False)

    def get_training_data(self):
        # shuffle data
        shuffle(self.agg_list)
        self.sort_agg_list()
        # adding training data of last dagger iterations to training list
        if self.dag_it_num > 1:
            train = self.agg_list[:self.len_train_set + self.len_agg_set * (self.dag_it_num - 1)]
            num_images_found = len(train)
        else:
            train = self.agg_list[:self.len_train_set]
            num_images_found = len(train)
        # training worst inference results to trainings list for the next dagger iteration
        stopping_idx = 0
        for i in range(num_images_found,
                       len(self.agg_list)):
            # aggregate worst data of inference of last iteration
            if self.agg_list[i][self.k_quota_g] >= 0.07 and num_images_found < self.num_imgs_to_train:
                self.agg_list[i][self.k_dag_it] = self.dag_it_num
                train.append(self.agg_list[i])
                num_images_found += 1
            # stop data aggregation if num of needed data for this dagger iteration is achieved
            if num_images_found >= self.num_imgs_to_train:
                print('%d images for next DAgger Iteration found at index %d of total images %d\n\n'
                      % (self.num_imgs_to_train, i, len(self.agg_list)))
                stopping_idx = i
                break
            if i == len(self.agg_list)-1:
                print('Stopping DAgger because no new Data could be aggregated.\nCreate more!')
                self.dag_done = True
                # is this break really necessary since if i == len(arr) the for-loop ends anyway?
                # at all it doesn't hurt to highlight it as a stopping condition
                break
        # shuffle this list ans split into training and validation
        shuffle(train)
        print(i, stopping_idx)
        return train[:int(len(train) * self.train_perc)], train[int(len(train) * self.train_perc):], i

    def save_list(self, list_to_save = None, name = None):
        if list_to_save is None:
            with open(self.agg_list_name, "w") as f:
                json.dump(self.agg_list, f)
        else:
            path = self.base_dir + self.dag_dir + '%02d/' % self.dag_it_num + name + '_%02d' % self.dag_it_num + '.json'
            with open(path, "w") as f:
                json.dump(list_to_save, f)
        f.close()

    def load_list(self, list_to_load = None, name = None):
        if list_to_load is None:
            with open(self.agg_list_name, "r") as f:
                list_chunk = json.load(f)
        else:
            path = self.base_dir + self.dag_dir + '%02d/' % self.dag_it_num + name + '_%02d' % self.dag_it_num + '.json'
            with open(path, "w") as f:
                list_chunk = json.load(f)
        f.close()
        return list_chunk
