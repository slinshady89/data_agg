import numpy as np
import cv2
import multiprocessing
from diagnostics import preprocess_inference, recall_rgb, precision_rgb, color_quota_rgb
from aggregator import Keys


class Evaluator(object):
    def __init__(self, _base_dir, _inf_label_dir, _gt_label_dir, _agg_list):
        self.base_dir_ = _base_dir
        self.inf_label_dir_ = _inf_label_dir
        self.gt_label_dir_ = _gt_label_dir
        self.agg_list_ = _agg_list
        # threshold's estimated via the PoR-Curve
        self.threshold_ = np.array((69, 75, 110), dtype = np.int)
        self.keys_ = Keys()
        self.dag_it = 0
        self.num_max_threads = multiprocessing.cpu_count()
        self.batch_size = 0
        self.stop_dagger = False

    # subprocess of on image in evaluation. send back precision, recall, quota for all 3 channels
    def process_image(self, i):
        img_name = self.base_dir_ + self.inf_label_dir_ + self.agg_list_[i][self.keys_.name]
        inf_label = cv2.imread(img_name)
        gt_label = cv2.imread(self.base_dir_ + self.gt_label_dir_ + self.agg_list_[i][self.keys_.name])
        # print(i)
        if inf_label is None:
            print(i, self.agg_list_[i][self.keys_.name])
            return 0
        inf_label = cv2.resize(inf_label, (1024, 256))

        h, w, _ = gt_label.shape
        if h > 256 and w > 1024:
            gt_label = gt_label[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        else:
            gt_label = gt_label
        gt_label = cv2.resize(gt_label, (1024, 256))

        inf_label_proc = preprocess_inference(inf = inf_label, threshold = self.threshold_)

        quota_gt = color_quota_rgb(gt_label)
        recall = recall_rgb(gt_label, inf_label_proc)
        precision = precision_rgb(gt_label, inf_label_proc)

        prec_rec_qut = np.array([[precision[0], recall[0], quota_gt[0]],  # blue channel
                                 [precision[1], recall[1], quota_gt[1]],  # green channel
                                 [precision[2], recall[2], quota_gt[2]]])  # red channel
        return prec_rec_qut

    # subprocessing a batch of images
    def process_batch(self, q, begin, batch_size):
        if begin + batch_size >= len(self.agg_list_):
            print(begin)
            print(batch_size)
            batch_size = len(self.agg_list_) - begin - 1
        prq = np.zeros((batch_size, 3, 3), dtype = np.float)
        for i in range(begin, begin + batch_size):
            prq[i - begin] = self.process_image(i)
        q.put(prq)

    # processes the data separated onto multiple threads
    def process_prediction(self, agg_chunk, idx_eval):
        jobs = []
        q = multiprocessing.Queue()
        for k in range(idx_eval, idx_eval + self.batch_size * self.num_max_threads, self.batch_size):
            p = multiprocessing.Process(target = self.process_batch, args = (q, k, self.batch_size))
            jobs.append([p, k])
            p.start()
        # evaluate the first num_max_threads * batch_size images with multiprocessing
        for l, job in enumerate(jobs):
            batch = q.get()
            try:
                for batch_idx, prec_rec_quot in enumerate(batch[:, 1]):
                    agg_chunk[job[1] + batch_idx][self.keys_.precision] = prec_rec_quot[0]
                    agg_chunk[job[1] + batch_idx][self.keys_.recall] = prec_rec_quot[1]
                    agg_chunk[job[1] + batch_idx][self.keys_.f1score] = prec_rec_quot[0] * prec_rec_quot[1]
                    agg_chunk[job[1] + batch_idx][self.keys_.quota_g] = prec_rec_quot[2]
            except ValueError as e:
                print(job[1])
        for job in jobs:
            job[0].join()
        print('Evaluationg rest of images from %d to %d' % (idx_eval + self.num_max_threads * self.batch_size,
                                                            len(agg_chunk)))
        # evaluate images that doesn't fit into multi processing batches
        for k in range(idx_eval + self.num_max_threads * self.batch_size, len(agg_chunk)):
            chunk = self.process_image(k)
            precision = chunk[1, 0]
            recall = chunk[1, 1]
            agg_chunk[k][self.keys_.precision] = precision
            agg_chunk[k][self.keys_.recall] = recall
            agg_chunk[k][self.keys_.f1score] = 2.0 * precision * recall / (precision + recall)
            agg_chunk[k][self.keys_.quota_g] = chunk[1, 2]

        return agg_chunk

    def estimate_batch_size(self, len_inference):
        self.batch_size = len_inference // self.num_max_threads
        # if batch_size == 0 a last learning process could be useful as far as enough data for training
        # is aggregated but unfortunately less than 40 images for evaluation should be considered
        # as a stopping condition
        if self.batch_size is 0:
            print('\nStopping DAgger in Iteration: %d\n' % self.dag_it)
            print('\nOnly %d Images for Evaluation remaining.\n' % len_inference)
            self.stop_dagger = True
