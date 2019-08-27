import numpy as np
import cv2

from diagnostics import preprocess_inference, recall_rgb, precision_rgb, colour_quota_rgb


## Evaluator for inferences of KITTI data structure
# sequences/XX/
# ---|image_2/xxxxxx.png
# ---|inference/xxxxxx.png
# ---|gt_labels/xxxxxx.png
class Evaluator(object):
    def __init__(self, _base_dir, _inf_label_dir, _gt_label_dir, _eval_list):
        self.base_dir_ = _base_dir
        self.inf_label_dir_ = _inf_label_dir
        self.gt_label_dir_ = _gt_label_dir
        self.eval_list = _eval_list
        self.threshold_ = np.array((69, 75, 110), dtype = np.int)

    def process_image(self, i):
        img_name = self.base_dir_ + self.inf_label_dir_ + self.eval_list[i]
        inf_label = cv2.imread(img_name)
        gt_label = cv2.imread(self.base_dir_ + self.gt_label_dir_ + self.eval_list[i])
        inf_label = cv2.resize(inf_label, (1024, 256))

        h, w, _ = gt_label.shape
        if h > 256 and w > 1024:
            gt_label = gt_label[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        else:
            gt_label = gt_label
        gt_label = cv2.resize(gt_label, (1024, 256))

        inf_label_proc = preprocess_inference(inf = inf_label, threshold = self.threshold_)

        quota_gt = colour_quota_rgb(gt_label)
        recall = recall_rgb(gt_label, inf_label_proc)
        precision = precision_rgb(gt_label, inf_label_proc)

        prec_rec_qut = [[precision[0], recall[0], quota_gt[0]],  # blue channel
                        [precision[1], recall[1], quota_gt[1]],  # green channel
                        [precision[2], recall[2], quota_gt[2]]]  # red channel
        return prec_rec_qut

    def process_batch(self, q, begin, batch_size):
        if begin + batch_size > len(self.eval_list):
            print(begin)
            print(batch_size)
            batch_size = len(self.eval_list) - begin
        prq = np.zeros((batch_size, 3, 3), dtype = np.float)
        for i in range(begin, begin + batch_size):
            prq[i - begin] = self.process_image(i)
        q.put(prq)

