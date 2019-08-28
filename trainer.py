import cv2
import os
import numpy as np
from model import unet
from keras.preprocessing.image import img_to_array
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras.backend as K


class Trainer(object):
    def __init__(self, _train_list, _val_list, _inf_list, _dag_it = 0, _input_shape = (256, 1024, 3),
                 _train_steps = 500, _val_steps = 200, _num_epochs = 15, _batch_size = 4):
        self.dag_it = _dag_it
        self.train_list = _train_list
        self.val_list = _val_list
        self.inf_list = _inf_list
        self.base_dir = '/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/'
        self.img_dir = 'images/'
        self.label_dir = 'labels/'
        self.inf_dir = 'inf/'
        self.dag_dir = 'dagger/'
        self.log_dir = 'log/'
        self.optimizer = 'adagrad'
        self.gpu_num = '0'  # '1'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_num
        self.untrained = 'store_true'
        self.loss = 'categorical_crossentropy'
        self.output_mode = 'softmax'
        self.pool_size = (2, 2)
        self.kernel = 3
        self.input_shape = _input_shape  # (128, 512, 3)
        self.n_labels = 3  # num classes
        self.val_steps = _val_steps
        self.epoch_steps = _train_steps
        self.n_epochs = _num_epochs
        self.batch_size = _batch_size
        self.filters = 8
        self.model = unet(self.input_shape, self.n_labels, self.filters, self.kernel, self.pool_size, self.output_mode)
        # print(self.model.summary())
        if len(self.gpu_num) >= 2:
            self.multi_model = multi_gpu_model(self.model, gpus = len(self.gpu_num) - 2)
        else:
            self.multi_model = self.model
        self.multi_model.compile(loss = self.loss, optimizer = self.optimizer, metrics = ['accuracy'])

        self.std = [0.32636853, 0.31895106, 0.30716496]
        self.mean = [0.39061851, 0.38151629, 0.3547171]

        self.path = self.base_dir + self.dag_dir + '%02d/' % self.dag_it + self.log_dir
        print('saving weights in %s' % self.path)
        # set callbacks
        self.cp_cb = ModelCheckpoint(
            filepath = self.path + '/weights{epoch:02d}.hdf5',
            # filepath = path + '/weights{val_loss:02d}.hdf5',
            monitor = 'val_loss',
            verbose = 1,
            save_best_only = True,
            mode = 'auto',
            period = 1)
        self.es_cb = EarlyStopping(
            monitor = 'val_loss',
            patience = 3,
            verbose = 1,
            mode = 'auto')
        self.tb_cb = TensorBoard(
            log_dir = self.path,
            write_images = True)

    def iou_loss_core(self, true, pred):  # this can be used as a loss if you make it negative
        intersection = true * pred
        notTrue = 1 - true
        union = true + (notTrue * pred)
        return (K.sum(intersection, axis = -1) + K.epsilon()) / (K.sum(union, axis = -1) + K.epsilon())

    def iou_loss(self, y_true, y_pred, channel = 0, smooth = 1.0):
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        The jaccard distance loss is usefull for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
        gradient.

        Ref: https://en.wikipedia.org/wiki/Jaccard_index

        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        intersection = K.sum(K.abs(y_true[:, :, :, channel] * y_pred[:, :, :, channel]), axis = -1)
        sum_ = K.sum(K.abs(y_true[:, :, :, channel]) + K.abs(y_pred[:, :, :, channel]), axis = -1)
        jac = intersection / sum_  # (sum_ - intersection)
        return (1 - jac) * smooth

    def iou_metric(self, y_true, y_pred, channel = 0):
        intersection = K.sum(K.abs(y_true[:, :, :, channel] * y_pred[:, :, :, channel]), axis = -1)
        sum_ = K.sum(K.abs(y_true[:, :, :, channel]) + K.abs(y_pred[:, :, :, channel]), axis = -1)
        iou = intersection / (sum_ - intersection)
        return iou

    def channelwise_IoU(self, true, pred):
        b = self.iou_metric(true, pred, 0)
        g = self.iou_metric(true, pred, 1)
        r = self.iou_metric(true, pred, 2)
        return (b + g + r) / 3.0

    def custom_loss(self, true, pred):
        b = self.iou_loss(true, pred, 0)
        g = self.iou_loss(true, pred, 1)
        r = self.iou_loss(true, pred, 2)
        return b + g + r
        # return K.sqrt(K.square(g) + 0.25 * K.square(b) + 0.25 * K.square(r))

    # generator that we will use to read the data from the directory
    def data_gen(self, lists):
        # mean color values and their standard deviation of KITTI data
        while True:
            ix = np.random.choice(np.arange(len(lists)), self.batch_size)
            imgs = []
            labels = []
            for i in ix:
                # images
                image_name = lists[i]  # os.path.join("%06d" % i)
                original_img = cv2.imread(self.base_dir + self.img_dir + image_name)
                # masks
                original_mask = cv2.imread(self.base_dir + self.label_dir + image_name)
                array_img = self.crop_resize_norm_bgr(original_img, self.input_shape)
                array_mask = self.crop_resive_mask(original_mask, self.input_shape)
                imgs.append(array_img)
                labels.append(array_mask)
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels

    def crop_resize_norm_bgr(self, img, dims):
        h, w, c = img.shape
        cropped_img = img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        resized_img = cv2.resize(cropped_img, (dims[1], dims[0]))
        normed = resized_img / 255.0
        mean_free = (normed[:, :] - self.mean) / self.std
        array_img = img_to_array(mean_free)
        return array_img


    def crop(self, img, dims):
        h, w, c = img.shape
        cropped_img = img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        resized_img = cv2.resize(cropped_img, (dims[1], dims[0]))
        return resized_img / 255.0


    def crop_resive_mask(self, mask, dims):
        h, w, c = mask.shape
        cropped_mask = mask[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        resized_mask = cv2.resize(cropped_mask, (dims[1], dims[0]))
        array_mask = img_to_array(resized_mask) / 255
        return array_mask

    def train(self):
        history = self.multi_model.fit_generator(generator = self.data_gen(self.train_list),
                                                 steps_per_epoch = self.epoch_steps,
                                                 epochs = self.n_epochs,
                                                 validation_data = self.data_gen(self.val_list),
                                                 validation_steps = self.val_steps,
                                                 initial_epoch = 0,
                                                 callbacks = [self.cp_cb, self.es_cb, self.tb_cb])
        return history.epoch

    def predict(self):
        path = self.base_dir + self.dag_dir + '%02d/' % self.dag_it + self.inf_dir
        print(path)
        for name in self.inf_list:
            imgs = []
            img = cv2.imread(self.base_dir + self.img_dir + name)

            imgs.append(self.crop_resize_norm_bgr(img, self.input_shape))
            imgs = np.array(imgs)

            inference = self.multi_model.predict(imgs)
            out = cv2.resize(inference[0], (1024, 256))

            cv2.imwrite(path + name, out * 255)

    def finish(self):
        K.clear_session()
