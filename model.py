from keras.models import Model
from keras.layers import Input, MaxPool2D, Dropout
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers.merge import Concatenate
from keras.regularizers import l2



def unet(input_shape,
            n_labels,
            filters = 16,
            kernel = 3,
            pool_size = (2, 2),
            output_mode = "softmax"):

    inputs = Input(shape = input_shape)
    activation = 'relu'
    # kernel_regularizer=l2(0.0001)
    # encoder
    conv_1 = Convolution2D(filters * 4, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(inputs)
    conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Dropout(0.5)(conv_1)
    conv_1 = Activation(activation)(conv_1)
    conv_2 = Convolution2D(filters * 4, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation(activation)(conv_2)
    # conv_2 = Dropout(0.5)(conv_2)

    pool_1 = MaxPooling2D(pool_size = pool_size)(conv_2)
    # pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(filters * 8, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation(activation)(conv_3)
    # conv_3 = Dropout(0.5)(conv_3)
    conv_4 = Convolution2D(filters * 8, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation(activation)(conv_4)
    # conv_4 = Dropout(0.5)(conv_4)

    pool_2 = MaxPooling2D(pool_size = pool_size)(conv_4)
    # pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(filters * 16, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation(activation)(conv_5)
    # conv_5 = Dropout(0.5)(conv_5)
    conv_6 = Convolution2D(filters * 16, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation(activation)(conv_6)
    # conv_6 = Dropout(0.5)(conv_6)
    conv_7 = Convolution2D(filters * 16, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation(activation)(conv_7)
    # conv_7 = Dropout(0.5)(conv_7)

    pool_3 = MaxPooling2D(pool_size = pool_size)(conv_7)
    # pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation(activation)(conv_8)
    # conv_8 = Dropout(0.5)(conv_8)
    conv_9 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation(activation)(conv_9)
    # conv_9 = Dropout(0.5)(conv_9)
    conv_10 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation(activation)(conv_10)
    # conv_10 = Dropout(0.5)(conv_10)

    pool_4 = MaxPooling2D(pool_size = pool_size)(conv_10)
    # pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation(activation)(conv_11)
    # conv_11 = Dropout(0.5)(conv_11)
    conv_12 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation(activation)(conv_12)
    # conv_12 = Dropout(0.5)(conv_12)
    conv_13 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation(activation)(conv_13)
    # conv_13 = Dropout(0.5)(conv_13)


    pool_5 = MaxPooling2D(pool_size = pool_size)(conv_13)
    # pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build encoder done..")

    # between encoder and decoder
    conv_14 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(pool_5)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation(activation)(conv_14)
    # conv_14 = Dropout(0.5)(conv_14)
    conv_15 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation(activation)(conv_15)
    # conv_15 = Dropout(0.5)(conv_15)
    conv_16 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation(activation)(conv_16)
    # conv_16 = Dropout(0.5)(conv_16)

    # decoder
    unpool_1 = UpSampling2D(size = pool_size)(conv_16)
    # unpool_1 = MaxUnpooling2D(pool_size)([conv_16, mask_5])
    concat_1 = Concatenate()([unpool_1, conv_13])

    conv_17 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(concat_1)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation(activation)(conv_17)
    # conv_17 = Dropout(0.5)(conv_17)
    conv_18 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation(activation)(conv_18)
    # conv_18 = Dropout(0.5)(conv_18)
    conv_19 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation(activation)(conv_19)
    # conv_19 = Dropout(0.5)(conv_19)

    unpool_2 = UpSampling2D(size = pool_size)(conv_19)
    # unpool_2 = MaxUnpooling2D(pool_size)([conv_19, mask_4])
    concat_2 = Concatenate()([unpool_2, conv_10])

    conv_20 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(concat_2)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation(activation)(conv_20)
    # conv_20 = Dropout(0.5)(conv_20)
    conv_21 = Convolution2D(filters * 32, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation(activation)(conv_21)
    # conv_21 = Dropout(0.5)(conv_21)
    conv_22 = Convolution2D(filters * 16, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation(activation)(conv_22)
    # conv_22 = Dropout(0.5)(conv_22)

    unpool_3 = UpSampling2D(size = pool_size)(conv_22)
    # unpool_3 = MaxUnpooling2D(pool_size)([conv_22, mask_3])
    concat_3 = Concatenate()([unpool_3, conv_7])

    conv_23 = Convolution2D(filters * 16, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(concat_3)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation(activation)(conv_23)
    # conv_23 = Dropout(0.5)(conv_23)
    conv_24 = Convolution2D(filters * 16, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation(activation)(conv_24)
    # conv_24 = Dropout(0.5)(conv_24)
    conv_25 = Convolution2D(filters * 8, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_24)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation(activation)(conv_25)
    # conv_25 = Dropout(0.5)(conv_25)

    unpool_4 = UpSampling2D(size = pool_size)(conv_25)
    # unpool_4 = MaxUnpooling2D(pool_size)([conv_25, mask_2])
    concat_4 = Concatenate()([unpool_4, conv_4])

    conv_26 = Convolution2D(filters * 8, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(concat_4)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Activation(activation)(conv_26)
    # conv_26 = Dropout(0.5)(conv_26)
    conv_27 = Convolution2D(filters * 4, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(conv_26)
    conv_27 = BatchNormalization()(conv_27)
    conv_27 = Activation(activation)(conv_27)
    # conv_27 = Dropout(0.5)(conv_26)

    unpool_5 = UpSampling2D(size = pool_size)(conv_27)
    # unpool_5 = MaxUnpooling2D(pool_size)([conv_27, mask_1])
    concat_5 = Concatenate()([unpool_5, conv_2])

    conv_28 = Convolution2D(filters * 4, (kernel, kernel), padding = "same", kernel_regularizer=l2(0.0001))(concat_5)
    conv_28 = BatchNormalization()(conv_28)
    conv_28 = Activation(activation)(conv_28)
    # conv_28 = Dropout(0.5)(conv_28)

    conv_29 = Convolution2D(n_labels, (1, 1), padding = "valid")(conv_28)
    conv_29 = BatchNormalization()(conv_29)
    outputs = Activation(output_mode)(conv_29)
    # conv_29 = Reshape((input_shape[0] * input_shape[1], n_labels),
    #                   input_shape=(input_shape[0], input_shape[1], n_labels))(conv_29)

    # outputs = Activation(output_mode)(conv_29)
    print("Build decoder done..")

    segunet = Model(inputs = inputs, outputs = outputs, name = "ContiPathNet")

    return segunet
