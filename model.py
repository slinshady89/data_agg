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
    conv_1 = Layer(activation, 4, filters, inputs, kernel)
    conv_2 = Layer(activation, 4, filters, conv_1, kernel)

    pool_1 = MaxPooling2D(pool_size = pool_size)(conv_2)
    # pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Layer(activation, 8, filters, pool_1, kernel)
    conv_4 = Layer(activation, 8, filters, conv_3, kernel)

    pool_2 = MaxPooling2D(pool_size = pool_size)(conv_4)
    # pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Layer(activation, 16, filters, pool_2, kernel)
    conv_6 = Layer(activation, 16, filters, conv_5, kernel)
    conv_7 = Layer(activation, 16, filters, conv_6, kernel)

    pool_3 = MaxPooling2D(pool_size = pool_size)(conv_7)
    # pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Layer(activation, 32, filters, pool_3, kernel)
    conv_9 = Layer(activation, 32, filters, conv_8, kernel)
    conv_10 = Layer(activation, 32, filters, conv_9, kernel)

    pool_4 = MaxPooling2D(pool_size = pool_size)(conv_10)
    # pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Layer(activation, 32, filters, pool_4, kernel)
    conv_12 = Layer(activation, 32, filters, conv_11, kernel)
    conv_13 = Layer(activation, 32, filters, conv_12, kernel)


    pool_5 = MaxPooling2D(pool_size = pool_size)(conv_13)
    # pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build encoder done..")

    # between encoder and decoder
    conv_14 = Layer(activation, 32, filters, pool_5, kernel)
    conv_15 = Layer(activation, 32, filters, conv_14, kernel)
    conv_16 = Layer(activation, 32, filters, conv_15, kernel)

    # decoder
    unpool_1 = UpSampling2D(size = pool_size)(conv_16)
    # unpool_1 = MaxUnpooling2D(pool_size)([conv_16, mask_5])
    concat_1 = Concatenate()([unpool_1, conv_13])

    conv_17 = Layer(activation, 32, filters, concat_1, kernel)
    conv_18 = Layer(activation, 32, filters, conv_17, kernel)
    conv_19 = Layer(activation, 32, filters, conv_18, kernel)

    unpool_2 = UpSampling2D(size = pool_size)(conv_19)
    # unpool_2 = MaxUnpooling2D(pool_size)([conv_19, mask_4])
    concat_2 = Concatenate()([unpool_2, conv_10])

    conv_20 = Layer(activation, 32, filters, concat_2, kernel)
    conv_21 = Layer(activation, 32, filters, conv_20, kernel)
    conv_22 = Layer(activation, 16, filters, conv_21, kernel)

    unpool_3 = UpSampling2D(size = pool_size)(conv_22)
    # unpool_3 = MaxUnpooling2D(pool_size)([conv_22, mask_3])
    concat_3 = Concatenate()([unpool_3, conv_7])

    conv_23 = Layer(activation, 16, filters, concat_3, kernel)
    conv_24 = Layer(activation, 16, filters, conv_23, kernel)
    conv_25 = Layer(activation, 8, filters, conv_24, kernel)

    unpool_4 = UpSampling2D(size = pool_size)(conv_25)
    # unpool_4 = MaxUnpooling2D(pool_size)([conv_25, mask_2])
    concat_4 = Concatenate()([unpool_4, conv_4])

    conv_26 = Layer(activation, 8, filters, concat_4, kernel)
    conv_27 = Layer(activation, 4, filters, conv_26, kernel)

    unpool_5 = UpSampling2D(size = pool_size)(conv_27)
    # unpool_5 = MaxUnpooling2D(pool_size)([conv_27, mask_1])
    concat_5 = Concatenate()([unpool_5, conv_2])

    conv_28 = Layer(activation, 4, filters, concat_5, kernel)

    conv_29 = Convolution2D(n_labels, (1, 1), padding = "valid")(conv_28)
    conv_29 = BatchNormalization()(conv_29)
    outputs = Activation(output_mode)(conv_29)
    # conv_29 = Reshape((input_shape[0] * input_shape[1], n_labels),
    #                   input_shape=(input_shape[0], input_shape[1], n_labels))(conv_29)

    # outputs = Activation(output_mode)(conv_29)
    print("Build decoder done..")

    segunet = Model(inputs = inputs, outputs = outputs, name = "ContiPathNet")

    return segunet


def Layer(activation, filter_factor, filters, inputs, kernel):
    conv_1 = Convolution2D(filters * filter_factor, (kernel, kernel), padding = "same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation(activation)(conv_1)
    return conv_1
