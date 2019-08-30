import os
from keras.layers import Conv3D, Convolution3D, MaxPooling3D
from keras.layers import GlobalMaxPooling3D
from keras.layers import Input, Concatenate, Activation, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras.utils.training_utils import multi_gpu_model
from tensorflow.python.client import device_lib

from pe.helpers.shared_helpers import SharedHelpers

sh = SharedHelpers()

l2_lambda = 0.0001


def get_model(input_shape, classes, base_dir="", save_model=True):
    sh.print("SQUEEZENET")
    local_device_protos = device_lib.list_local_devices()
    list_of_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    NUM_GPUS = len(list_of_gpus)

    model = squeezenet(input_shape, len(classes))

    model.summary()

    # Save Model
    if save_model:
        save_template_model(model, base_dir)

    gpu_model = multi_gpu_model(model, gpus=NUM_GPUS)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    gpu_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

    return model, gpu_model


def save_template_model(model, base_dir):
    """
    Save model to the experiment directory
    :param model: Training model
    :return: None
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(base_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
    sh.print("Model Saved")


def squeezenet(input_dim, num_classes):
    img_input = Input(shape=input_dim)
    x = Convolution3D(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same",
                      # kernel_regularizer=l2(l2_lambda),
                      # kernel_initializer='he_uniform',
                      activation="relu",
                      name='sqconv1')(img_input)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool1', padding="valid")(x)

    x = firemodule(x, (16, 64, 64), None)
    x = firemodule(x, (16, 64, 64), None)

    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool3', padding="valid")(x)
    x = firemodule(x, (32, 128, 128), None)
    x = firemodule(x, (32, 128, 128), None)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool5', padding="valid")(x)
    x = firemodule(x, (48, 192, 192), None)
    x = firemodule(x, (48, 192, 192), None)
    x = firemodule(x, (64, 256, 256), None)
    x = firemodule(x, (64, 256, 256), None)
    # Dropout after the last Fire Module
    # x = Dropout(0.2)(x)
    # x = BatchNormalization()(x)

    x = GlobalMaxPooling3D(name="maxpool10")(x)
    x = Dense(num_classes,
              init='normal',
              # kernel_regularizer=l2(l2_lambda),
              # kernel_initializer='he_uniform',
              )(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x, name="squeezenet3d")

    return model


def firemodule(x, filters, name):
    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None

    squeeze_filter, expand_filter1, expand_filter2 = filters
    squeeze = Conv3D(squeeze_filter, (1, 1, 1), activation='relu',
                     # kernel_regularizer=l2(l2_lambda),
                     # kernel_initializer='he_uniform',
                     padding='same',
                     name=conv_name)(x)
    expand1 = Conv3D(expand_filter1, (1, 1, 1), activation='relu',
                     # kernel_regularizer=l2(l2_lambda),
                     # kernel_initializer='he_uniform',
                     padding='same',
                     name=conv_name)(squeeze)
    expand2 = Conv3D(expand_filter2, (3, 3, 3), activation='relu',
                     # kernel_regularizer=l2(l2_lambda),
                     # kernel_initializer='he_uniform',
                     padding='same',
                     name=conv_name)(squeeze)
    x = Concatenate(axis=-1, name=name)([expand1, expand2])
    return x
