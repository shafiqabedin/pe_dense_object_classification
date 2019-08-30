import sys

import math
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

import pe.config as config
from pe.candidate_classifier.data import DataSet
from pe.helpers.shared_helpers import SharedHelpers
from pe.models import model_densenet3d
from pe.models import model_resnet3d
from pe.models import model_squeezenet

sh = SharedHelpers()


class Trainer():
    """
    Trainer Class
    """

    def __init__(self, base_dir, is_trainable):
        """
        Initialize the Trainer class
        """
        # Set the base experiment dir
        self.base_dir = base_dir
        self.is_trainable = is_trainable
        self.data = None
        self.workers = config.CANDIDATE_CLASSIFIER_CONFIG["workers"]
        self.max_queue_size = config.CANDIDATE_CLASSIFIER_CONFIG["max_queue_size"]
        self.nb_epoch = config.CANDIDATE_CLASSIFIER_CONFIG["nb_epoch"]
        self.patch_size = config.CANDIDATE_CLASSIFIER_CONFIG['patch_size']
        self.patch_classes = config.CANDIDATE_CLASSIFIER_CONFIG['patch_classes']

        # Get Checkpoint
        self.checkpointer = self.get_checkpoint()

        # Get Checkpoint
        self.earlystopper = self.get_earlystopper()

        # Tensorboard
        self.tensorboard = TensorBoard(log_dir=self.base_dir)

        # Custom save Model
        self.best_val_acc = 0
        self.best_val_loss = sys.float_info.max

    def train(self):
        """
        Train function decides which model to fire up given the name of the model.
        """

        # Choose Model

        if config.CANDIDATE_CLASSIFIER_CONFIG["model_name"] == "RESNET3D":
            # Get Data
            self.data = DataSet(self.is_trainable)
            # Model
            self.resnet_3d()
        elif config.CANDIDATE_CLASSIFIER_CONFIG["model_name"] == "DENSENET3D":
            # Get Data
            self.data = DataSet(self.is_trainable)
            # Model
            self.densenet_3d()
        elif config.CANDIDATE_CLASSIFIER_CONFIG["model_name"] == "SQUEEZENET":
            # Get Data
            self.data = DataSet(self.is_trainable)
            # Model
            self.squeezenet()

    def step_decay(epoch, initial_lrate, drop, epochs_drop):
        """
        # learning rate schedule
        :param initial_lrate:
        :param drop:
        :param epochs_drop:
        :return:
        """
        return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))

    def get_checkpoint(self):
        """
        Returns the checkpoint.
        Returns:
            checkpoint: Keras ModelCheckpoint
        """
        return ModelCheckpoint(
            # filepath=os.path.join(self.base_dir, "{epoch:03d}-{val_loss:.2f}.hdf5"),
            filepath=os.path.join(self.base_dir, "weights.hdf5"),
            verbose=1,
            # monitor='val_loss',  # val_acc
            # mode='min',
            save_best_only=True,  # True
            save_weights_only=True
        )

    def save_base_model(self, epoch, logs, base_model=None, gpu_model=None):
        """
        Custom ModelCheckpoint function to save both base and the parallel gpu model
        :param logs: logg source
        :param epoch: epoch
        :return:
        """
        str = ("Firing save_base_model @ " + str(epoch))
        val_acc = logs['val_acc']
        val_loss = logs['val_loss']
        model_filepath = os.path.join(self.base_dir, "base_model_weights.hdf5")
        gpu_model_filepath = os.path.join(self.base_dir, "gpu_model_weights.hdf5")

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            sh.print(str)
            base_model.save(model_filepath)
            gpu_model.save(gpu_model_filepath)

        elif val_acc == self.best_val_acc:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                sh.print(str)
                base_model.save(model_filepath)
                gpu_model.save(gpu_model_filepath)

    def get_earlystopper(self):
        """
        Returns the EarlyStopping.
        Returns:
            earlystopping: Keras EarlyStopping
        """
        return EarlyStopping(patience=20)

    def get_reduce_lr_on_plateau(self):
        """
        Returns the ReduceLROnPlateau.
        Returns:
            reduce_lr_on_plateau: Keras ReduceLROnPlateau
        """
        return ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)

    def train_model(self, model, callbacks=[]):
        """
        Method where the actual training kicks off.
        Args:
            model: The compiled model
            callbacks: List of callbacks
        """

        train_generator, validation_generator = self.data.generators

        # Fit
        model.fit_generator(
            train_generator,
            steps_per_epoch=self.data.train_steps,
            validation_data=validation_generator,
            validation_steps=self.data.validation_steps,
            workers=self.workers,
            use_multiprocessing=False,
            max_queue_size=self.max_queue_size,
            epochs=self.nb_epoch,
            callbacks=callbacks
        )

        return model

    def resnet_3d(self):
        """
        Runs the actual training and prediction finctions
        """
        # Get the model
        patch_size_ch = self.patch_size + (1,)
        sh.print(self.patch_size, patch_size_ch, self.patch_classes)
        model = model_resnet3d.get_model(patch_size_ch, self.patch_classes, base_dir=self.base_dir,
                                         save_model=True)
        # model = model_resnet3d_eh.get_model(patch_size_ch, self.patch_classes, base_dir=self.base_dir,
        #                                     save_model=True)

        # Start training
        if self.is_trainable:
            sh.print("Training... ")
            _ = self.train_model(model, [self.checkpointer, self.tensorboard, self.get_earlystopper()])

    def densenet_3d(self):
        """
        Runs the actual training and prediction finctions
        """
        # Get the model
        patch_size_ch = self.patch_size + (1,)
        sh.print(self.patch_size, patch_size_ch, self.patch_classes)
        model = model_densenet3d.get_model(patch_size_ch, self.patch_classes, base_dir=self.base_dir,
                                           save_model=True)
        # model = model_resnet3d_eh.get_model(patch_size_ch, self.patch_classes, base_dir=self.base_dir,
        #                                     save_model=True)

        # Start training
        if self.is_trainable:
            sh.print("Training... ")
            _ = self.train_model(model, [self.checkpointer, self.tensorboard, self.get_earlystopper()])

    def squeezenet(self):
        """
        Runs the actual training and prediction finctions
        """
        # Get the model
        patch_size_ch = self.patch_size + (1,)
        sh.print(self.patch_size, patch_size_ch, self.patch_classes)
        model, gpu_model = model_squeezenet.get_model(patch_size_ch, self.patch_classes, base_dir=self.base_dir,
                                           save_model=True)

        # Start training
        if self.is_trainable:
            sh.print("Training... ")
            _ = self.train_model(gpu_model, [self.checkpointer, self.tensorboard, self.get_earlystopper()])
