import numpy as np
import os

import pe.config as config
from pe.helpers import nifti_generator
from pe.helpers.candidate_class_data_processor import DataProcessor
from pe.helpers.shared_helpers import SharedHelpers

sh = SharedHelpers()


class DataSet:
    """
    Candidate Generator DataSet class
    """

    def __init__(self, is_trainable):
        """
        Initialize the DataSet class

        """
        self.batch_size = config.CANDIDATE_CLASSIFIER_CONFIG['batch_size']
        self.patch_size = config.CANDIDATE_CLASSIFIER_CONFIG['patch_size']
        self.training_patches_path = config.CANDIDATE_CLASSIFIER_CONFIG['training_patches_path']
        self.patch_classes = config.CANDIDATE_CLASSIFIER_CONFIG['patch_classes']

        self.train_steps = 0
        self.validation_steps = 0
        self.label_map = ()

        # Pre-Process the data if needed
        if config.CANDIDATE_CLASSIFIER_CONFIG['preprocessing']:
            preprocessor = DataProcessor()
            preprocessor.preprocess()

        # Get the data
        # Load training data / Generators only if model is trainable
        if is_trainable:
            self.generators = self.get_generators()

    def get_generators(self):
        """
        Get Generators. Returns the image generators.
        """
        # For Multi Slice
        data_gen_training_args = dict(
            rotation_range=20.0,
            shear_range=0.2,
            zoom_range=[0.8, 1.2],
            horizontal_flip=True,
            vertical_flip=True,
            back_forth_flip=True,
            augmentation_probability=0.6,
        )
        # For Multi Slice
        data_gen_valid_args = dict(
            horizontal_flip=False,
            vertical_flip=False,
            back_forth_flip=False,
            augmentation_probability=0,
        )

        seed = 1498
        sh.print('Creating DataGen...')
        train_datagen = nifti_generator.ImageDataGenerator(**data_gen_training_args)
        validation_datagen = nifti_generator.ImageDataGenerator(**data_gen_valid_args)

        train_generator = train_datagen.flow_from_nifti_directory(
            os.path.join(self.training_patches_path, "training/"),
            batch_size=self.batch_size,
            target_size=self.patch_size,
            classes=self.patch_classes,
            class_mode='categorical',
            shuffle=True,
            seed=seed
        )

        validation_generator = validation_datagen.flow_from_nifti_directory(
            os.path.join(self.training_patches_path, "validation/"),
            batch_size=self.batch_size,
            target_size=self.patch_size,
            classes=self.patch_classes,
            class_mode='categorical',
            shuffle=True,
            seed=seed
        )

        sh.print("Train Sample Size: " + str(train_generator.samples) + " - Train Batch Size: " + str(
            train_generator.batch_size))
        sh.print("Validation Sample Size: " + str(validation_generator.samples) + " - Validation Batch Size: " + str(
            validation_generator.batch_size))
        self.label_map = (train_generator.class_indices)
        sh.print("Label Map", self.label_map)

        # Calculate Train and Validation steps
        # print(train_image_generator)
        self.train_steps = (np.ceil(train_generator.samples / train_generator.batch_size)) + 1
        self.validation_steps = (np.ceil(
            validation_generator.samples / validation_generator.batch_size)) + 1

        return train_generator, validation_generator
