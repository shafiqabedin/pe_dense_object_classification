import csv
from ast import literal_eval

import SimpleITK as sitk
import math
import numpy as np
import os
import pandas as pd
from keras.models import model_from_json
from keras.utils.training_utils import multi_gpu_model
from tensorflow.python.client import device_lib

import pe.config as config
import pe.lung_segmentation.inference as lung_inference
from pe.helpers.candidate_class_data_processor import DataProcessor
from pe.helpers.patch_processor import PatchProcessor
from pe.helpers.shared_helpers import SharedHelpers
from pe.models import model_squeezenet

sh = SharedHelpers()


class Predictor:
    """
    Initializes the Candidate Generator predictor class
    """

    def __init__(self, base_dir):
        """
        Initialize the Predictor class
        Args:
            base_dir: Experiment save path

        """
        # Other paths
        self.prediction_images_path = config.CANDIDATE_CLASSIFIER_CONFIG["prediction_images_path"]
        self.prediction_candidates_path = config.CANDIDATE_CLASSIFIER_CONFIG["prediction_candidates_path"]
        self.prediction_candidates_csv_path = config.CANDIDATE_CLASSIFIER_CONFIG["prediction_candidates_csv_path"]
        self.classification_patch_size = config.CANDIDATE_CLASSIFIER_CONFIG["patch_size"]

        self.patch_size = config.CANDIDATE_CLASSIFIER_CONFIG['patch_size']
        self.patch_margin = config.CANDIDATE_CLASSIFIER_CONFIG['patch_margin']
        self.patch_classes = config.CANDIDATE_CLASSIFIER_CONFIG['patch_classes']

        # Weight path
        self.base_dir = base_dir

        self.candidate_class_model = self.load_model(os.path.join(base_dir, 'model.json'),
                                                     os.path.join(base_dir, 'weights.hdf5'))
        # Data Processor
        self.postprocessor = DataProcessor()
        # Patch Processor
        self.patch_processor = PatchProcessor()
        # init the lung segmentation inference
        lung_inference.lung_segmentation_init('/home/sabedin/Data/cad_pe_segmentation/lung_segmentation_resources')

    def load_model(self, model_arch, model_weight):
        """
        Load the model and the weights
        :param model_arch:
        :param model_weight:
        :return:
        """
        local_device_protos = device_lib.list_local_devices()
        list_of_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        num_gpus = len(list_of_gpus)

        json_file = open(model_arch, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        model = multi_gpu_model(model, gpus=num_gpus)
        model.load_weights(model_weight)

        return model

    def load_model_and_save_single_gpu_model(self, model_arch, model_weight):
        """
        Load the model and the weights
        :param model_arch:
        :param model_weight:
        :return:
        """
        local_device_protos = device_lib.list_local_devices()
        list_of_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        num_gpus = len(list_of_gpus)

        # Get the model
        # Get the model
        patch_size_ch = self.patch_size + (1,)
        model, gpu_model = model_squeezenet.get_model(patch_size_ch, self.patch_classes, base_dir=self.base_dir,
                                                      save_model=False)

        model_weights_before = model.get_weights()

        gpu_model.load_weights(model_weight)

        model_weights_after = model.get_weights()

        if not np.array_equal(model_weights_before, model_weights_after):
            print("Weights are NOT similar")
        else:
            print("Weights ARE similar")

        # serialize weights to HDF5
        model.save_weights(os.path.join(self.base_dir, "single_gpu_weights.hdf5"))
        print("Saved model to disk")

        return gpu_model

    def predict_old(self):
        """
        Method that actually runs the prediction
        :return: None
        """
        # Get the list of image files
        # *** We assume that each image has paired mask ***
        image_extension = "_raw.nii.gz"
        candidate_extension = ".csv"

        if os.path.isfile(self.prediction_candidates_csv_path):

            try:
                f = open(self.prediction_candidates_csv_path, 'r')
            except IOError:
                print("Could not read file:", self.prediction_candidates_csv_path)
                exit(0)
            with f:
                reader = csv.reader(f)
                image_files = [row[1] for row in reader]
                sh.print("Processing CSV file of size ", len(image_files))
        else:
            image_files = [val for sublist in
                           [[os.path.join(i[0], j) for j in i[2] if j.endswith(image_extension)] for i in
                            os.walk(self.prediction_images_path)] for val in
                           sublist]
            sh.print("Processing local file of size ", len(image_files))

        # File Count
        i = 1

        for image_file_name in image_files:
            sh.print('Loading... {0}/{1} images'.format(i, len(image_files)))
            source_name = os.path.basename(os.path.normpath(image_file_name))[:-len(image_extension)]

            # Assume that the pair is in the candidates directory
            candidate_file_name = os.path.join(self.prediction_candidates_path, source_name + candidate_extension)
            sh.print(candidate_file_name)

            # Read Files with SITK
            image = sitk.f(image_file_name, sitk.sitkFloat32)

            # Read the csv file
            candidate_df = pd.read_csv(candidate_file_name,
                                       converters={"centroid": literal_eval, "bounding_box": literal_eval,
                                                   "is_patch_valid": literal_eval})
            sh.print("Total Number of Candidates", str(len(candidate_df)))

            # Get batch of patches
            batch_of_patches = self.postprocessor.get_batch_of_patches(image, candidate_df)
            # Steps
            steps = int(np.ceil(len(candidate_df) / 16))
            # Predict
            prediction = self.candidate_class_model.predict(batch_of_patches, steps=None, batch_size=16)
            sh.print(prediction)
            # Get scores -  Label Map {'Negative': 0, 'Positive': 1}
            scores = prediction[:, 1]
            scores = np.around(scores, decimals=10)
            # Update the prediction column
            candidate_df = candidate_df.assign(probability=pd.Series(scores))

            # Write csv file
            candidate_df.to_csv(candidate_file_name, encoding='utf-8', index=False)

            i += 1

    def predict(self):
        """
        Method that actually runs the prediction
        :return: None
        """
        # Get the list of image files
        # *** We assume that each image has paired mask ***
        image_extension = "_raw.nii.gz"
        mask_extension = "_mask.nii.gz"

        if os.path.isfile(self.prediction_candidates_csv_path):

            try:
                f = open(self.prediction_candidates_csv_path, 'r')
            except IOError:
                print("Could not read file:", self.prediction_candidates_csv_path)
                exit(0)
            with f:
                reader = csv.reader(f)
                image_files = [row[1] for row in reader]
                sh.print("Processing CSV file of size ", len(image_files))
        else:
            image_files = [val for sublist in
                           [[os.path.join(i[0], j) for j in i[2] if j.endswith(image_extension)] for i in
                            os.walk(self.prediction_images_path)] for val in
                           sublist]
            sh.print("Processing local file of size ", len(image_files))

        # File Count
        file_counter = 1

        for image_file_name in image_files:
            sh.print('Loading... {0}/{1} images'.format(file_counter, len(image_files)))
            source_name = os.path.basename(os.path.normpath(image_file_name))[:-len(image_extension)]

            # Assume that the pair is in the candidates directory

            mask_file_name = os.path.join(self.prediction_candidates_path, source_name + mask_extension)
            sh.print(mask_file_name)

            # Read Files with SITK
            image = sitk.ReadImage(image_file_name, sitk.sitkFloat32)
            mask_file = sitk.ReadImage(mask_file_name, sitk.sitkUInt8)

            # Read the csv file
            # candidate_df = pd.read_csv(candidate_file_name,
            #                            converters={"centroid": literal_eval, "bounding_box": literal_eval,
            #                                        "is_patch_valid": literal_eval})

            lung_mask, lung_mask_bbox = lung_inference.lung_segmentation_inference_from_image(image)
            sh.print('Lung Mask Bbox', lung_mask_bbox)

            # # Get Candidates - Saves the candidates in the generate_candidates method
            candidate_df = self.generate_candidates(mask_file, lung_mask_bbox)

            sh.print("Total Number of Candidates: ", str(len(candidate_df)))

            batch_size = 16

            batch = np.zeros((batch_size, 64, 64, 64, 1), dtype=np.float)
            counter = 0
            index_list = []
            scores = []
            for index, row in candidate_df.iterrows():
                patch = self.patch_processor.get_patch_image_with_depth(image=image, centroid=row['centroid'],
                                                                        bounding_box=row['bounding_box'],
                                                                        classification_patch_size=self.patch_size,
                                                                        patch_min_depth=9)
                patch_data = sitk.GetArrayFromImage(patch).astype(np.float)
                patch_data = np.transpose(patch_data, (2, 1, 0))
                batch[counter, ..., 0] = patch_data

                counter += 1
                index_list.append(index)

                if counter == batch_size or index == len(candidate_df) - 1:

                    if index == len(candidate_df) - 1:
                        batch = batch[0:counter, ...]

                    if batch.shape[0] == 1:
                        batch_of_two = np.zeros((2, 64, 64, 64, 1), dtype=np.float)
                        batch_of_two[0, ...] = batch[0, ...]
                        batch = batch_of_two

                        outputs = self.candidate_class_model.predict(batch, batch_size=batch.shape[0])
                        outputs = outputs[:-1]
                    else:
                        outputs = self.candidate_class_model.predict(batch, batch_size=batch.shape[0])

                    for idx in range(0, outputs.shape[0]):
                        scores.append(outputs[idx, 1])

                    counter = 0
                    index_list = []

            # image_prob = df['Probability'].max()
            # Update the prediction column
            candidate_df = candidate_df.assign(probability_sq=pd.Series(scores))

            # Write csv file
            candidate_file_name = os.path.join(self.prediction_candidates_path, source_name + '.csv')
            candidate_df.to_csv(candidate_file_name, encoding='utf-8', index=False)

            file_counter += 1

    def generate_candidates(self, candidate_mask, lung_mask_bbox, minimum_label_coverage=40.0):
        """
        This method generates all the candidates
        AKA The segmentation part of the algorithm

        :param image: Takes the original SITK volume
        :return: dataframe of candidates with location - see the candidate_df def below
        """
        # Init Local vars
        patch_count = 0
        candidate_df = pd.DataFrame([], columns=["patch_id", "label_no", "bounding_box", "label_area", "centroid",
                                                 "is_patch_valid", 'probability'])

        # Connected Component
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.FullyConnectedOn()
        candidate_mask = cc_filter.Execute(candidate_mask)

        # Get the mask label stat
        label_stat = sitk.LabelShapeStatisticsImageFilter()
        label_stat.Execute(candidate_mask)

        # Label
        for label_number in label_stat.GetLabels():
            area = label_stat.GetPhysicalSize(label_number)
            centroid = label_stat.GetCentroid(label_number)
            centroid_transformed = candidate_mask.TransformPhysicalPointToIndex(centroid)
            bounding_box = label_stat.GetBoundingBox(label_number)
            width = bounding_box[3]
            height = bounding_box[4]
            depth = bounding_box[5]

            # Check if the label starting pixel in inside the lung bBox
            if self.rect_contains((bounding_box[0], bounding_box[1], bounding_box[2]), lung_mask_bbox):

                # Fork here if emboli is smaller than 64^3
                if (width < self.classification_patch_size[0]) and (height < self.classification_patch_size[1]) and (
                        depth < self.classification_patch_size[2]):
                    # Save the image information in df
                    candidate_df = candidate_df.append({'patch_id': patch_count, 'label_no': label_number,
                                                        "bounding_box": bounding_box,
                                                        "label_area": area,
                                                        "centroid": centroid, "is_patch_valid": False,
                                                        "probability": -1},
                                                       ignore_index=True)
                    patch_count += 1
                else:
                    # print('generate_candidates', label_number, bounding_box)
                    # Loop thru the extent of the emboli
                    for x_loop in range(int(math.ceil(width / 64)) + 1):
                        for y_loop in range(int(math.ceil(height / 64)) + 1):
                            for z_loop in range(int(math.ceil(depth / 64)) + 1):
                                x1 = int(bounding_box[0] + (x_loop * 64))
                                y1 = int(bounding_box[1] + (y_loop * 64))
                                z1 = int(bounding_box[2] + (z_loop * 64))
                                x2 = min((x1 + 64), bounding_box[0] + width)
                                y2 = min((y1 + 64), bounding_box[1] + height)
                                z2 = min((z1 + 64), bounding_box[2] + depth)

                                cuboid_image = candidate_mask[x1:x2, y1:y2, z1:z2]

                                # Get the mask label stat
                                cuboid_label_stat = sitk.LabelShapeStatisticsImageFilter()
                                cuboid_label_stat.Execute(cuboid_image)

                                if label_number in cuboid_label_stat.GetLabels():
                                    cuboid_label_area = cuboid_label_stat.GetPhysicalSize(label_number)
                                    coverage = ((cuboid_label_area / area) * 100)

                                    if coverage > minimum_label_coverage:
                                        cuboid_centroid = cuboid_label_stat.GetCentroid(label_number)
                                        cuboid_bounding_box = (x1, y1, z1, 64, 64, 64)

                                        candidate_df = candidate_df.append(
                                            {'patch_id': patch_count, 'label_no': label_number,
                                             "bounding_box": cuboid_bounding_box,
                                             "label_area": area, "centroid": cuboid_centroid,
                                             "is_patch_valid": False,
                                             "probability": -1}, ignore_index=True)
                                        patch_count += 1
            else:

                print("Label Outside Lung BBox", label_number, (bounding_box[0], bounding_box[1], bounding_box[2]),
                      lung_mask_bbox)

        return candidate_d
