from ast import literal_eval

import SimpleITK as sitk
import os
import pandas
import numpy as np

import pe.config as config
from pe.helpers.patch_processor import PatchProcessor
from pe.helpers.shared_helpers import SharedHelpers

sh = SharedHelpers()


class DataProcessor:
    """
    Candidate Generator Data Pre Processor
    Creates multi-slice dataset
    use_windowing: Yes or No to windowing
    resample: Resample with SITK takes a long time - use wisely
    validation_split: How much should be validation and train
    slab_depth: Slab depth
    window_minimum: Minimum window value
    window_maximum: Maximum window value

    """

    def __init__(self):
        """
        Initialize the Pre Processor class

        """
        self.patch_size = config.CANDIDATE_CLASSIFIER_CONFIG['patch_size']
        self.patch_margin = config.CANDIDATE_CLASSIFIER_CONFIG['patch_margin']
        self.raw_images_path = config.CANDIDATE_CLASSIFIER_CONFIG['raw_images_path']
        self.predicted_candidates_path = config.CANDIDATE_CLASSIFIER_CONFIG['predicted_candidates_path']
        self.training_patches_save_path = config.CANDIDATE_CLASSIFIER_CONFIG['training_patches_save_path']

        self.patch_processor = PatchProcessor()

    def postprocess(self, image, candidate_df):
        """
        Data post-process method
        :param image: source
        :return: batch of slabs
        """
        # # Resample image
        # image = self.resample_to_spacing(image, new_spacing=(
        #     image.GetSpacing()[0], image.GetSpacing()[1], self.slice_spacing))

        # Windowing
        # image = sitk.IntensityWindowing(image, self.window_minimum, self.window_maximum)

        # Convert volume to batches of slab
        batch_of_patches = self.get_batch_of_patches(image, candidate_df)

        return batch_of_patches

    def get_batch_of_patches(self, image, candidate_df):
        """
        Method to convert the entire volume into a batch (i.e. 512,512,9,264)
        :param image: Original SITK Image
        :param candidate_df: Candidates dataframe
        :return: Batches of patches - from Whoville ;)
        """

        data = np.zeros((len(candidate_df), self.patch_size[0], self.patch_size[1], self.patch_size[2], 1),
                        dtype=np.float32)

        # Iterate each row - each candidate patch
        # Loop thru rows fro each df
        for index, row in candidate_df.iterrows():
            # Create the patch image
            patch = self.patch_processor.get_patch_image_exp(image=image, centroid=row['centroid'],
                                                             bounding_box=row['bounding_box'],
                                                             classification_patch_size=self.patch_size,
                                                             patch_margin=self.patch_margin)

            patch_data = sitk.GetArrayFromImage(patch).astype(np.float)
            patch_data = np.transpose(patch_data, (2, 1, 0))
            data[index, ..., 0] = patch_data
        sh.print('Image converted to batch size of', data.shape)
        return data

    def preprocess(self):
        """
        Data pre-process method
        :return:
        """
        # This seems to be an issue with SITK where multithread doesnt work
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

        # Starting index for images
        main_index = 0

        # Get the list of image files
        # *** We assume that each image has paired mask ***
        image_extension = "_raw.nii.gz"
        candidate_extension = ".csv"

        image_files = [val for sublist in
                       [[os.path.join(i[0], j) for j in i[2] if j.endswith(image_extension)] for i in
                        os.walk(self.raw_images_path)] for val in
                       sublist]

        candidate_files = [val for sublist in
                           [[os.path.join(i[0], j) for j in i[2] if j.endswith(candidate_extension)] for i in
                            os.walk(self.predicted_candidates_path)] for val in
                           sublist]
        # File Count
        i = 1

        sh.print('Doing some cleanup...')
        self.remove_files(os.path.join(self.training_patches_save_path, "training/Positive/"))
        self.remove_files(os.path.join(self.training_patches_save_path, "training/Negative/"))

        sh.print('Creating training patches...')

        for image_file_name in image_files:
            sh.print('Loading... {0}/{1} images'.format(i, len(image_files)))
            source_name = os.path.basename(os.path.normpath(image_file_name))[:-len(image_extension)]

            # Assume that the pair is in the candidates directory
            candidate_file_name = os.path.join(self.predicted_candidates_path, source_name + candidate_extension)
            sh.print(candidate_file_name)

            # Read Files with SITK
            image = sitk.ReadImage(image_file_name)

            # Read the csv file
            candidate_df = pandas.read_csv(candidate_file_name,
                                           converters={"centroid": literal_eval, "bounding_box": literal_eval,
                                                       "is_patch_valid": literal_eval})

            # Loop thru rows fro each df
            for index, row in candidate_df.iterrows():

                # Create the patch image
                patch = self.patch_processor.get_patch_image(image=image, centroid=row['centroid'],
                                                             bounding_box=row['bounding_box'],
                                                             classification_patch_size=self.patch_size,
                                                             patch_margin=self.patch_margin)

                patch_class_name = "Positive" if row['is_patch_valid'] else "Negative"
                # Save images to the directory
                patch_path = os.path.join(self.training_patches_save_path, "training", patch_class_name,
                                          format(main_index, '06') + ".nii.gz")

                if "Validation" in image_file_name:
                    patch_path = os.path.join(self.training_patches_save_path, "validation", patch_class_name,
                                              format(main_index, '06') + ".nii.gz")

                # Write patches
                sitk.WriteImage(patch, patch_path)

                # Main Idx
                main_index += 1

        sh.print('Finished saving all Patches.')

    @staticmethod
    def remove_files(path):
        """
        Remove all the file sin the path
        :param path: Path to look into
        :return: None
        """
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                sh.print(e)
