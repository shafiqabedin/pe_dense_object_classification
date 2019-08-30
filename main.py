import sys
sys.path.insert(0, "/gpfs/fs0/data/DeepLearning/sabedin/DeployedProjects/incidental-findings/")

import os

import pe.candidate_generator.predictor as candidate_gen_predictor
import pe.candidate_classifier.predictor as candidate_class_predictor
import pe.candidate_generator.trainer as candidate_gen_trainer
import pe.candidate_classifier.trainer as candidate_class_trainer
import pe.config as config
from pe.helpers.shared_helpers import SharedHelpers
from pe.helpers.data_selector import DataSelector

sh = SharedHelpers()


def create_experiemnt_dir(model_name, experiment_id):

    """
    Creates the experiment directory
    :return: Path to the experiment directory
    """
    is_trainable = True
    # Create model dir
    model_dir_path = os.path.join(config.DEFAULT_CONFIG['experiment_base_dir'],
                                  model_name)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    # Create experiment dir
    experiemnt_dir = os.path.join(model_dir_path, experiment_id)
    if not os.path.exists(experiemnt_dir):
        os.makedirs(experiemnt_dir)
    else:
        # Since the directory already exists,
        # lets assume that there is a weights.hdf file thatwe can use and don't need to train again
        is_trainable = False
        sh.print('New experiment directory could not be created!')

    # SharedHelpers._print(os.path.realpath(experiemnt_dir))
    return experiemnt_dir, is_trainable


def dump_config(directory):
    """
    Function to save the config parameters
    :param directory: Save Directory
    :return: None
    """
    if not os.path.exists(os.path.join(directory, 'config.txt')):
        with open(os.path.join(directory, 'config.txt'), 'w') as f:
            # DEFAULT_CONFIG
            f.write('DEFAULT_CONFIG\n\n')
            for item in config.DEFAULT_CONFIG:
                f.write('{k}:{v}\n'.format(k=item, v=config.DEFAULT_CONFIG[item]))
            # CANDIDATE_CLASSIFIER_CONFIG
            f.write('\nCANDIDATE_CLASSIFIER_CONFIG\n\n')
            for item in config.CANDIDATE_CLASSIFIER_CONFIG:
                f.write('{k}:{v}\n'.format(k=item, v=config.CANDIDATE_CLASSIFIER_CONFIG[item]))
            # CANDIDATE_GENERATOR_CONFIG
            f.write('\nCANDIDATE_GENERATOR_CONFIG\n\n')
            for item in config.CANDIDATE_GENERATOR_CONFIG:
                f.write('{k}:{v}\n'.format(k=item, v=config.CANDIDATE_GENERATOR_CONFIG[item]))
            sh.print('Finished writing ', os.path.join(directory, 'config.txt'))


if __name__ == '__main__':
    """
    Main method
    
    """

    "Set environment variable"
    os.environ["PYTHONUNBUFFERED"] = "1"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["DISPLAY"] = "localhost:10.0"

    # Run Data Selector
    if config.DEFAULT_CONFIG['run_data_selector']:
        # Init the data selector class
        data_selector = DataSelector()
        # Select Genertor Data
        data_selector.select_generator_data()


    # Run the Candidate Classifier
    if config.DEFAULT_CONFIG['run_candidate_classifier']:

        # Lets prepare the experiment directory
        experiment_dir, is_trainable = create_experiemnt_dir(
            model_name=config.CANDIDATE_CLASSIFIER_CONFIG['model_name'],
            experiment_id=config.CANDIDATE_CLASSIFIER_CONFIG['experiment_id']
        )
        sh.print('Experiment Dir', experiment_dir)

        if (is_trainable):
            # Dump config in the experiment dir
            dump_config(experiment_dir)
            # Init the Candidate Generator
            classifier_trainer = candidate_class_trainer.Trainer(base_dir=experiment_dir, is_trainable=is_trainable)
            # Start Training
            classifier_trainer.train()
        sh.print("Skipping to Prediction")

        # Init the Candidate Classifier Predictor
        classifier_predictor = candidate_class_predictor.Predictor(base_dir=experiment_dir)
        # Start Prediction
        classifier_predictor.predict()
