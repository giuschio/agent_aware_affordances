"""
This script trains (and saves) a network pipeline, given a dataset path
"""

from executables.core.training_job_core import train_model


if __name__ == '__main__':
    # todo: set the path to your consolidated dataset here (relative to the src folder)
    # the script assumes the following folder structure:
    # +-- dataset_dir
    #     +-- train_p
    #     +-- val_p
    # where train_p and val_p are the directories containing the consolidated training and validation data
    # obtained using the consolidate_dataset_run script
    dataset_dir = ""
    train_model(dataset_path=dataset_dir)
    