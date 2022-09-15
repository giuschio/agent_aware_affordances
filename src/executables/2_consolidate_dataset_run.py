"""
The collect_data_* scripts save the results of each simulation as a collection of files in a folder.
This script takes as input a collection of job folders, extracts data of interest, and
    saves it in a consolidated format (hdf5) for faster load times during training
"""

from tools.data_tools.dataset_preprocessing import preprocess_dataset

if __name__ == '__main__':
    infolder = ''   # input dir (e.g. containing job folders from data collection)
    outfolder = ''  # output dir
    preprocess_dataset(outfolder, infolder)
