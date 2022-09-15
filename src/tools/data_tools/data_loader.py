import h5py
import numpy as np

from os.path import join as pj

import transforms3d.quaternions


class DataLoader:
    def __init__(self, dataset_folder):
        self.rotation_representation = 'quat'    # alternative is 'r6'
        self.cost_representation = 'success'     # alternative is 'continuous'
        self.success_file = h5py.File(pj(dataset_folder, "success_pcd.hdf5"), 'r')
        self.not_success_file = h5py.File(pj(dataset_folder, "not_success_pcd.hdf5"), 'r')
        self.d_success = self.success_file['data']
        self.d_not_success = self.not_success_file['data']

        self.costs_success = np.load(pj(dataset_folder, 'success_cost.npy')).astype('float32')
        self.rt_success = np.load(pj(dataset_folder, 'success_rt.npy')).astype('float32')
        self.indexes_success = np.load(pj(dataset_folder, 'success_indexes.npy'))
        self.poses_success = np.load(pj(dataset_folder, 'success_poses.npy')).astype('float32')

        self.costs_not_success = np.load(pj(dataset_folder, 'not_success_cost.npy')).astype('float32')
        self.rt_not_success = np.load(pj(dataset_folder, 'not_success_rt.npy')).astype('float32')
        self.indexes_not_success = np.load(pj(dataset_folder, 'not_success_indexes.npy'))
        self.poses_not_success = np.load(pj(dataset_folder, 'not_success_poses.npy')).astype('float32')

        ls, ln = self.shape()

        self.range_success = np.arange(0, ls)
        self.range_not_success = np.arange(0, ln)

        self.range_success_c = np.copy(self.range_success)
        self.range_not_success_c = np.copy(self.range_not_success)
        self.reset_called = False
        self.__epoch_done = False

        # hardcode the seed - ensure that training data is always the same and always shown in the same order
        self.rng = np.random.default_rng(0)

    def shape(self):
        return len(self.costs_success), len(self.costs_not_success)

    def set_range(self, success_range: np.arange, non_success_range: np.arange):
        self.range_success = success_range
        self.range_not_success = non_success_range

    @property
    def epoch_done(self):
        return self.__epoch_done

    def reset(self, shuffle=True):
        self.range_success_c = np.copy(self.range_success)
        self.range_not_success_c = np.copy(self.range_not_success)
        if shuffle:
            # if shuffle is False, then every epoch the data presented is the same, and in the same order
            # this is useful for validation (i.e. make sure the validation set is constant)
            self.rng.shuffle(self.range_success_c)
            self.rng.shuffle(self.range_not_success_c)
        self.reset_called = True
        self.__epoch_done = False

    def get_batch(self, dimension=10):
        # replace = False -> a value cannot be sampled more than once
        dim = int(dimension / 2)
        if self.reset_called is False:
            raise ValueError("You need to call DataLoader.reset at least once")
        indexes_success = self.range_success_c[:dim]
        indexes_not_success = self.range_not_success_c[:dim]

        self.range_success_c = self.range_success_c[dim:]
        self.range_not_success_c = self.range_not_success_c[dim:]
        if min(len(self.range_success_c), len(self.range_not_success_c)) < dim:
            self.__epoch_done = True
            self.reset_called = False

        return self.__get_batch_util(np.sort(indexes_success), np.sort(indexes_not_success))

    def __get_batch_util(self, indexes_success, indexes_not_success):
        points = np.concatenate((self.d_success[indexes_success, :, :],
                                 self.d_not_success[indexes_not_success, :, :]))

        costs = np.concatenate((self.costs_success[indexes_success],
                                self.costs_not_success[indexes_not_success]))

        tasks = np.concatenate((self.rt_success[indexes_success],
                                self.rt_not_success[indexes_not_success]))

        poses = np.concatenate((self.poses_success[indexes_success, 3:],
                                self.poses_not_success[indexes_not_success, 3:]))

        if self.rotation_representation == 'r6':
            poses_n = list()
            for pose in poses:
                mm = transforms3d.quaternions.quat2mat(pose)
                # x and y vectors
                poses_n.append(np.concatenate((mm[:3, 0].flatten(), mm[:3, 1].flatten())))
            poses = np.array(poses_n)

        if self.cost_representation == 'success':
            # success is 0, non-success is 1: this is to keep with the "lower-is-better" convention
            # used for the normalized cost
            costs = np.concatenate((np.zeros_like(indexes_success), np.ones_like(indexes_not_success)))

        point_indexes = np.concatenate((self.indexes_success[indexes_success],
                                        self.indexes_not_success[indexes_not_success]))

        return points, point_indexes, poses, costs, tasks

    def close(self):
        self.success_file.close()
        self.not_success_file.close()


def check_batch_size(loader: DataLoader,
                     default_batch_size: int):
    """
    Deals with edge cases where there is very little training data (e.g. only 2 points)
    and the default batch size is too large
    """
    a, b = loader.shape()
    batch_size = min(default_batch_size, a, b)
    # the batch size should be even (half success, half not success)
    if batch_size % 2 == 1:
        batch_size -= 1
    return batch_size
