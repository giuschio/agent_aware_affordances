import multiprocessing

import h5py
import numpy as np
import os
import sapien.core as sapien
import shutil
import typing

from os.path import join as pj
from pathlib import Path
from tqdm import tqdm

from tools.data_tools.data_loader import DataLoader
from tools.sim_tools.sim_data import SimParams, TrajectoryData, get_cost, TransformData, VisualData
from tools.sim_tools.camera import Camera
from tools.utils.configuration import Configuration
from tools.utils.data import get_job_folders
from tools.utils.misc import dir_exists, human_sort


def get_point(job_folder):
    """
    Processes the data in the job_folder and return the formatted data for training
    :return: points_downsampled: downsampled pointcloud
    :return: sampled_point_index: index in the pointcloud of the interaction point
    :return: ref2cam_q: ref2cam transform as a quaternion
    :return: cost: normalized cost
    :return: relative_task: target_state - initial_state
    """
    sim = SimParams.load_from_path(job_folder)
    trj = TrajectoryData.load_from_path(job_folder)
    trf = TransformData.load_from_path(job_folder)
    vis = VisualData.load_from_path(job_folder, load_options=['xyza'])

    # cost is scalar, success is a bool
    cost, success = get_cost(sim, trj, negative_unreachable=False)
    # scalar
    relative_task = sim.target_object_pose - sim.initial_object_pose
    ref2world = trf.handle2world @ trf.ref2handle
    world2cam = np.linalg.inv(trf.cam_mount2world)
    ref2cam = world2cam @ ref2world
    pose = sapien.Pose.from_transformation_matrix(ref2cam)
    ref2cam_q = np.concatenate((pose.p, pose.q))

    v, h = vis.sampled_pixel_vertical_coordinate, vis.sampled_pixel_horizontal_coordinate
    cam_point_xyz = vis.xyza[v, h, :3]

    # open3d pointcloud
    pcd = Camera.xyza_to_pointcloud(xyza=vis.xyza)
    # points to array
    points = np.array(pcd.points)
    # get all points that are invalid (i.e. have coordinates [0,0,0])
    invalid_points = np.multiply(np.multiply(points[:, 0] == 0, points[:, 1] == 0), points[:, 2] == 0)

    # get valid points
    points_valid = points[np.invert(invalid_points)]
    size = len(points_valid)

    # downsample the valid pointcloud
    # no distribution is given, numpy uses uniform by default
    samples = np.random.choice(a=size, replace=False, size=Configuration.pointcloud_dimension)
    # shape is (10000,3)
    points_downsampled = points_valid[samples, :]

    # shape is (10000,)
    diff = np.linalg.norm((points_downsampled - cam_point_xyz), axis=1)
    # index of the closes point in the downsampled pointcloud to the interaction point
    sampled_point_index = np.argmin(diff)

    res = dict(folder=job_folder, success=success, points=points_downsampled, point_index=sampled_point_index,
               pose=ref2cam_q, cost=cost, relative_task=relative_task)

    return res


def map_to_points(job_list):
    n_processes = 4
    res_list_success = list()
    res_list_not_success = list()
    with multiprocessing.Pool(processes=n_processes) as pool:
        pbar = tqdm(total=len(job_list), smoothing=0., ncols=100)
        for i, res in enumerate(pool.imap_unordered(get_point, job_list)):
            if res['success']:
                res_list_success.append(res)
            else:
                res_list_not_success.append(res)
            pbar.update()
        pbar.close()

    return res_list_success, res_list_not_success


def pre_process_dataset_util(job_folders: typing.List[str], output_folder='_tmp/datasets'):
    # make sure the output path exists
    dir_exists(output_folder)
    # --- DIVIDE JOBS INTO SUCCESS AND NOT SUCCESS ----
    print("Pre-proccesing data...")
    jobs_success, jobs_not_success = map_to_points(job_folders)

    n_success = len(jobs_success)
    n_not_success = len(jobs_not_success)
    print("Total jobs in dataset: " + str(len(job_folders)))
    print("\t- successful: " + str(n_success))
    print("\t- not successful: " + str(n_not_success))
    # --- CREATE HDF5 FILES ---------------------------

    frame_success = h5py.File(pj(output_folder, "success_pcd.hdf5"), "w")
    frame_not_success = h5py.File(pj(output_folder, "not_success_pcd.hdf5"), "w")

    d_success = frame_success.create_dataset("data", (n_success, Configuration.pointcloud_dimension, 3), dtype='f')
    d_not_success = frame_not_success.create_dataset("data", (n_not_success, Configuration.pointcloud_dimension, 3), dtype='f')

    # --- CREATE NUMPY ARRAYS -------------------------
    costs_success = np.ones(n_success)
    costs_not_success = np.ones(n_not_success)

    rt_success = np.zeros(n_success)
    rt_not_success = np.zeros(n_not_success)

    indexes_success = np.zeros(n_success, dtype='int')
    indexes_not_success = np.zeros(n_not_success, dtype='int')

    poses_success = np.zeros((n_success, 7))
    poses_not_success = np.zeros((n_not_success, 7))
    pbar = tqdm(total=n_success + n_not_success, smoothing=0., ncols=100)
    for idx, job in enumerate(jobs_success):
        d_success[idx, :, :] = job['points']
        costs_success[idx] = job['cost']
        rt_success[idx] = job['relative_task']
        indexes_success[idx] = job['point_index']
        poses_success[idx, :] = job['pose']
        pbar.update()

    for idx, job in enumerate(jobs_not_success):
        d_not_success[idx, :, :] = job['points']
        costs_not_success[idx] = job['cost']
        rt_not_success[idx] = job['relative_task']
        indexes_not_success[idx] = job['point_index']
        poses_not_success[idx, :] = job['pose']
        pbar.update()

    # --- SAVE NUMPY ARRAYS --------------------------
    np.save(pj(output_folder, 'success_cost.npy'), costs_success)
    np.save(pj(output_folder, 'success_rt.npy'), rt_success)
    np.save(pj(output_folder, 'success_indexes.npy'), indexes_success)
    np.save(pj(output_folder, 'success_poses.npy'), poses_success)

    np.save(pj(output_folder, 'not_success_cost.npy'), costs_not_success)
    np.save(pj(output_folder, 'not_success_rt.npy'), rt_not_success)
    np.save(pj(output_folder, 'not_success_indexes.npy'), indexes_not_success)
    np.save(pj(output_folder, 'not_success_poses.npy'), poses_not_success)

    # --- CLOSE HDF5 FILES ----------------------------
    frame_success.close()
    frame_not_success.close()

    return n_success


def preprocess_dataset(outfolder, infolder):
    """
    Turn a dataset from a collection of human-readable folders to a consolidated format
    with faster read-times
    :param outfolder: output folder
    :param infolder: path to folder containing the simulation jobs
    :return: n_success: how many jobs in infolder were marked as successful
    """
    # get points
    infolder = Configuration.get_abs(infolder)
    outfolder = Configuration.get_abs(outfolder)
    points = get_job_folders(infolder)

    n_points = len(points)
    for i in range(n_points):
        points[i] = os.path.join(infolder, points[i])

    n_success = pre_process_dataset_util(points, output_folder=outfolder)
    return n_success


def combine_datasets(path1, path2, output_folder):
    """
    Combines path1 and path2 and saves the result to the output_folder
    :param path1: path to folder containing a pre-processed dataset
    :param path2: path to folder containing a pre-processed dataset
    :param output_folder:
    :return:
    """
    # load the two datasets
    d1 = DataLoader(path1)
    d2 = DataLoader(path2)

    # get their shapes (i.e. number of positive and negative examples)
    s1, ns1 = d1.shape()
    s2, ns2 = d2.shape()
    # shape of the combined dataset
    so, nso = s1 + s2, ns1 + ns2

    frame_success = h5py.File(pj(output_folder, "success_pcd.hdf5"), "w")
    frame_not_success = h5py.File(pj(output_folder, "not_success_pcd.hdf5"), "w")

    d_success = frame_success.create_dataset("data", (so, Configuration.pointcloud_dimension, 3), dtype='f')
    d_not_success = frame_not_success.create_dataset("data", (nso, Configuration.pointcloud_dimension, 3), dtype='f')

    # --- CREATE NUMPY ARRAYS -------------------------
    costs_success = np.concatenate((d1.costs_success, d2.costs_success))
    costs_not_success = np.concatenate((d1.costs_not_success, d2.costs_not_success))

    rt_success = np.concatenate((d1.rt_success, d2.rt_success))
    rt_not_success = np.concatenate((d1.rt_not_success, d2.rt_not_success))

    indexes_success = np.concatenate((d1.indexes_success, d2.indexes_success))
    indexes_not_success = np.concatenate((d1.indexes_not_success, d2.indexes_not_success))

    poses_success = np.concatenate((d1.poses_success, d2.poses_success))
    poses_not_success = np.concatenate((d1.poses_not_success, d2.poses_not_success))

    # fill the combined pointcloud datasets
    i = 0
    for j in range(s1):
        d_success[i, :, :] = d1.d_success[j, :, :]
        i += 1
    for j in range(s2):
        d_success[i, :, :] = d2.d_success[j, :, :]
        i += 1

    i = 0
    for j in range(ns1):
        d_not_success[i, :, :] = d1.d_not_success[j, :, :]
        i += 1
    for j in range(ns2):
        d_not_success[i, :, :] = d2.d_not_success[j, :, :]
        i += 1

    # --- SAVE NUMPY ARRAYS --------------------------
    np.save(pj(output_folder, 'success_cost.npy'), costs_success)
    np.save(pj(output_folder, 'success_rt.npy'), rt_success)
    np.save(pj(output_folder, 'success_indexes.npy'), indexes_success)
    np.save(pj(output_folder, 'success_poses.npy'), poses_success)

    np.save(pj(output_folder, 'not_success_cost.npy'), costs_not_success)
    np.save(pj(output_folder, 'not_success_rt.npy'), rt_not_success)
    np.save(pj(output_folder, 'not_success_indexes.npy'), indexes_not_success)
    np.save(pj(output_folder, 'not_success_poses.npy'), poses_not_success)

    # --- CLOSE HDF5 FILES ----------------------------
    frame_success.close()
    frame_not_success.close()


def append_to_dataset(path_to: str, path_from: str):
    """
    Appends data in path2 to data in path1. path1 is overwritten
    :param path_to: path to folder containing a pre-processed dataset
    :param path_from: path to folder containing a pre-processed dataset
    :return:
    """
    if not os.listdir(path_to):
        # path1 is empty, copy path2 into path1
        files = os.listdir(path_from)
        for f in files:
            fp = pj(path_from, f)
            shutil.copy(fp, path_to)
    else:
        # create temp directory
        p_root = os.path.dirname(path_to)
        tmp_outpath = pj(p_root, "_tmp")
        Path(tmp_outpath).mkdir(parents=True, exist_ok=True)
        # store merged dataset in tmp
        combine_datasets(path_to, path_from, tmp_outpath)
        # delete path 1 dir
        shutil.rmtree(path_to)
        # create empty path 1
        Path(path_to).mkdir(parents=True, exist_ok=True)
        # move all files from the tmp directory to path1
        files = os.listdir(tmp_outpath)
        for f in files:
            fp = pj(tmp_outpath, f)
            shutil.copy(fp, path_to)
        shutil.rmtree(tmp_outpath)


def add_jobs(source_folder, target_folder):
    source_points = get_job_folders(source_folder)
    target_points = get_job_folders(target_folder)

    target_points = human_sort(target_points)
    source_points = human_sort(source_points)

    last_target_job_number = int(target_points[-1].replace('job_', ''))
    print("last target job number is", last_target_job_number)

    for jobfolder in source_points:
        job_n = int(jobfolder.replace('job_', '')) + last_target_job_number + 1
        fname = 'job_' + str(job_n)
        shutil.move(src=pj(source_folder, jobfolder), dst=pj(target_folder, fname))


def divide_jobs_by_object(dirpath):
    print("working on folder", dirpath)
    points = get_job_folders(dirpath=dirpath)
    all_names = ['7290', '12259', '11826', '11700', '12540', '12428', '12592', '101917',
                 '12553', '12583', '12580', '12558', '12614', '12480', '12590', '12484',
                 '12543', '12606', '12596']
    pbar = tqdm(total=len(points))
    for job in points:
        pbar.update()
        fullp = os.path.join(dirpath, job)
        sim = SimParams.load_from_path(fullp)
        name = 'undefined'
        for n in all_names:
            if n in sim.object_name:
                name = n

        dest_path = os.path.join(dirpath, name)
        dir_exists(dest_path)
        shutil.move(src=fullp, dst=dest_path)
    pbar.close()


def get_outliers_no_reach(dirpath):
    points = get_job_folders(dirpath)
    pbar = tqdm(total=len(points))
    pj = os.path.join
    count = 0.
    tot = 0.
    bin_path = pj(dirpath, 'excluded')
    dir_exists(bin_path)
    for point in points:
        pbar.update()
        fullp = pj(dirpath, point)
        sim = SimParams.load_from_path(fullp)
        no_reach = 'could NOT reach reference' in sim.simulation_result
        if no_reach:
            count += 1.
            shutil.move(src=fullp, dst=pj(bin_path, str(sim.job_number)))
        tot += 1.


    print(100.* count/tot)
    print("percentage of outliers marked as successful")
    pbar.close()
