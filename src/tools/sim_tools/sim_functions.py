"""
This file contains functions that are called during a simulation_job and that should not require tuning
for each setup (i.e. they will not change much)
"""
import logging
import random

import numpy as np
import os

import open3d as o3d
import sapien.core as sapien
import typing

from pathlib import Path
# docs at https://github.com/clemense/yourdfpy, https://pypi.org/project/yourdfpy/
import torch
from yourdfpy import URDF

from tools.sim_tools.actors import Object, Robot
from tools.sim_tools.sim_data import TransformData, SimParams, VisualData
from tools.urdf_tools.urdf_scaling import scale_urdf_util, move_urdf_util, set_urdf_joint_dynamics
from tools.utils.configuration import Configuration
from tools.utils.misc import sample_from_bounds, sample_object_state, dir_exists
from tools.utils.transforms import get_normal_ref_system, sample_ref_system

from tools_mppi.utils import build_mppi_config


def debug_set_marker(scene: sapien.Scene,
                     pose: sapien.Pose,
                     color=None,
                     name='') -> sapien.Actor:
    """
    Set a visual marker (box with a reference system) in the scene
    :param scene: sapien.Scene to create a box.
    :param pose: 6D pose of the box (in world frame).
    :param color: [3] or [4], rgb or rgba
    :param name: name of the marker.
    :return: box Actor
    """
    half_size = np.array([0.02, 0.02, 0.02])
    builder = scene.create_actor_builder()
    # only visual shape for debugging purposes
    builder.add_box_visual(half_size=half_size, color=color)
    marker = builder.build(name=name)
    marker.set_pose(pose)
    return marker


def build_object_urdf(sim_params: SimParams):
    """
    :param sim_params
    """
    logging.disable()
    urdf_path = os.path.join(Configuration.get_abs(sim_params.object_name), 'mobility_original.urdf')
    model = URDF.load(urdf_path, mesh_dir="")
    if sim_params.dynamically_build_urdf:
        # if dynamically_build = True -> apply modifications. else -> copy the original
        model = scale_urdf_util(model, scale=sim_params.object_scale)
        model = move_urdf_util(model, center_xyz=sim_params.object_center_xyz)
        model = set_urdf_joint_dynamics(model, damping=sim_params.art_damping, friction=sim_params.art_friction)
    model.write_xml_file(os.path.join(Configuration.get_abs(sim_params.object_name), 'mobility.urdf'))

# todo: this should be cleaned up, and combined with the VAT-mart task sampling in some way
def sample_task_by_task(sim_params: SimParams, bounds):
    tasks = list(sim_params.task_weights.keys())
    weights = list(sim_params.task_weights.values())
    if sim_params.task is None:
        sim_params.task = random.choices(population=tasks, weights=weights)[0]
    if sim_params.task in ['open-zero', 'close-ninety']:
        # open from zero or close from ninety
        if sim_params.task == 'open-zero': init = bounds[0]
        else: init = bounds[1]
        target = sample_from_bounds(None, bounds)
        while abs(init-target) < sim_params.min_relative_task:
            target = sample_from_bounds(None, bounds)
    else:
        a, b = 0.0, 0.0
        while abs(a-b) < sim_params.min_relative_task:
            a, b = sample_from_bounds(None, bounds), sample_from_bounds(None, bounds)
        if sim_params.task == 'open':
            # open
            init, target = min(a, b), max(a, b)
        else:
            # close
            init, target = max(a, b), min(a, b)
    sim_params.initial_object_pose = init
    sim_params.target_object_pose = target
    return sim_params


def sample_task_by_state(sim_params, bounds):
    sim_params.initial_object_pose = sample_object_state(var=sim_params.initial_object_pose,
                                                         bounds=bounds,
                                                         pdf=sim_params.object_init_state_pdf)
    sim_params.target_object_pose = sample_object_state(var=sim_params.target_object_pose,
                                                        bounds=bounds,
                                                        pdf=sim_params.object_init_state_pdf)
    while abs(sim_params.initial_object_pose - sim_params.target_object_pose) < sim_params.min_relative_task:
        sim_params.target_object_pose = None
        sim_params.target_object_pose = sample_object_state(var=sim_params.target_object_pose,
                                                            bounds=bounds,
                                                            pdf=sim_params.object_init_state_pdf)
    return sim_params


def set_params(sim_params: SimParams, replay=False) -> SimParams:
    """
    Generate missing simulation parameters (if necessary) and build the MPPI config file
    :return: sim_params: the class is edited in-place, but I return it anyways for readability
    """
    # generate missing simulation data (required before the engine has started)
    sim_params.savepath = os.path.join(sim_params.data_folder, "job_" + str(sim_params.job_number))
    # make sure the save directory exists
    if not replay:
        dir_exists(sim_params.savepath)
    # ---- ROBOT INIT STATE
    if sim_params.initial_robot_base_pose is None:
        sim_params.initial_robot_base_pose = Robot.sample_base_pose(radius_bounds=sim_params.robot_initial_distance_bounds,
                                                                    angle_bounds=sim_params.robot_initial_angle_bounds)
    # ---- OBJECT PROPERTIES
    sim_params.object_scale = sample_from_bounds(var=sim_params.object_scale, bounds=sim_params.object_scale_bounds)
    sim_params.object_center_xyz = sample_from_bounds(var=sim_params.object_center_xyz, bounds=sim_params.object_center_xyz_bounds)
    build_object_urdf(sim_params)
    # ---- OBJECT INIT STATE
    # since each object is different, we need to get the articulated joint limits dynamically from the urdf
    object_model_urdf_path = Configuration.get_abs(os.path.join(sim_params.object_name, "mobility.urdf"))
    object_model_urdf = URDF.load(object_model_urdf_path)
    articulation = object_model_urdf.actuated_joints[0]
    bounds = [articulation.limit.lower, articulation.limit.upper]
    if sim_params.initial_object_pose is None:
        # if the task is not hardcoded, sample it
        sim_params = sample_task_by_task(sim_params=sim_params, bounds=bounds)
    # ---- MPPI PARAMS
    sim_params.controller_params = build_mppi_config(src2obj_path=sim_params.object_name,
                                                     initial_base_pose=sim_params.initial_robot_base_pose,
                                                     initial_object_state=sim_params.initial_object_pose,
                                                     kind="controller")
    sim_params.simulation_params = build_mppi_config(src2obj_path=sim_params.object_name,
                                                     initial_base_pose=sim_params.initial_robot_base_pose,
                                                     initial_object_state=sim_params.initial_object_pose,
                                                     kind="simulation")
    return sim_params


def get_visuals(camera,
                m_object,
                visual_data: VisualData) -> VisualData:
    """
    Get visual data as np arrays
    :param camera: reference to the engine camera instance
    :param m_object: reference to the object actor
    :param visual_data: empty visual data class
    """
    visual_data.rgb = camera.get_rgb()
    visual_data.xyza = camera.get_xyza()
    visual_data.normal_map = camera.get_normal_map()
    movable_links_mask = camera.get_links_mask(m_object.get_movable_links_ids()).astype('float32')
    visual_data.movable_links_mask = movable_links_mask
    # pass a copy to avoid the array being modified in-place suring sampling
    line, col = camera.sample_pixel(np.copy(movable_links_mask))
    # interaction pixel
    visual_data.sampled_pixel_horizontal_coordinate = col
    visual_data.sampled_pixel_vertical_coordinate = line
    return visual_data


def sample_interaction_orientation(camera, sim_params, visual_data):
    """
    Sample an interaction orientation, given an interaction PIXEL
    :param camera:
    :param sim_params:
    :param visual_data:
    :return:
    """
    # get sampled point coordinates and normal vector (in SAPIEN camera frame)
    v, h = visual_data.sampled_pixel_vertical_coordinate, visual_data.sampled_pixel_horizontal_coordinate
    cam_point_xyz = visual_data.xyza[v, h, :3]
    cam_nv = visual_data.normal_map[v, h, :3]
    ground_vector = np.array([0, 0, 1])

    # get transforms
    cam2world = camera.cam2world
    normal2cam = get_normal_ref_system(point_xyz=cam_point_xyz, normal_vector=cam_nv, ground_vector=ground_vector)
    ref2normal = sample_ref_system(direction_distribution=dict(type=sim_params.interaction_orientation_sampling_mode))
    # get derived transforms
    normal2world = cam2world @ normal2cam
    ref2world = normal2world @ ref2normal
    return ref2world


def get_transforms(camera, m_object, sim_params, transform_data, ref2world) -> TransformData:
    handle_link = sim_params.controller_params["cost"]["handle_frame"]
    world2handle = m_object.get_transform(source_link='world', target_link=handle_link).to_transformation_matrix()
    ref2handle = world2handle @ ref2world
    # save transform data
    transform_data.ref2handle = ref2handle
    transform_data.handle2world = np.linalg.inv(world2handle)
    transform_data.cam_mount2world = camera.cam2world
    transform_data.cam2cam_mount = np.identity(4)

    return transform_data


def visualize_actionability(sampled_pcd, colors, sim_params=None, task=None, visualize=True):
    pcd_points_r = np.copy(sampled_pcd)
    # switch to open3d space for correct visualization
    pcd_points_r[:, 2] = -sampled_pcd[:, 0]
    pcd_points_r[:, 0] = -sampled_pcd[:, 1]
    pcd_points_r[:, 1] = sampled_pcd[:, 2]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points_r)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name="Actionability map (warmer is better). CLOSE TO CONTINUE")
    return pcd


def load_checkpoint(checkpoint: typing.Union[str, dict]):
    if type(checkpoint) == str:
        # I passed the path to the checkpoint for some reason
        checkpoint = torch.load(checkpoint)
    elif type(checkpoint) == dict:
        # I passed an already-loaded model
        # (useful since the most expensive part is loading the file from memory)
        pass
    else:
        raise ValueError("input: checkpoint was neither a dict or a path to a dict")
    return checkpoint


def eval_movement(init_pose, final_pose, target_pose):
    initial_delta = abs(init_pose - target_pose)
    final_delta = abs(final_pose - target_pose)
    movement_towards_target = initial_delta - final_delta
    return movement_towards_target