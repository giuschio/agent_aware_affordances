"""
       _
    ___|
   [n n]
 o=     =o
   |GS_|
    d b
"""

import numpy as np
import os
import sapien.core as sapien
import torch
import typing

from random import random
from time import time as get_time

import transforms3d.quaternions

from tools_pytorch.network_full import Pipeline

from tools.probability import Laplace, sample_interaction_index
from tools.sim_tools.sim_data import TrajectoryData, TransformData, SimParams, VisualData
from tools.sim_tools.sim_engine import SimulationEngine
from tools.sim_tools.sim_functions import set_params, get_visuals, sample_interaction_orientation, get_transforms, \
    visualize_actionability, load_checkpoint, eval_movement
from tools.utils.configuration import Configuration
from tools.utils.misc import Logger2 as Logger, dir_exists, heatmap

from tools_mppi.mppi import Reference, Observation


class FeasibilityChecker:
    def __init__(self, sim_engine, lock, ik_check, collision_check):
        self.ik_check = ik_check
        self.collision_check = collision_check
        self.sim_engine = sim_engine
        self.lock = lock

    def __call__(self, ref2world, initial_base_position):
        sim_engine = self.sim_engine
        sim_engine.object.set_joint_position(Observation.object_qpos(sim_engine.current_observation))
        sim_engine.robot.set_qpos(Observation.robot_qpos(sim_engine.current_observation))
        sim_engine.scene.update_render()
        sim_engine.viewer.render()
        # return True if no check is performed
        feas = True
        if self.ik_check:
            with self.lock:
                # pinocchio needs to load the robot urdf file -> lock
                feas, q = sim_engine.robot.inverse_kinematics(ref2world=ref2world,
                                                              base_position=initial_base_position,
                                                              target_frame_name=sim_engine.params.controller_params["cost"][
                                                                  "tracked_frame"])
            if feas and self.collision_check:
                self.sim_engine.robot.set_qpos(q)
                self.sim_engine.robot.set_qvel(np.zeros_like(q))
                self.sim_engine.scene.step()
                # if there is any contact (robot-robot or robot-object) we lose
                feas = len(self.sim_engine.scene.get_contacts()) == 0
        return feas


def full_network_forward_pass(camera,
                              visual_data: VisualData,
                              sim_params: SimParams,
                              checkpoint: typing.Union[str, dict],
                              feasibility_check: FeasibilityChecker) -> typing.Optional[np.ndarray]:
    """
    Given the downsampled pointcloud, the task, and the network, perform the network forward pass
    :param checkpoint: network state_dict (passed either as a path to the .pt file, or as a dict if already loaded)
    """
    sampled_pcd = camera.xyza_to_downsampled_pointcloud(visual_data.xyza, n_points=Configuration.pointcloud_dimension)
    if sampled_pcd is None:
        return None
    pixels = camera.pointcloud_to_pixels(sampled_pcd)
    task = sim_params.target_object_pose - sim_params.initial_object_pose
    checkpoint = load_checkpoint(checkpoint)
    network = Pipeline(orientation_encoding=sim_params.network_orientation_encoding,
                       cost_representation=sim_params.network_cost_representation)
    m_key = 'network_state' if 'network_state' in checkpoint else 'full_state_dict'
    network.load_state_dict(checkpoint[m_key])
    network.eval()

    with torch.no_grad():
        # note: interaction pose is in SAPIEN camera frame
        cost_pred_np, cost_unc_np, point_feats_np = network.get_actionability_whole_pcd(points_np=sampled_pcd, tasks_np=np.array([task]))
        robust_cost = Laplace.get_percentiles(cost_pred_np, cost_unc_np, percentile=sim_params.robust_cost_percentile).flatten()
        # turn cost into point_utility
        point_utility = np.exp(-sim_params.actionability_utility_k * robust_cost).flatten()
        mask = visual_data.movable_links_mask
        filtered_utility = np.array([point_utility[idx] * mask[pixels[idx, 1], pixels[idx, 0]] for idx in range(len(pixels))])
        pose_feasible = False
        coll_checks = 0
        while not pose_feasible and coll_checks < sim_params.testing_max_pose_proposals:
            coll_checks += 1
            # sample interaction point using thompson sampling
            ip_idx = sample_interaction_index(utility=np.copy(filtered_utility), sampling_strategy=sim_params.sampling_strategy)
            # interaction point xyz
            interaction_xyz = np.copy(sampled_pcd[ip_idx, :])

            # assign to visual data
            y_row, x_col = np.copy(pixels[ip_idx, 1]), np.copy(pixels[ip_idx, 0])
            # assign pixel coords
            visual_data.sampled_pixel_horizontal_coordinate = int(x_col)
            visual_data.sampled_pixel_vertical_coordinate = int(y_row)
            # for debugging purposes
            visual_data.rgb[y_row - 2:y_row + 2, x_col - 2:x_col + 2, :] = np.array([1, 0, 0])

            # sample the interaction orientation
            point_feat_np = np.copy(point_feats_np[ip_idx, :])
            sorted_costs, sorted_orientations = network.get_interaction_orientation(point_feat_np=point_feat_np,
                                                                                    xyz_np=interaction_xyz,
                                                                                    task_np=np.array([task]))
            orientation_utility = np.exp(-sim_params.orientation_utility_k * np.copy(sorted_costs).flatten())
            # sample orientation using thompson sampling
            io_idx = sample_interaction_index(utility=orientation_utility, sampling_strategy=sim_params.sampling_strategy)
            # interaction orientation quat
            interaction_quat = np.copy(sorted_orientations[0, io_idx, :])

            if sim_params.network_orientation_encoding == 'r6':
                # interaction_quat is really an r6 vector encoding a rotation matrix
                vv = interaction_quat.flatten()
                _x, _y = vv[:3], vv[3:]
                _z = np.cross(_x, _y).flatten()
                m = np.concatenate((_x.reshape((3, 1)), _y.reshape((3, 1)), _z.reshape((3, 1))), axis=1)
                interaction_quat = transforms3d.quaternions.mat2quat(m)
            ref2cam = sapien.Pose(p=interaction_xyz, q=interaction_quat).to_transformation_matrix()

            # ------ FEASIBILITY CHECK
            _tmp_ref2world = np.copy(camera.cam2world @ ref2cam)
            if feasibility_check.ik_check and sim_params.interaction_orientation_sampling_mode == 'eps-greedy':
                raise ValueError("eps-greedy sampling and feas checking are not compatible (see SimParams class)")
            pose_feasible = feasibility_check(_tmp_ref2world, initial_base_position=sim_params.initial_robot_base_pose)
        sim_params.number_feas_checks += coll_checks

    if sim_params.show_gui:
        mask = np.where(filtered_utility > 0)
        colors = heatmap(array=robust_cost, mask=mask, equalize=1)
        # uncomment to print blue dot on chosen interaction point
        # m_indexes = np.linalg.norm(sampled_pcd - interaction_xyz, axis=1) < 0.03
        # colors[m_indexes] = np.array([137., 209., 254.])/255.
        visualize_actionability(sampled_pcd, colors, sim_params, task)

    del network
    return ref2cam


def simulation_job(sim_params: SimParams,
                   lock=None,
                   network_checkpoint=None,
                   log_to_screen=False):
    dir_exists(sim_params.data_folder)
    if type(network_checkpoint) == str:
        network_checkpoint = Configuration.get_abs(network_checkpoint)
        sim_params.network_checkpoint = network_checkpoint
    log = Logger(log_title="job n " + str(sim_params.job_number),
                 fname=os.path.join(sim_params.data_folder, "process_log_" + str(sim_params.process_number) + ".txt"),
                 log_to_screen=log_to_screen)
    log.append("job started on process number " + str(sim_params.process_number))
    start_time = get_time()
    with lock:
        trajectory_data = TrajectoryData()
        visual_data = VisualData()
        transform_data = TransformData()
        # dynamically create urdf model
        sim_params = set_params(sim_params)
        log.append("sampled sim_params")

        # load simulation Engine (load object and robot to sapien)
        engine = SimulationEngine(params=sim_params, data=trajectory_data)
        log.append("initialized sim engine")
        # load object and robot to MPPI
        engine.load_controller()
        log.append("initialized mppi controller")

    # for convenience
    camera = engine.camera
    m_object = engine.object
    robot = engine.robot
    controller = engine.controller
    clock = engine.clock
    # initialize feasibility checkers (pre and post-sampling)
    fc_pre = FeasibilityChecker(engine, lock, ik_check=sim_params.filter_ik_check, collision_check=sim_params.filter_collision_check)
    fc_post = FeasibilityChecker(engine, lock, ik_check=sim_params.perform_ik_check, collision_check=sim_params.perform_collision_check)
    assert sim_params.closed_loop > 0, "sim_params.closed_loop should be more than zero"
    # uncomment for visualization purposed
    # engine.run_settle()
    for iteration in range(1, sim_params.closed_loop + 1):
        log.append("starting interaction iteration number " + str(iteration))
        sim_params.total_number_of_interactions = iteration
        closed_loop_flag = bool(iteration < sim_params.closed_loop)

        # render scene and call camera.take_picture so that I can get visual data
        engine.render_scene()
        camera.take_picture()
        # get visual data and sample interaction pixel uniformly at random
        visual_data = get_visuals(camera=camera, m_object=m_object, visual_data=visual_data)
        # --- FOR DEBUGGING (with demo object at 2.0 meters, at height of 0.75m)
        # visual_data.sampled_pixel_horizontal_coordinate = 360
        # visual_data.sampled_pixel_vertical_coordinate = 470
        # sim_params.interaction_orientation_sampling_mode = 'fixed'
        log.append("collected visual data")

        if random() > sim_params.eps_greedy and network_checkpoint is not None:
            if type(network_checkpoint) == str:
                log.append("using network checkpoint at " + network_checkpoint)
            # interaction pose in camera frame, need to get them
            ref2cam = full_network_forward_pass(camera=camera,
                                                visual_data=visual_data,
                                                sim_params=sim_params,
                                                checkpoint=network_checkpoint,
                                                feasibility_check=fc_pre)
            ref2world = camera.cam2world @ ref2cam if ref2cam is not None else None
        else:
            ref2world = sample_interaction_orientation(camera, sim_params, visual_data)
        torch.cuda.empty_cache()
        log.append("interaction point sampled successfully")

        if ref2world is not None:
            transform_data = get_transforms(camera=camera,
                                            m_object=m_object,
                                            sim_params=sim_params,
                                            transform_data=transform_data,
                                            ref2world=ref2world)
            log.append("generated kinematic transforms")
            pose_is_reachable = fc_post(ref2world, initial_base_position=sim_params.initial_robot_base_pose)
            # ref_box = debug_set_marker(engine.scene,
            #                            pose=sapien.Pose.from_transformation_matrix(ref2world),
            #                            color=[0., 0., 1.], name='ee_ref')
        else:
            log.append("pose was not sampled successfully due to an invalid pointcloud")
            pose_is_reachable = False
        msg = "pinocchio thinks the reference is " + ("reachable" if pose_is_reachable else "not reachable")
        log.append(msg)

        if pose_is_reachable:
            sim_params.simulation_result.append('pinocchio reachable')
            ref1 = Reference(pose=sapien.Pose.from_transformation_matrix(transform_data.ref2handle), mode=1)
            # run the simulation
            log.append("moving...")
            controller.set_reference([ref1.serialize()], [clock.time()])
            # move to reference
            obj_state_bm = Observation.object_qpos(engine.current_observation)
            pose_reference_reached = engine.run_movement(log=log)
            obj_state = Observation.object_qpos(engine.current_observation)
            if abs(obj_state - obj_state_bm) > sim_params.interaction_successful_tolerance / 2.0:
                # the robot collided with the object during setup motion and moved it by quite a lot
                # mark simulation as non-successful
                sim_params.simulation_result.append('invalid collision during run_movement')
                pose_reference_reached = False
        else:
            sim_params.simulation_result.append('pinocchio not reachable')
            pose_reference_reached = False

        if pose_reference_reached:
            log.append("target reached")
            sim_params.simulation_result.append('reached reference')
            log.append("interacting...")
            ref2 = Reference(pose=sapien.Pose.from_transformation_matrix(transform_data.ref2handle),
                             mode=2, target_pose=sim_params.target_object_pose)
            controller.set_reference([ref2.serialize()], [clock.time()])
            obj_state_beginning_interaction = Observation.object_qpos(engine.current_observation)
            sim_params.final_object_pose = float(engine.run_interaction(log=log, closed_loop=closed_loop_flag))
            movement_tw = eval_movement(init_pose=obj_state_beginning_interaction,
                                        final_pose=sim_params.final_object_pose,
                                        target_pose=sim_params.target_object_pose)
            sim_params.number_successful_interactions += int(movement_tw > sim_params.interaction_successful_tolerance)
        else:
            log.append("target NOT reached")
            sim_params.simulation_result.append('could NOT reach reference')
            sim_params.final_object_pose = float(Observation.object_qpos(engine.current_observation))

        # walk away
        ref3 = Reference(pose=sapien.Pose(), mode=3)
        ref3.arm_j = np.array([0.0, -0.52, 0.0, -1.785, 0.0, 1.10, 0.69])
        bx, by, bt = Observation.robot_qpos(engine.current_observation)[:3].tolist()
        bx -= 0.5
        bt = float(np.arctan2(-by, -bx))
        ref3.base_j = np.array([bx, by, bt])
        controller.set_reference([ref3.serialize()], [clock.time()])
        engine.run_settle()

        if abs(sim_params.final_object_pose - sim_params.target_object_pose) < sim_params.task_completion_tolerance:
            # break out of the loop if the task was completed
            break

    initial_delta = abs(sim_params.initial_object_pose - sim_params.target_object_pose)
    final_delta = abs(sim_params.final_object_pose - sim_params.target_object_pose)
    sim_params.movement_towards_target = initial_delta - final_delta

    log.append("initial object pose: {:0.2f} degs".format(float(np.rad2deg(sim_params.initial_object_pose))))
    log.append("target object pose: {:0.2f} degs".format(float(np.rad2deg(sim_params.target_object_pose))))
    log.append("final object pose: {:0.2f} degs".format(float(np.rad2deg(sim_params.final_object_pose))))
    log.append("movement towards target: {:0.2f} degs".format(float(np.rad2deg(sim_params.movement_towards_target))))
    log.append("number of collision checks: {:2d}".format(sim_params.number_feas_checks))

    final_success = pose_reference_reached and sim_params.movement_towards_target >= sim_params.interaction_successful_tolerance
    final_msg = 'interaction successful' if final_success else 'interaction NOT successful'
    sim_params.simulation_result.append(final_msg)
    sim_params.total_time = clock.time()

    log.append("simulation end")
    if sim_params.show_gui:
        # if we are showing the simulation rendering, do not close the rendering window at
        # the end of the simulation
        while not engine.viewer.closed:
            engine.scene.update_render()
            engine.viewer.render()
    else:
        engine.viewer.close()

    # save data
    visual_data.save(sim_params.savepath)
    transform_data.save(sim_params.savepath)
    sim_params.save(sim_params.savepath)
    trajectory_data.save(sim_params.savepath)
    log.append("saved simulation data")

    log.append("job n " + str(sim_params.job_number) + " completed in {:0.2f} seconds".format(get_time() - start_time))


def build_params(object_name) -> SimParams:
    # ---- Default values (do not change)
    params = SimParams(object_name=object_name,
                       job_number=0,
                       data_folder="_tmp/data")

    # whether to show the simulation rendering or not
    params.show_gui = True
    # eps_greedy = 0 -> always use the network to sample the interaction point, eps_greedy = 1.0 -> sample interaction point uniformly at random
    params.eps_greedy = 0.
    # task is complete when abs(obj_state - target_obj_pose) < toll
    params.task_completion_tolerance = float(np.deg2rad(1))
    params.robust_cost_percentile = 0.5
    # how much time the robot has to interact with the object in the simulation
    params.interaction_time = 20.0
    # time between each interaction
    params.settle_time = 10.
    # whether the script should dynamically sample the scale and position of the object
    # instead of using the values in the URDF
    params.sampling_network = 'full'
    params.camera_noise = 0.0
    params.network_orientation_encoding = 'quat'
    params.network_cost_representation = 'success'
    params.sampling_strategy = 'eps-greedy'
    params.pose_cost_threshold = 1.0
    params.ik_check = False
    params.object_scale = 0.5
    return params
