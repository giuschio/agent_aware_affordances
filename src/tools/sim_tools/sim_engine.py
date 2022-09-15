"""
       _
    ___|
   [n n]
 o=     =o
   |GS_|
    d b

This file contains the simulation engine (i.e. the core of the simulator)
The simulation engine deals with rendering (via SAPIEN) and physics (via MPPI interface) and defines
several functions and objects that are called during a simulation_job
"""
import numpy as np
import os
import sapien.core as sapien

from sapien.utils.viewer import Viewer
from typing import List

from tools.sim_tools.actors import Object, Robot, FlyingHand
from tools.sim_tools.camera import Camera
from tools.sim_tools.sim_data import TrajectoryData, SimParams
from tools.utils.misc import StreamingMovingAverage
from tools.utils.configuration import Configuration
from tools.utils.time_utils import Clock

from tools_mppi.mppi import Controller, Simulation, Observation


class MockViewer:
    """
    Used during simulations when rendering is not needed (instead of a lot of if/else statements)
    """
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def render(self):
        pass


class SimulationEngine:
    def __init__(self, params: SimParams, data: TrajectoryData):
        self.params = params
        self.data = data

        # initialize engine and renderer
        self.engine = sapien.Engine()
        self.engine.set_log_level('critical')
        self.renderer = sapien.VulkanRenderer(offscreen_only=not self.params.show_gui)
        self.engine.set_renderer(self.renderer)

        # initialize scene and lights
        # (Sapien is only used for rendering and collision checking)
        scene_config = sapien.SceneConfig()
        scene_config.gravity = np.zeros((3, 1))
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1/100)
        self.rscene = self.scene.get_renderer_scene()
        self.rscene.set_ambient_light(np.array([0.5, 0.5, 0.5]))
        self.rscene.add_directional_light(np.array([0, 1, -1]), np.array([0.5, 0.5, 0.5]), shadow=False)

        # initialize Viewer
        if self.params.show_gui:
            self.viewer = Viewer(self.renderer, resolutions=params.render_resolution)
            self.viewer.toggle_camera_lines(False)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(x=-1, y=-2.5, z=1.5)
            self.viewer.set_camera_rpy(r=0, p=0, y=-1.57)
        else:
            self.viewer = MockViewer()

        robot_urdf = Configuration.get_abs(self.params.robot_model)
        object_urdf = Configuration.get_abs(os.path.join(self.params.object_name, "mobility.urdf"))

        # load simulation assets into the scene
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True

        # sample robot initial pose and load robot onto the scene
        if "full" in Configuration.robot_type:
            self.robot = Robot(robot_urdf, loader, base_pose=np.array(self.params.initial_robot_base_pose))
        elif Configuration.robot_type == "hand":
            self.robot = FlyingHand(robot_urdf, loader, base_pose=np.array(self.params.initial_robot_base_pose))
        # load Object
        self.object = Object(object_urdf, loader, rendering_only=False)
        self.object.set_joint_position(self.params.initial_object_pose)

        # load camera
        camera_offset = sapien.Pose(p=self.params.camera_mount_offset[:3], q=self.params.camera_mount_offset[3:])
        self.camera = Camera(scene=self.scene,
                             mount_actor=self.robot.get_actor(self.params.camera_mount_actor),
                             offset=camera_offset,
                             noise=params.camera_noise)

        # initialize mppi controller and mppi simulation
        self.controller = None
        self.simulation = None
        self.current_observation = np.array(params.simulation_params["dynamics"]["initial_state"])
        self.current_input = Controller.get_zero_input()
        self.current_cost = None
        self.invalid_contact = 0.

        # initialize simulation clock
        self.clock = None

    def load_controller(self):
        self.controller = Controller(self.params.controller_params["fpath"])
        self.controller.initialize_controller()

        self.simulation = Simulation(self.params.simulation_params["fpath"])
        self.current_observation = self.simulation.get_state()
        self.current_input = self.controller.get_zero_input()
        self.current_cost = None

        # initialize simulation clock
        sim_rate = 1.0 / self.simulation.get_dt()
        controller_rate = 1.0 / self.params.controller_params["dynamics"]["dt"]
        self.clock = Clock(sim_rate=sim_rate, controller_rate=controller_rate, render_rate=12)

    def render_scene(self):
        self.scene.update_render()
        if self.params.show_gui:
            self.viewer.render()

    def step(self):
        """
        This should be the only function in the SimulationEngine class that generates entries for SimulationData
        :return:
        """
        if self.clock.is_controller_update():
            # update mppi controller and simulation
            self.controller.set_observation(self.current_observation, self.clock.time())
            self.controller.update_policy()
            self.current_input = self.controller.get_input(self.current_observation, self.clock.time())
        self.current_observation = self.simulation.step(self.current_input, self.clock.sim_timestep)
        self.current_cost = self.controller.get_cost_map(self.current_observation, self.current_input, self.clock.time())

        if self.clock.is_controller_update():
            # rule: observations are logged only at controller update
            # if this is changed, you need to also change the cost normalization
            # push generated data to the SimulationData class
            self.data.time.append(self.clock.time())
            self.data.inputs.append(self.current_input)
            self.data.observations.append(self.current_observation)
            self.data.costs.append(self.current_cost)

        if self.params.show_gui and self.clock.is_render_update():
            # update SAPIEN scene
            self.object.set_joint_position(Observation.object_qpos(self.current_observation))
            self.robot.set_qpos(Observation.robot_qpos(self.current_observation))
            self.scene.update_render()
            self.viewer.render()

        # as a last step, update the clock
        self.clock.step()

    def run_settle(self) -> bool:
        self.clock.set_timer()
        while not self.viewer.closed and not self.clock.get_timer(self.params.settle_time):
            self.step()
        return True

    def run_movement(self, log) -> (bool, List):
        """
        :return: reference pose reached (within a tolerance)
        """
        self.clock.set_timer()
        window = StreamingMovingAverage(self.clock.sim_rate * 1)
        last_avg = 1.0e10
        while not self.viewer.closed and not self.clock.get_timer(self.params.movement_time):
            self.step()
            pose_cost = self.current_cost["pose_cost"]
            window.add(pose_cost)
            if pose_cost < self.params.pose_cost_threshold:
                log.append("objective reached within tolerance")
                return True
            if self.clock.tick(self.params.pose_cost_drop_time):
                curr_avg = window.average
                if curr_avg < (last_avg * (1 - self.params.pose_cost_min_percentage_drop)):
                    # True if the curr_average has decreased by at least toll %
                    last_avg = curr_avg
                else:
                    log.append("objective not reached - cost not improving")
                    return False
        log.append("objective not reached - max time reached")
        return False

    def check_invalid_contacts(self) -> bool:
        # sanity check
        if self.current_observation[26] > 0 and np.sum(np.abs(self.current_observation[-3:])) < 1e-8:
            self.invalid_contact += 1
        else:
            self.invalid_contact = 0

        thresh = self.clock.sim_rate/10.0

        # invalid contact for more than 0.1 seconds
        return self.invalid_contact > thresh

    def run_interaction(self, log, closed_loop=False) -> float:
        """
        :return: task completed (within a tolerance)
        """
        self.clock.set_timer()
        obj_pose = Observation.object_qpos(self.current_observation)
        old_obj_cost = 1.0e10
        while not self.viewer.closed and not self.clock.get_timer(self.params.interaction_time):
            self.step()
            obj_pose = Observation.object_qpos(self.current_observation)
            if self.check_invalid_contacts() is True:
                log.append("invalid contact for more than 10 consecutive steps")
                # mark the simulation as non-successful
                obj_pose = self.params.initial_object_pose
                break
            if abs(obj_pose - self.params.target_object_pose) < self.params.task_completion_tolerance:
                break

            if closed_loop:
                flag_obj_cost = False
                # flag_pose_cost = False
                # break if the object cost has not improved in the last 5 seconds
                if self.clock.tick(5.0):
                    new_obj_cost = self.current_cost['object_cost']
                    flag_obj_cost = bool(new_obj_cost/old_obj_cost > 0.9)
                    old_obj_cost = new_obj_cost if new_obj_cost > 0. else 1.0e10

                # flag_pose_cost = bool(self.current_cost['pose_cost'] > 1.0)

                if flag_obj_cost:
                    log.append("closed loop: breaking interaction")
                    break

        return obj_pose
