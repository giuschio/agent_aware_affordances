import numpy as np
import sapien.core as sapien

from tools_mppi import pymppi_manipulation as mppi
from typing import List

ARM_DOFS = 7
BASE_DOFS = 3
GRIPPER_DOFS = 2
ROBOT_DOFS = ARM_DOFS + BASE_DOFS + GRIPPER_DOFS


class Reference:
    def __init__(self, pose, mode=0, target_pose=0):
        self.p = np.zeros(3)
        self.q = np.zeros(4)
        self.arm_j = np.zeros(ARM_DOFS)
        self.base_j = np.zeros(BASE_DOFS)  # [x, y, theta]
        self.set_pose(pose)
        self.obstacle_position = np.zeros(3)
        self.object_target_pose = target_pose
        self.mode = mode

    def set_pose(self, pose: sapien.Pose):
        self.p = pose.p
        # sapien and mppi have different quaternion conventions
        # in sapien quaternions are written as (w,x,y,z), in mppi as (x,y,z,w), where w is the real part
        mppi_q = [pose.q[1], pose.q[2], pose.q[3], pose.q[0]]
        self.q = np.array(mppi_q)

    def serialize(self):
        if self.mode == 3:
            res = np.concatenate((self.arm_j, self.base_j, np.array([self.object_target_pose]), np.array([self.mode])))
        else:
            res = np.concatenate((self.p, self.q, self.obstacle_position, np.array([self.object_target_pose]), np.array([self.mode])))
        return res

    def __str__(self):
        if self.mode == 3:
            s1 = "ref = [arm joints: " + np.array2string(self.arm_j, precision=2)
            s2 = "base joints: " + np.array2string(self.base_j, precision=2)
            s3 = ""
        else:
            s1 = "ref = [p.x: {:0.2f}, p.y: {:0.2f}, p.z: {:0.2f}, ".format(self.p[0], self.p[1], self.p[2])
            s2 = "q.x: {:0.2f}, q.y: {:0.2f}, q.z: {:0.2f}, q.w: {:0.2f}, ".format(self.q[0], self.q[1], self.q[2], self.q[3])
            s3 = "{:0.2f}, {:0.2f}, {:0.2f}, ".format(self.obstacle_position[0], self.obstacle_position[1], self.obstacle_position[2])
        s4 = "target: {:0.2f}, mode: {}]".format(self.object_target_pose, self.mode)
        return s1 + s2 + s3 + s4


class Observation:
    """
    Convenience functions to avoid using hard-coded values in the rest of the code
    """

    @staticmethod
    def robot_qpos(observation):
        return observation[:ROBOT_DOFS]

    @staticmethod
    def robot_qvel(observation):
        return observation[ROBOT_DOFS:2 * ROBOT_DOFS]

    @staticmethod
    def object_qpos(observation):
        return observation[2 * ROBOT_DOFS]


class Controller(mppi.PandaControllerInterface):
    """
    Wrapper class for mppi.PandaControllerInterface, provides documentation
    """

    def __init__(self, config_file_path: str):
        """
        :param config_file_path: absolute path to the configuration file
        """
        super().__init__(config_file_path)

    def initialize_controller(self):
        super().init()

    def set_observation(self, observation: np.array, current_time: float):
        # dimension of obs is (39,) see configs for details
        super().set_observation(observation, current_time)

    def set_reference(self, references: List[np.array], time_list: List[float]):
        super().set_reference(references, time_list)

    def update_policy(self):
        super().update_policy()

    @staticmethod
    def get_zero_input():
        return np.zeros(ROBOT_DOFS)

    def get_input(self, observation: np.array, current_time: float):
        """
        :param observation: current observed state
        :param current_time: current simulation time
        :return: velocity commands
        """
        # dimension of obs is (39,) see configs for details
        return super().get_input(observation, current_time)

    def get_stage_cost(self, observation: np.array, u: np.array, current_time: float):
        return super().get_stage_cost(observation, u, current_time)

    def get_cost_map(self, observation: np.array, u: np.array, current_time: float):
        return super().get_cost_map(observation, u, current_time)


class MockController:
    """
    This mock controller can be used to import a controller in a simulation without having to worry about
    compatibility with pymppi. It also conveniently exposes the same API as pymppi, such that the two
    can be exchanged without having to change the rest of the code
    """

    def __init__(self, config_file_path: str):
        pass

    def initialize_controller(self):
        pass

    def set_observation(self, observation: np.array, current_time: float):
        pass

    def set_reference(self, references: List[np.array], time_list: List[float]):
        pass

    def update_policy(self):
        pass

    def get_input(self, observation: np.array, current_time: float):
        """
        :param observation: current observed state
        :param current_time: current simulation time
        :return: velocity commands
        """
        return np.zeros(ROBOT_DOFS)

    def get_stage_cost(self, observation: np.array, u: np.array, current_time: float):
        return 0.0

    def get_cost_map(self, observation: np.array, u: np.array, current_time: float):
        return dict()


class Simulation(mppi.PandaRaisimDynamics):
    def __init__(self, config_file_path):
        """
        :param config_file_path: absolute path to the configuration file
        """
        super().__init__(config_file_path)

    def step(self, u: np.array, dt: float):
        return super().step(u, dt)

    def get_dt(self):
        return super().get_dt()

    def get_state(self):
        return super().get_state()
