import copy
import numpy as np
import pinocchio
import os
import sapien.core as sapien

from random import random
from typing import List

from tools.utils.misc import StreamingMovingAverage


class Object:
    """
    Assume the Object has only one movable revolute joint
    """

    def __init__(self, path: str, loader: sapien.URDFLoader, material=None, friction_scaling=0.1, rendering_only=True,
                 m_object: sapien.Articulation = None):
        """
        Either load an object using passed parameters, or copy-initialize with an already loaded object
        :param path: path to urdf file defining the object
        :param friction_scaling: allows scaling the sapien friction and damping of articulations w.r.t. those defined in the urdf file
        """
        if m_object is not None:
            self.object = copy.deepcopy(m_object)
        else:
            urdf_path = path if ".urdf" in path else os.path.join(path, "mobility.urdf")
            self.object = self.load(urdf_path, material, loader, friction_scaling, rendering_only)
        self.joint = self.object.get_active_joints()[0]
        self.object_limits = self.joint.get_limits()[0]

    @staticmethod
    def load(urdf_path: str, material, loader: sapien.URDFLoader, friction_scaling=1.0, rendering_only=True) -> sapien.Articulation:
        """
        Load an object from urdf and set properties
        """
        # load as articulation builder, this way I can remove all collision bodies if rendering_only is True
        object_builder = loader.load_file_as_articulation_builder(urdf_path, {"material": material})
        if rendering_only:
            for link_builder in object_builder.get_link_builders():
                link_builder.remove_all_collisions()
        m_object = object_builder.build(fix_root_link=True)
        joints = m_object.get_active_joints()
        if len(joints) != 1:
            raise ValueError("Only objects with one degree of freedom are supported, the object "
                             + urdf_path + " has " + str(len(joints)) + " dofs")
        # set joint controller properties
        joint = joints[0]
        joint.set_drive_property(stiffness=0, damping=joint.damping * friction_scaling)
        joint.set_friction(joint.friction * friction_scaling)
        joint.set_drive_velocity_target(0)
        return m_object

    def get_state(self):
        return np.concatenate((self.object.get_qpos(), self.object.get_qvel()))

    def get_movable_links_ids(self) -> List:
        movable_link_ids = list()
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                movable_link_ids.append(j.get_child_link().get_id())
        return movable_link_ids

    def get_random_joint_state(self):
        return random() * self.object_limits[1] + self.object_limits[0]

    @staticmethod
    def sample_joint_pose(lower, upper):
        if lower > upper:
            raise ValueError("Invalid joint lower and upper bounds")
        return lower + random() * (upper - lower)

    def set_joint_position(self, q):
        self.object.set_qpos(np.array([q]))

    def get_actor(self, link_name) -> sapien.Actor:
        links = self.object.get_links()
        base_actor = None
        for actor in links:
            if actor.get_name() == link_name:
                base_actor = actor
        return base_actor

    def get_transform(self, source_link: str, target_link: str) -> sapien.Pose:
        """
        :param source_link: name of the source link
        :param target_link: name of the target link
        :return: source2target transform
        """
        source_actor = self.get_actor(source_link)
        target_actor = self.get_actor(target_link)
        source2world = source_actor.get_pose()
        target2world = target_actor.get_pose()
        source2target = target2world.inv() * source2world
        return source2target


class Robot:
    default_robot_pose = np.array([0.0, 0.0, 0.0, 0.0, -0.52, 0.0, -1.785, 0.0, 1.10, 0.69, 0.04, 0.04])
    arm_limits_min = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.017, -2.89, 0.04, 0.04])
    arm_limits_max = np.array([2.89, 1.76, 2.89, 0.06, 2.89, 3.75, 2.89, 0.040, 0.040])

    def __init__(self, urdf_path: str, loader: sapien.URDFLoader, base_pose=None):
        self.urdf_path = urdf_path
        self.robot = self.load(urdf_path, loader, base_pose)
        self.hand_link = self.get_actor('panda_hand')

    @classmethod
    def load(cls, urdf_path: str, loader: sapien.URDFLoader, base_pose=None) -> sapien.Articulation:
        """
        Load a robot model from urdf and set its initial pose
        :param base_pose: initial pose of the robot base w.r.t. the world frame
        """
        robot_builder = loader.load_file_as_articulation_builder(urdf_path, {})
        # remove collision bodies in hand and moving base, as they produce too many
        # false positives during collision checking
        hand_links = ['panda_hand', 'panda_leftfinger', 'base_link']
        for link_builder in robot_builder.get_link_builders():
            if link_builder.get_name() in hand_links:
                link_builder.remove_all_collisions()

        robot = robot_builder.build(fix_root_link=True)
        robot_pose = cls.default_robot_pose
        if base_pose is not None:
            robot_pose[:3] = base_pose
        robot.set_qpos(robot_pose)
        # set joint properties
        for idx, joint in enumerate(robot.get_active_joints()):
            if idx < 3:
                joint.set_drive_property(stiffness=0, damping=1000)
                joint.set_friction(1)
            elif idx < 10:
                joint.set_drive_property(stiffness=0, damping=10)
                joint.set_friction(0)
            else:
                # robot finger remains stiff
                joint.set_drive_property(stiffness=100000, damping=500)
                joint.set_friction(10000)
                # set position of the finger
                joint.set_drive_target(0.04)
        return robot

    def get_state(self):
        return np.concatenate((self.get_qpos(), self.get_qvel()))

    def set_qvel(self, v):
        self.robot.set_qvel(v)

    def get_qvel(self):
        return self.robot.get_qvel()

    def set_qpos(self, q):
        self.robot.set_qpos(q)

    def get_qpos(self):
        return self.robot.get_qpos()

    def set_base_pose(self, pose: np.array):
        current_pose = self.robot.get_qpos()
        current_pose[:3] = pose
        self.robot.set_qpos(current_pose)

    def get_actor(self, link_name) -> sapien.Actor:
        links = self.robot.get_links()
        base_actor = None
        for actor in links:
            if actor.get_name() == link_name:
                base_actor = actor
        return base_actor

    def get_transform(self, source_link: str, target_link: str) -> sapien.Pose:
        """
        :param source_link: name of the source link
        :param target_link: name of the target link
        :return: source2target transform
        """
        source_actor = self.get_actor(source_link)
        target_actor = self.get_actor(target_link)
        source2world = source_actor.get_pose()
        target2world = target_actor.get_pose()
        source2target = target2world.inv() * source2world
        return source2target

    def is_still(self, thresholds=(0.1, 0.1, 0.05)):
        hand_speed = np.linalg.norm(self.hand_link.get_velocity())
        q_vel = np.absolute(self.robot.get_qvel())
        base_vel, joint_vel = q_vel[:2], q_vel[2:]
        base_speed = np.linalg.norm(base_vel)
        max_joint_speed = np.max(joint_vel)
        return hand_speed < thresholds[0] and base_speed < thresholds[1] and max_joint_speed < thresholds[2]

    def is_base_still(self, threshold=0.1):
        q_vel = np.absolute(self.robot.get_qvel())
        base_vel, joint_vel = q_vel[:2], q_vel[2:]
        base_speed = np.linalg.norm(base_vel)
        return base_speed < threshold

    @staticmethod
    def sample_base_pose(radius_bounds: List, angle_bounds: List) -> List:
        """
        :param radius_bounds: min and max distance from world origin to robot base origin
        :param angle_bounds: min and max rotation of base wrt world (degrees)
        :return:
        """
        if radius_bounds[0] > radius_bounds[1] or angle_bounds[0] > angle_bounds[1]:
            raise ValueError("Radius or angle bounds incorrect")
        rad = radius_bounds[0] + random() * (radius_bounds[1] - radius_bounds[0])
        theta = np.deg2rad(angle_bounds[0] + random() * (angle_bounds[1] - angle_bounds[0]))

        _x = float(-np.cos(theta) * rad)
        _y = float(-np.sin(theta) * rad)
        _rot = float(theta)
        return [_x, _y, _rot]

    def inverse_kinematics(self, ref2world, base_position, target_frame_name, n_tries=20):
        return Robot.inverse_kinematics_class_method(ref2world=ref2world,
                                                     robot_urdf=self.urdf_path,
                                                     base_position=base_position,
                                                     n_tries=n_tries,
                                                     target_frame_name=target_frame_name)

    @classmethod
    def inverse_kinematics_class_method(cls,
                                        ref2world,
                                        robot_urdf,
                                        base_position,
                                        n_tries=10,
                                        target_frame_name='panda_grasp_finger_edge'):
        # For debugging purposes, it is useful to be able to calle the function statically
        bp = base_position
        # robot joint limits
        minimum = np.concatenate((np.array([bp[0], bp[1], bp[2]]), cls.arm_limits_min))
        maximum = np.concatenate((np.array([bp[0], bp[1], bp[2]]), cls.arm_limits_max))
        pose = np.concatenate((np.array([bp[0], bp[1], bp[2]]), cls.default_robot_pose[3:]))
        model = pinocchio.buildModelFromUrdf(robot_urdf)
        for i in range(n_tries):
            success, q = cls.inverse_kinematics_algo(ref2world, model, pose, target_frame_name)
            if success:
                return True, q
            pose = np.random.uniform(low=minimum, high=maximum)
        return False, None

    @staticmethod
    def inverse_kinematics_algo(ref2world,
                                model,
                                init_pose,
                                target_frame_name='panda_grasp_finger_edge'):
        # adapted from https://github.com/stack-of-tasks/pinocchio/blob/master/doc/b-examples/i-inverse-kinematics.md
        # see https://github.com/stack-of-tasks/pinocchio/issues/802
        # model = pinocchio.buildModelFromUrdf(robot_urdf)
        data = model.createData()
        FRAME_ID = model.getFrameId(target_frame_name)
        BASE_FRAME_ID = model.getFrameId('base_link')
        oMdes = pinocchio.SE3(ref2world[:3, :3], ref2world[:3, 3])

        q = init_pose
        eps = 1e-2
        IT_MAX = 1000
        DT = 1e-1
        damp = 1e-12

        filter_window_size = 20
        convergence_check_period = 10
        improvement_threshold = 0.1

        success = False
        window = StreamingMovingAverage(filter_window_size)
        last_avg = 1000
        for idx in range(IT_MAX):
            pinocchio.forwardKinematics(model, data, q)
            pinocchio.updateFramePlacement(model, data, FRAME_ID)
            dMi = oMdes.actInv(data.oMf[FRAME_ID])
            err = pinocchio.log(dMi).vector
            err_n = np.linalg.norm(err)
            window.add(err_n)
            if err_n < eps:
                success = True
                break
            if idx % convergence_check_period == 0 and idx != 0:
                # calculate the moving average of the error norm
                curr_avg = window.average
                if curr_avg < (last_avg * (1 - improvement_threshold)):
                    # check that the average error has improved by at least improvement_threshold%
                    last_avg = curr_avg
                else:
                    # error has converged to a local minimum
                    success = False
                    break
            J = pinocchio.computeFrameJacobian(model, data, q, FRAME_ID)
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(model, q, v * DT)

        if success:
            finger_pose = np.array(data.oMf[FRAME_ID])
            finger_x, finger_y = finger_pose[0, 3], finger_pose[1, 3]
            pinocchio.updateFramePlacement(model, data, BASE_FRAME_ID)
            base_pose = np.array(data.oMf[BASE_FRAME_ID])
            base_x, base_y = base_pose[0, 3], base_pose[1, 3]
            delta_x = abs(finger_x - base_x)
            delta_y = abs(finger_y - base_y)
            # base dimensions are (x,y) = (0.90 0.75)
            if delta_x < 0.45 or delta_y < 0.375:
                success = False
        return success, q


class FlyingHand(Robot):
    default_robot_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.57, 0.78, 0.04, 0.04])
    arm_limits_min = np.array([-10, -10, -10, -10, -30, -30, -30, 0.04, 0.04])
    arm_limits_max = np.array([10, 10, 10, 10, 30, 30, 30, 0.040, 0.040])
