"""
Functions and classes used to help deal with reference systems
As standard, a reference system is usually defined as a 4x4 transform matrix
"""
import numpy as np
import transforms3d

from random import random


class Transform:
    @staticmethod
    def from_origin_and_axes(origin, x, y, z):
        mat = np.identity(4)
        mat[:3, 3] = origin
        mat[:3, 0] = x
        mat[:3, 1] = y
        mat[:3, 2] = z
        return mat

    @staticmethod
    def transform_vector(vector: np.array, matrix: np.array):
        """
        :param vector:
        :param matrix: transformation matrix
        :return:
        """
        if matrix.shape != (4, 4) or vector.shape != (3,):
            raise ValueError("Transform.transform_vector size mismatch")
        v = np.append(vector, 1.)
        w = matrix @ v.T
        return w.T[:3]


def normalize(vec: np.array):
    normalizer = np.linalg.norm(vec)
    if normalizer > 0:
        vec /= np.linalg.norm(vec)
    return vec


def get_normal_ref_system(point_xyz, normal_vector, ground_vector) -> np.array:
    """
    :param point_xyz: point coordinates w.r.t. system A
    :param normal_vector: unit vector normal to the surface at point_xyz w.r.t. system A
    :param ground_vector: optional, unit vector normal to the ground w.r.t. system A
    :return: B2A transform, where B is a reference system centered on point_xyz where the z axis is the normal vector and x is parallel to the ground
    """
    # the z axis should enter the surface (oss: the tracked grasp frame on the robot hand has the z axis pointing forwards
    _z = -normalize(normal_vector)
    # x is parallel to the ground and belongs to the surface
    _x = normalize(np.cross(_z, ground_vector))
    if np.abs(np.dot(_z, ground_vector)) > 0.9999:
        # _z is parallel to the ground_vector -> np.cross(_z, ground_vector) will get a vector of norm zero
        _x = np.array([0, 1, 0])
    _y = normalize(np.cross(_z, _x))
    b2a = Transform.from_origin_and_axes(origin=point_xyz, x=_x, y=_y, z=_z)
    return b2a


def sample_ee_direction_angle_uniform():
    # this rotation is relative to the normal_ref_system obtained by get_normal_ref_system
    _z = np.array([0, 0, 1]).T

    # get a random vector in the x,y plane
    _x = 0.0
    _y = 0.0
    while _x == 0.0 and _y == 0.0:
        # make absolutely sure we don't get a [0,0,0] vector
        _x = random() - 0.5
        _y = random() - 0.5
    rotation_axis = normalize(np.array([_x, _y, 0]))

    theta = (random() - 0.5) * 90  # random rotation between -45 and +45 degs
    rotation_matrix = transforms3d.axangles.axangle2mat(axis=rotation_axis, angle=np.deg2rad(theta))

    # now rotate the z axis by the rotation matrix, to get the end-effector direction
    z = rotation_matrix @ _z
    return z


def sample_sphere():
    _i, _j, _k = 0.0, 0.0, 0.0
    while _i == 0.0 and _j == 0.0 and _k == 0.0:
        # make absolutely sure we don't get a [0,0,0] vector
        _i, _j, _k = random() - 0.5, random() - 0.5, random() - 0.5
    return np.array([_i, _j, _k])


def sample_ref_system(direction_distribution: dict):
    z_base = np.array([0., 0., 1.])
    # part 1 - sample ee direction
    if direction_distribution['type'] == 'normal-to-surface':
        # finger orthogonal to the surface
        direction = z_base
    elif direction_distribution['type'] == 'fixed':
        # fixed rotation, not sampled. Robot finger normal to the surface and pointing down
        mat = np.identity(4)
        mat[:3, :3] = transforms3d.axangles.axangle2mat(axis=z_base, angle=np.deg2rad(180))
        return mat
    elif direction_distribution['type'] == 'angle-uniform':
        while True:
            direction = normalize(sample_ee_direction_angle_uniform())
            if np.dot(direction, z_base) > 0: break
    elif direction_distribution['type'] == 'sphere-uniform':
        # sample the entire sphere, even "behind" the surface
        direction = normalize(sample_sphere())
    elif direction_distribution['type'] == 'semi-sphere-uniform':
        # only sample in the semi-sphere corresponding to the surface normal
        while True:
            direction = normalize(sample_sphere())
            if np.dot(direction, z_base) > 0: break
    else:
        raise ValueError("orientation sampling distribution not recognized")

    # the direction becomes the z axis of the sampled ref system
    z = direction

    # part 2 - given a direction, sample ee rotation
    # sample a unit vector
    _x, _y, _z = 0.0, 0.0, 0.0
    random_vector = np.array([_x, _y, _z])
    while (_x == 0.0 and _y == 0.0 and _z == 0.0) and np.abs(np.dot(random_vector, z)) < 0.9999:
        # guard against getting a [0,0,0] vector,
        # or a vector that is parallel to z (would get an invalid x later)
        _x, _y, _z = random() - 0.5, random() - 0.5, random() - 0.5
        random_vector = np.array([_x, _y, _z])
    # this determines the end-effector rotation
    x = normalize(np.cross(z, random_vector))
    y = normalize(np.cross(z, x))
    mat = Transform.from_origin_and_axes([0, 0, 0], x, y, z)
    return mat
