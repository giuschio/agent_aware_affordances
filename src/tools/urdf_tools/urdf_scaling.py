"""
Functions used to dynamically edit URDF object definitions
"""

import logging
import numpy as np
import typing
import xml.etree.ElementTree as ET

from yourdfpy import URDF, Visual, Collision


def get_scaling_matrix(scale):
    """
    Used to scale the origin dimensions
    """
    sm = np.ones((4, 4))
    sm[0, 3], sm[1, 3], sm[2, 3] = scale, scale, scale
    return sm


def scale_visual(visual: Visual, scale=1.0):
    """
    Scale visual meshes (in-place)
    """
    old_scale = visual.geometry.mesh.scale
    if old_scale is None:
        old_scale = 1.0
    else:
        old_scale = old_scale[0]
    relative_scale = scale / old_scale
    # this is an annotation
    visual.geometry.mesh.scale = scale
    # here instead we are multiplying matrices, so we need to take into account the old scale
    visual.origin = np.multiply(visual.origin, get_scaling_matrix(relative_scale))
    return relative_scale


def scale_collision(collision: Collision, scale=1.0):
    """
    Scale collision meshes (in-place)
    """
    old_scale = collision.geometry.mesh.scale
    if old_scale is None:
        old_scale = 1.0
    else:
        old_scale = old_scale[0]
    relative_scale = scale / old_scale
    # this is an annotation
    collision.geometry.mesh.scale = scale
    # here instead we are multiplying matrices, so we need to take into account the old scale
    collision.origin = np.multiply(collision.origin, get_scaling_matrix(relative_scale))
    return relative_scale


def scale_urdf_util(model: URDF, scale: typing.Optional[float] = None) -> URDF:
    """
    Generate a copy of the urdf model with the required scale
    """
    relative_scale = 1.0
    if scale is not None:
        # all scaling operations are done in-place
        for link_name, link in model.link_map.items():
            for collision in link.collisions:
                # in-place (i.e. collision is passed by reference)
                scale_collision(collision, scale)
            for visual in link.visuals:
                relative_scale = scale_visual(visual, scale)

        for joint_name, joint in model.joint_map.items():
            joint.origin = np.multiply(joint.origin, get_scaling_matrix(relative_scale))
            if joint.type == 'prismatic':
                # is the joint is prismatic, the joint limits also need to be scaled
                joint.limit.upper *= scale
                joint.limit.lower *= scale

    return model


def scale_urdf(model_path, new_model_path, mesh_dir, scale: typing.Optional[float] = None):
    """
    Save a scaled copy of a urdf model
    :param scale: new absolute scale of the URDF model
    :param model_path:
    :param new_model_path:
    :param mesh_dir: directory relative to which the mesh paths are written in the urdf
    """
    logging.disable()
    model = URDF.load(model_path, mesh_dir=mesh_dir)
    new_model = scale_urdf_util(model, scale=scale)
    new_model.write_xml_file(new_model_path)


def move_urdf_util(model: URDF, center_xyz: typing.Optional[typing.List] = None) -> URDF:
    """
    Generate a copy of the urdf model with the required scale
    """
    if center_xyz is not None:
        joint_map = model.joint_map
        world_joint = joint_map['fixed']
        # world_joint.origin is a 4x4 matrix
        world_joint.origin[0, 3] = center_xyz[0]
        world_joint.origin[1, 3] = center_xyz[1]
        world_joint.origin[2, 3] = center_xyz[2]
    return model


def move_urdf(model_path, new_model_path, mesh_dir, center_xyz: typing.Optional[typing.List] = None):
    """
    Save a scaled copy of a urdf model
    :param center_xyz: new center position of the model
    :param model_path:
    :param new_model_path:
    :param mesh_dir: directory relative to which the mesh paths are written in the urdf
    """
    logging.disable()
    model = URDF.load(model_path, mesh_dir=mesh_dir)
    new_model = move_urdf_util(model, center_xyz=center_xyz)
    new_model.write_xml_file(new_model_path)


def set_urdf_joint_dynamics(model: URDF, damping: float, friction: float) -> URDF:
    """
    Generate a copy of the urdf model with the required scale
    """
    for joint_name, joint in model.joint_map.items():
        if joint.type == 'fixed': continue
        joint.dynamics.damping = damping
        joint.dynamics.friction = friction
    return model


def add_steel_contact(fpath):
    """
    The material(steel) tag is not URDF standard, and gets thrown out
    when I manipulate the URDF files with yourdfpy for scaling.
    This function is a path: I directly write to the XML the material tag again
    :param fpath:
    :return:
    """
    with open(fpath) as f:
        tree = ET.parse(f)
        for elem in tree.iter():
            if elem.tag == 'collision':
                material = ET.Element('material', attrib=dict(name="steel"))
                contact = ET.Element('contact', attrib=dict(name="steel"))
                material.append(contact)
                elem.append(material)

    tree.write(fpath, encoding='ASCII')

