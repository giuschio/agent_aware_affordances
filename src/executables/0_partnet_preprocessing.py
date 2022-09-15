"""
This script allows the user to import objects from PartNet and export them such that
they comply with the conventions required by the code in this repository.
"""

import json
import glob
import os
import pandas as pd
import shutil
import yaml
import xml.etree.ElementTree as ET

from os.path import join as pj

from tools.utils.configuration import Configuration
from tools.utils.misc import dir_exists
from tools.urdf_tools.mesh_tools import combine_meshes_by_link


def load_txt(fpath):
    data = pd.read_csv(fpath, sep=" ", header=None, index_col=0)
    return data.to_dict('index')


def dict_to_yaml(dictionary: dict, save_path: str):
    with open(save_path, 'w') as file:
        yaml.dump(dictionary, file)


def get_joint_types(model_path):
    """
    For each joint in the object, get the joint type (e.g. door or drawer) as a dictionary
    """
    semantics = load_txt(pj(model_path, 'semantics.txt'))
    joint_types = dict()
    for key, link in semantics.items():
        joint_name = key.replace('link', 'joint')
        joint_type = link[2]
        joint_types[joint_name] = joint_type
    return joint_types


def urdf_tree_conditioning(fpath, joint_types):
    """
    Add world joint,
    Add limit effort and velocity tags for movable joints
    Add damping and friction tags to movable joints
    :param fpath: path to urdf file
    :param joint_types: allowed joint types -> everything else will be set to "fixed"
    """
    with open(fpath) as f:
        tree = ET.parse(f)

        for elem in tree.iter():
            if elem.tag == 'robot':
                link = ET.Element('link', attrib=dict(name="world"))
                elem.append(link)
                joint = ET.Element('joint', attrib=dict(name="fixed", type="fixed"))
                parent = ET.Element('parent', attrib=dict(link="world"))
                child = ET.Element('child', attrib=dict(link="base"))
                origin = ET.Element('origin', attrib=dict(xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0"))
                axis = ET.Element('axis', attrib=dict(xyz="1.0 0.0 0.0"))

                joint.append(parent)
                joint.append(child)
                joint.append(origin)
                joint.append(axis)
                elem.append(joint)

            if elem.tag == 'limit':
                elem.attrib['effort'] = "1000.0"
                elem.attrib['velocity'] = "1.0"

            if elem.tag == 'joint' and elem.attrib['type'] in ['revolute', 'prismatic']:
                dynamics = ET.Element('dynamics', attrib=dict(damping="20.0", friction="40.0"))
                elem.append(dynamics)

            if elem.tag == 'joint':
                joint_name = elem.attrib['name']
                if joint_name in joint_types:
                    if joint_types[joint_name] not in ['door', 'drawer', 'rotation_door']:
                        elem.attrib['type'] = "fixed"

    tree.write(fpath, encoding='ASCII')


def urdf_count_movable_joints(fpath):
    with open(fpath) as f:
        tree = ET.parse(f)
        n_joints = 0
        for elem in tree.iter():
            if elem.tag == 'joint' and elem.attrib['type'] != "fixed":
                n_joints += 1
    return n_joints


def urdf_make_one_dof(opath, joint_types, target_joint):
    """
    For an object with multiple joints of type 'door' and 'drawer', this function
    sets all joints to be fixed, except for the one at index target_joint
    :param opath: object path
    :return:
    """
    fpath = pj(opath, 'mobility.urdf')
    with open(fpath) as f:
        tree = ET.parse(f)

        n_joints = 0
        for elem in tree.iter():
            if elem.tag == 'joint':
                joint_name = elem.attrib['name']
                if joint_name in joint_types:
                    if joint_types[joint_name] in ['door', 'drawer', 'rotation_door']:
                        if n_joints == target_joint:
                            pass
                        else:
                            elem.attrib['type'] = 'fixed'
                        n_joints += 1

    tree.write(fpath, encoding='ASCII')


def urdf_conditioning(opath):
    fpath = pj(opath, 'mobility.urdf')
    joint_types = get_joint_types(opath)
    urdf_tree_conditioning(fpath, joint_types)
    n_joints = urdf_count_movable_joints(fpath)
    # if the object has more than one movable joint, make multiple
    # copies, of which each one has one single joint "active"
    ll = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm']
    max_j = len(ll) - 1
    if n_joints > 1:
        current_joint = 0
        # more than one good movable link
        while current_joint < n_joints and current_joint < max_j:
            new_opath = opath + ll[current_joint]
            shutil.copytree(src=opath, dst=new_opath)
            urdf_make_one_dof(new_opath, joint_types, current_joint)
            current_joint += 1

        shutil.rmtree(opath)


def set_mppi_options(opath):
    """
    Set mppi options for each object, including the name of the movable joint
    :param opath:
    :return:
    """
    fpath = pj(opath, 'mobility.urdf')
    joint_name = None
    with open(fpath) as f:
        tree = ET.parse(f)
        for elem in tree.iter():
            if elem.tag == 'joint' and elem.attrib['type'] in ['revolute', 'prismatic']:
                joint_name = elem.attrib['name']
                link_name = joint_name.replace('joint', 'link')

    if joint_name is None:
        shutil.rmtree(opath)
    else:
        mppi_options = dict(dynamics=dict(ignore_object_self_collision=True), cost=dict(tracked_frame='panda_grasp_finger_edge'))
        mppi_options['dynamics']['object_handle_link'] = link_name
        mppi_options['dynamics']['object_handle_joint'] = joint_name
        mppi_options['dynamics']['articulation_joint'] = joint_name
        mppi_options['cost']['handle_frame'] = link_name

        dict_to_yaml(mppi_options, save_path=pj(opath, 'controller_config_overwrite.yaml'))
        dict_to_yaml(mppi_options, save_path=pj(opath, 'simulation_config_overwrite.yaml'))


def divide_by_class(rpath):
    obj_names = os.listdir(rpath)
    classes = dict()
    for oname in obj_names:
        opath = pj(rpath, oname)
        details = pj(opath, "meta.json")
        with open(details) as json_file:
            data = json.load(json_file)
            if "model_cat" not in data:
                print(oname, 'has no category')
                continue
            category = data["model_cat"]
            classes[category] = 0
            dir_exists(pj(rpath, str(category)))
            shutil.move(opath, pj(rpath, str(category), oname))
    return list(classes.keys())


def set_mobility_original(rpath):
    categories = ['StorageFurniture', 'Microwave', 'Door', 'Refrigerator', 'Safe', 'WashingMachine', 'Table']
    for c in categories:
        cat_path = pj(rpath, c)
        if os.path.exists(cat_path):
            print(c)
            obj_names = os.listdir(cat_path)
            for oname in obj_names:
                opath = pj(cat_path, oname)
                shutil.copy(src=pj(opath, 'mobility.urdf'), dst=pj(opath, 'mobility_original.urdf'))


def main(input_path, output_path):
    print("copy objects to output paths...")
    shutil.copytree(input_path, output_path)

    objects_path = output_path
    obj_names = os.listdir(objects_path)
    print("rewriting urdf trees...")
    for oname in obj_names:
        opath = pj(objects_path, oname)
        urdf_conditioning(opath)

    obj_names = os.listdir(objects_path)
    print("writing required YAML files...")
    for oname in obj_names:
        opath = pj(objects_path, oname)
        set_mppi_options(opath)

    obj_names = os.listdir(objects_path)
    print("combining meshes to make the urdf cleaner...")
    for oname in obj_names:
        opath = pj(objects_path, oname, 'mobility.urdf')
        combine_meshes_by_link(opath)

    print("renaming mobility to mobility_original to enable dynamic scaling...")
    for mobility_file in glob.glob(objects_path + '*mobility.urdf'):
        shutil.copy(src=mobility_file, dst=pj(os.path.dirname(mobility_file), 'mobility_original.urdf'))

    print("dividing objects by class...")
    classes = divide_by_class(objects_path)


if __name__ == "__main__":
    # path to directory containing object folders (relative to the src folder)
    input_path = ''
    # output path where processed objects should be saved (relative to the src folder)
    output_path = ''
    main(input_path=Configuration.get_abs(input_path), output_path=Configuration.get_abs(output_path))
