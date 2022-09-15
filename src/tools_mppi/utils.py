import os

from tools.utils.data import yaml_to_dict, dict_to_yaml, update_dictionary
from tools.utils.configuration import Configuration


def build_mppi_config(src2obj_path: str,
                      initial_base_pose=None,
                      initial_object_state=None,
                      kind: str = "controller"):
    """
    Create a full config file for mppi out of the default config + overwrite.
    This function is specific to the way the object dataset has been set up
    :param kind: either "controller" or "simulation"
    :param initial_object_state: objest articulation state at time zero
    :param initial_base_pose: robot base pose at time zero
    :param src2obj_path: path of object model relative to the src directory
    :return: path to full config
    """
    object_abs_path = Configuration.get_abs(src2obj_path)
    short = "hd_" if Configuration.robot_type == "hand" else "rd_"
    fname = os.path.join(object_abs_path, kind + "_config_overwrite.yaml")
    if os.path.isfile(fname):
        config_overwrite = yaml_to_dict(fname)
        # get absolute path of mppi library folder
        mppi_abs_path = Configuration.get_abs(Configuration.src2mppi)
        default_config = yaml_to_dict(os.path.join(mppi_abs_path, short + kind + "_config.yaml"))
        full_config = update_dictionary(default_config, config_overwrite)

        # setup object-specific absolute paths needed for pymppi
        full_config["dynamics"]["object_description_raisim"] = os.path.join(object_abs_path, "mobility.urdf")
        full_config["dynamics"]["object_description"] = os.path.join(object_abs_path, "mobility.urdf")
        full_config["dynamics"]["raisim_object_res_path"] = object_abs_path
        full_config["dynamics"]["raisim_robot_res_path"] = Configuration.get_abs(Configuration.src2robot_folder)

        # setup robot-specific absolute paths needed for pymppi
        if "controller" in kind:
            raisim_model = Configuration.src2robot_controller_raisim
        else:
            raisim_model = Configuration.src2robot_simulation_raisim
        full_config["dynamics"]["robot_description_raisim"] = Configuration.get_abs(raisim_model)
        full_config["dynamics"]["robot_description"] = Configuration.get_abs(Configuration.src2robot)

        # setup absolute path to solver configuration for pymppi
        full_config["options"]["solver_config_file"] = os.path.join(mppi_abs_path, short + kind + "_solver_config.yaml")

        if initial_base_pose is not None:
            # setup initial robot state (need to set the initial base pose dynamically)
            full_config["dynamics"]["initial_state"][:3] = initial_base_pose[:]

        if initial_object_state is not None:
            # setup initial object state (need to set the initial base pose dynamically)
            full_config["dynamics"]["initial_state"][24] = initial_object_state

        fpath = os.path.join(object_abs_path, kind + "_config_full_generated.yaml")
        full_config["fpath"] = fpath
        dict_to_yaml(full_config, fpath)

        return full_config
    else:
        raise ValueError("No file in " + fname)
