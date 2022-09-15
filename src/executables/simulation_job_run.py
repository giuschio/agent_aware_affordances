"""
This script runs a simulation job given an object, task and network
See main for instructions and options
"""
import multiprocessing

from executables.core.simulation_job_core import simulation_job, build_params

if __name__ == '__main__':
    # ---- HARDCODE PARAMS FOR REPEATABILITY DURING TESTING
    # path to the object folder (relative to the src directory)
    object_name = 'demo_objects/demo_oven'
    params = build_params(object_name)

    # ---- TODO: USER-DEFINED JOB PARAMETERS
    # use an agent-aware or end-effector-aware model
    model_type = 'agent-aware'  # 'ee-aware'
    # initial state of the object door (0.0 = closed, 1.57 = fully open)
    params.initial_object_pose = 0.0
    params.target_object_pose = 1.57
    # init pose of the robot base [x, y, theta]
    params.initial_robot_base_pose = [-2., 0., 0.]
    # position of the object center [x, y, z]
    params.object_center_xyz = [0., 0., 0.75]
    # 1.0 is a realistic scale for the demo_oven
    params.object_scale = 1.0

    # maximum number of interactions allowed (if 1, the reference interaction pose is not updated)
    params.closed_loop = 6
    # save simulation here (interaction data e.g. for replay)
    params.data_folder = "_tmp/data"
    # ---- END OF CONFIGURABLE JOB PARAMETERS ---

    network_checkpoint = "demo_models/"
    network_checkpoint += "open_" if params.initial_object_pose < params.target_object_pose else "close_"
    network_checkpoint += model_type
    network_checkpoint += ".pt"

    shared_lock = multiprocessing.Lock()
    simulation_job(sim_params=params,
                   lock=shared_lock,
                   network_checkpoint=network_checkpoint,
                   log_to_screen=True)
