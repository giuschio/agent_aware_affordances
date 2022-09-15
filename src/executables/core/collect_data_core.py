import typing

from datetime import datetime
from copy import deepcopy

from tools.sim_tools.simulation_multi import collect_data
from tools.sim_tools.sim_data import SimParams
from tools.utils.configuration import Configuration

DEFAULT_N_PROCESSES_OFFLINE = 3
DEFAULT_N_PROCESSES_ONLINE = 3


def sim_params_eval(params: SimParams) -> SimParams:
    params.eps_greedy = 0.
    if params.closed_loop > 1:
        # use longer times when testing the full closed-loop pipeline
        params.interaction_time = 200.
        params.settle_time = 15.
    else:
        # use shorter times when testing pose quality (sample success rate)
        params.movement_time = 40.
        params.interaction_time = 30.

    # deactivate these to increase the chances we reach the point
    params.pose_cost_drop_time = 100.
    params.perform_ik_check = False
    return params


def sim_params_collection(params: SimParams) -> SimParams:
    params.eps_greedy = 1.0
    params.network_checkpoint = None
    params.interaction_time = 10.
    return params


def set_params(params: SimParams, task: str, testing: bool) -> SimParams:
    params = sim_params_eval(params) if testing else sim_params_collection(params)
    if task == "open":
        params.task_weights = {'open-zero': 0.5, 'close-ninety': 0., 'open': 0.5, 'close': 0.}
    elif task == "full_open":
        params.task_weights = {'open-zero': 1., 'close-ninety': 0., 'open': 0., 'close': 0.}
        params.initial_object_pose = 0.
        params.target_object_pose = 1.57
    elif task == "close":
        params.task_weights = {'open-zero': 0., 'close-ninety': 0.5, 'open': 0., 'close': 0.5}
    elif task == "full_close":
        params.task_weights = {'open-zero': 0., 'close-ninety': 1., 'open': 0., 'close': 0.}
        params.initial_object_pose = 1.57
        params.target_object_pose = 0.

    return params


def data_collection(objects: typing.Dict[str, typing.List[str]],
                    n_jobs: int,
                    common_params: SimParams,
                    testing: bool = True,
                    n_processes: typing.Optional[int] = None):
    cp = common_params
    if testing and cp.network_checkpoint is None: print('[WARNING]: network checkpoint is None. Are you sure you are testing?')

    if n_processes is None:
        n_processes = DEFAULT_N_PROCESSES_OFFLINE if cp.network_checkpoint is None else DEFAULT_N_PROCESSES_ONLINE
    sim_list = list()
    if cp.data_folder is None:
        cp.data_folder = "data_" + datetime.today().strftime('%Y%m%d_%H%M%S')

    print("\n\nstarting data collection with the following arguments:")
    print("\tnumber of jobs:", n_jobs)
    print("\tnumber of processes:", n_processes)
    print("\tsave location:", cp.data_folder)
    print("\tsimulation objects:", objects)
    print("\torientation sampling mode:", cp.interaction_orientation_sampling_mode)
    print("\tusing network:", cp.network_checkpoint)
    print("\tcost representation:", cp.network_cost_representation)
    print("\tcurrent robot type:", Configuration.robot_type)
    print("\tclosed-loop:", cp.closed_loop)
    print("\tTESTING:", testing)

    obj_classes = list(objects.keys())
    class_dict = {k: 0 for k in obj_classes}

    for idx in range(n_jobs):
        # get a class
        current_class = obj_classes[idx % len(obj_classes)]
        # sample uniformly within the class
        object_name = objects[current_class][class_dict[current_class] % len(objects[current_class])]
        class_dict[current_class] += 1
        params = deepcopy(cp)
        params.object_name = object_name
        params.job_number = idx
        sim_list.append(params)

    time_per_job = 30. if testing else 10.
    collect_data(simulations_list=sim_list,
                 n_processes=n_processes,
                 chunk_size=500,
                 time_per_job=time_per_job,
                 network_checkpoint=cp.network_checkpoint)

    return cp.data_folder
