"""
This script runs multiple simulation jobs given a set of objects and params.
It can:
  - use random interaction points to collecting training data
  - use a pre-trained network to evaluate its performance
Before running it, some user-defined params must be set (see below)
"""

from torch import multiprocessing as multiprocessing

from executables.core.collect_data_core import data_collection, set_params

from tools.sim_tools.sim_data import SimParams
from tools.sim_tools.simulation_multi import get_objects


if __name__ == '__main__':
    # set mp to spawn (otherwise we get weird behavior from pytorch)
    multiprocessing.set_start_method('spawn')
    # ---- GET OBJECT LISTS
    # get demo oven (always available)
    demo_object = ["demo_objects/demo_oven"]

    # ---- USER-DEFINED JOB PARAMETERS
    # todo: object models (other than the demo_oven) are not provided.
    #   Please create your own "data/objects" folders and change the paths below
    #   get_objects: directory path (w.r.t. src) -> list of objects in that directory
    objects_dir_path = "demo_objects"
    objects_list = get_objects(objects_dir_path)
    # this should be objects = {class_name_1: list(object_dirs), ... , class_name_n, list(object_dirs)}
    # objects are sampled uniformly and balanced by class
    objects = {'Objects': objects_list}

    # todo: Select overall parameters:
    # TASK
    #   - "open" -> initial state closed or random (50/50 split). Desired state sampled at random
    #   - "full_open" -> initial state closed. Desired state fully-open (90 degs)
    #   - "close" -> ...
    #   - "full_close" -> ...
    # EXAMPLE --> task 'open': training data collection
    task = "open"
    closed_loop = 1    # max number of interactions allowed (1 = open-loop affordances)
    # If network_checkpoint is provided, it will be used to sample the interaction pose
    # if None, the pose is sampled at random
    network_checkpoint = None
    testing = False    # sim params change during testing (e.g. time allowed for interaction)
    n_jobs = 10        # number of simulation jobs
    out_path = None    # savepath of results. If None, defaults to data_YYYYmmdd

    # EXAMPLE --> testing 'open' model with full task (zero to fully-open)
    # task = "full_open"
    # closed_loop = 6   # max number of interactions allowed (1 = open-loop affordances)
    # network_checkpoint = 'demo_models/open_agent-aware.pt'
    # testing = True    # sim params change during testing (e.g. time allowed for interaction)
    # n_jobs = 10       # number of simulation jobs
    # out_path = None   # savepath of results. If None, defaults to data_YYYYmmdd
    # todo: for advanced users: more params can be set through the common_params variable below
    # ---- END OF USER-DEFINED JOB PARAMETERS

    # set common parameters for all simulations (see SimParams class for param definition)
    common_params = SimParams()
    common_params = set_params(common_params, task=task, testing=testing)
    common_params.closed_loop = closed_loop
    common_params.network_checkpoint = network_checkpoint
    common_params.data_folder = out_path
    # realistic scale bounds might be different for other object categories
    # common_params.object_scale_bounds = [0.9, 1.1]

    data_collection(objects=objects, n_jobs=n_jobs, common_params=common_params, testing=testing)
