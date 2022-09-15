import numpy as np
import os
import pandas
import typing

from tools.sim_tools.camera import Camera
from tools.utils.data import dict_to_yaml, yaml_to_dict
from tools.utils.configuration import Configuration


class SimParams:
    """
    This class contains all parameters that define a simulation job instance, and their default value
    It can be saved to a YAML file and loaded
    """
    def __init__(self, object_name=None, job_number=None, data_folder=None):
        # simulation arguments (i.e. these are set before the Simulation starts)
        self.object_name: str = object_name       # object path relative to the src directory
        self.job_number: int = job_number
        self.process_number: int = 0
        self.data_folder: str = data_folder
        self.dynamically_build_urdf: bool = True  # True: sample object scale and position, False: use values in the URDF
        self.savepath: typing.Optional[str] = None
        self.render_resolution: typing.Tuple[typing.Tuple[int, int]] = ((1080, 720),)

        self.network_cost_representation: str = 'success'  # or 'continuous'
        self.closed_loop: int = 1                          # maximum number of interactions (1 = open-loop)

        self.object_scale_bounds: typing.List = [0.9, 1.1]
        self.object_center_xyz_bounds: typing.List = [0, 0, -0.3, 0.3, 0.75, 0.95]
        # sub-task sampling
        self.task_weights: dict = {'open-zero': 0.25, 'close-ninety': 0.25, 'open': 0.25, 'close': 0.25}
        self.task: typing.Optional[str] = None

        self.art_damping: float = 20.   # damping of the object articulated joint
        self.art_friction: float = 40.  # friction of the object articulated joint
        self.robot_model: str = Configuration.src2robot_simulation_raisim
        self.robot_model_controller: str = Configuration.src2robot_controller_raisim
        self.robot_initial_distance_bounds: typing.List = [1.75, 3]
        self.robot_initial_angle_bounds: typing.List = [-20, 20]
        self.camera_mount_actor: str = "panda_link0"
        # identity quaternion, SAPIEN convention (p.x, p.y, p.z, q.w, q.x, q.y, q.z)
        self.camera_mount_offset: typing.List = [0, 0, 0, 1, 0, 0, 0]

        self.show_gui: bool = False
        self.settle_time: float = 0.    # 'rest' time between each interaction
        self.movement_time: float = 30.
        self.interaction_time: float = 5.
        self.sampling_strategy: str = 'eps-greedy'  # other options are 'eps-95' 'thompson'
        self.eps_greedy: float = 1.0    # 1.0 = only exploration, 0.0 = only exploitation
        self.network_checkpoint: typing.Optional[str] = None
        # Reduce time for training data collection by performing feasibility checks on the sampled pose.
        # If the check fails, the simulation/task is labeled as unsuccessful
        self.perform_ik_check: bool = True
        self.perform_collision_check: bool = False
        # Improve test performance by using a feasibility filter
        # Use the network to sample poses, and choose one that passes the IK and collision check
        # todo: if filter_*check is True, the sampling_strategy should be different than the default eps-greedy (thompson is recommended)
        #       Otherwise, the network will keep sampling the same pose over and over
        self.filter_ik_check: bool = False
        self.filter_collision_check: bool = False
        # for each interaction, max number of network proposals that can fail the feasibility checks before
        # the task is considered unsuccessful
        self.testing_max_pose_proposals: int = 100
        # logs the number of feasibility check performed during one sim instance
        self.number_feas_checks: int = 0
        # options are 'fixed' 'normal-to-surface' 'sphere-gaussian' 'sphere-uniform' 'angle-uniform'
        self.interaction_orientation_sampling_mode: str = 'angle-uniform'

        self.pose_cost_threshold: float = 1.0                  # interaction pose is reached when pose_cost < threshold
        self.pose_cost_min_percentage_drop: float = 10 / 100   # pose reaching movement stops when the pose cost stops improving
        self.pose_cost_drop_time: float = 2.0                  # seconds
        self.min_relative_task: float = float(np.deg2rad(20))  # min distance between init and target object configurations
        self.task_completion_tolerance: float = float(np.deg2rad(5))
        self.interaction_successful_tolerance: float = float(np.deg2rad(5))  # min movement towards target for an interaction to be
        # successful

        # sampled at the beginning of the job if dynamically_build_urdf is True
        self.object_scale: typing.Optional[float] = None
        self.object_center_xyz: typing.Optional[list] = None

        # generated at the beginning of the job
        self.simulation_params: typing.Optional[dict] = None
        self.controller_params: typing.Optional[dict] = None

        # initial and target poses (object and robot) are sampled at the beginning of the
        # simulation if they are still None
        self.initial_object_pose: typing.Optional[float] = None  # object articulation state at time zero
        self.target_object_pose: typing.Optional[float] = None   # target object configuration
        self.initial_robot_base_pose: typing.Optional[typing.List] = None

        self.simulation_result: typing.List[str] = ['started']
        self.final_object_pose: typing.Optional[float] = None        # object configuration at the end of the simulation
        self.movement_towards_target: typing.Optional[float] = None  # how much the object pose improved during the simulation
        self.total_time: typing.Optional[float] = None
        self.total_number_of_interactions: typing.Optional[int] = None
        self.number_successful_interactions: int = 0

        # ---- LEGACY OPTIONS ----
        self.network_orientation_encoding: str = 'quat'    # other option is 'r6'
        # full = get actionability -> orientation -> rank with affordance
        # affordance = choose an orientation at random, calculate affordance for all points
        self.sampling_network: str = 'full'
        # the utility_k params are relevant for thompson sampling
        self.actionability_utility_k: float = 5.  # somewhere between 1 and 10, where 10 is greedier
        self.orientation_utility_k: float = 5.  # somewhere between 1 and 10, where 10 is greedier
        self.robust_cost_percentile: float = 0.5  # for use if the network outputs a variance estimate, percentile of 0.5 = return the mean
        self.object_init_state_pdf: typing.List = [0.25, 0.25, 0.50]  # closed, open, in-between
        self.camera_noise: float = 0.0  # RMSE error (meters)

    def to_dict(self):
        return vars(self)

    def save(self, path):
        data = self.to_dict()
        dict_to_yaml(data, os.path.join(path, 'sim_params.yaml'))

    @staticmethod
    def load_from_path(path):
        data = yaml_to_dict(os.path.join(path, 'sim_params.yaml'))
        return SimParams.from_dict(data)

    @staticmethod
    def from_dict(dictionary):
        res = SimParams()
        for key, value in dictionary.items():
            setattr(res, key, value)
        return res


class VisualData:
    """
    Visual data generated during a simulation job.
    Can be saved and loaded.
    """
    def __init__(self):
        # pictures (at time zero)
        # xyz coordinates are in sapien camera frame
        self.xyza: typing.Optional[np.array] = None
        self.rgb: typing.Optional[np.array] = None
        # normal vectors are expressed in sapien camera frame
        self.normal_map: typing.Optional[np.array] = None
        self.movable_links_mask: typing.Optional[np.array] = None

        self.sampled_pixel_vertical_coordinate: typing.Optional[int] = None
        self.sampled_pixel_horizontal_coordinate: typing.Optional[int] = None

    def save(self, path):
        np.savez_compressed(os.path.join(path, "xyza.npz"), a=self.xyza)
        np.savez_compressed(os.path.join(path, "rgb.npz"), a=self.rgb)
        np.savez_compressed(os.path.join(path, "normal_map.npz"), a=self.normal_map)
        np.savez_compressed(os.path.join(path, "movable_links_mask.npz"), a=self.movable_links_mask)
        Camera.save_png(self.rgb, os.path.join(path, "rgb.png"))
        d = dict(sampled_pixel_vertical_coordinate=self.sampled_pixel_vertical_coordinate,
                 sampled_pixel_horizontal_coordinate=self.sampled_pixel_horizontal_coordinate)
        dict_to_yaml(d, os.path.join(path, 'visual_data.yaml'))

    @staticmethod
    def load_npz(fname):
        with np.load(fname) as data:
            res = data['a']
        return res

    @staticmethod
    def load_from_path(path, load_options=None):
        res = VisualData()
        if load_options is None:
            load_options = ['xyza', 'rgb', 'normal_map', 'movable_links_mask']
        if 'xyza' in load_options:
            res.xyza = VisualData.load_npz(os.path.join(path, "xyza.npz"))
        if 'rgb' in load_options:
            res.rgb = VisualData.load_npz(os.path.join(path, "rgb.npz"))
        if 'normal_map' in load_options:
            res.normal_map = VisualData.load_npz(os.path.join(path, "normal_map.npz"))
        if 'movable_links_mask' in load_options:
            res.movable_links_mask = VisualData.load_npz(os.path.join(path, "movable_links_mask.npz"))
        d = yaml_to_dict(os.path.join(path, 'visual_data.yaml'))
        res.sampled_pixel_horizontal_coordinate = d["sampled_pixel_horizontal_coordinate"]
        res.sampled_pixel_vertical_coordinate = d["sampled_pixel_vertical_coordinate"]
        return res


class TransformData:
    """
    Transform data generated during a simulation job.
    Can be saved and loaded.
    """
    def __init__(self):
        # data
        self.ref2handle: typing.Optional[np.array] = None
        self.handle2world: typing.Optional[np.array] = None
        self.cam2cam_mount: typing.Optional[np.array] = None
        self.cam_mount2world: typing.Optional[np.array] = None

    def save(self, path):
        np.savetxt(os.path.join(path, 'ref2handle.txt'), self.ref2handle)
        np.savetxt(os.path.join(path, 'handle2world.txt'), self.handle2world)
        np.savetxt(os.path.join(path, 'cam2cam_mount.txt'), self.cam2cam_mount)
        np.savetxt(os.path.join(path, 'cam_mount2world.txt'), self.cam_mount2world)

    @staticmethod
    def load_from_path(path):
        res = TransformData()
        res.ref2handle = np.loadtxt(os.path.join(path, 'ref2handle.txt'))
        res.handle2world = np.loadtxt(os.path.join(path, 'handle2world.txt'))
        res.cam2cam_mount = np.loadtxt(os.path.join(path, 'cam2cam_mount.txt'))
        res.cam_mount2world = np.loadtxt(os.path.join(path, 'cam_mount2world.txt'))
        return res


class TrajectoryData:
    """
    Trajectory data generated during a simulation job.
    Can be saved and loaded to 'replay' a simulation (see simulation_job_replay.py script)
    """
    def __init__(self):
        self.time: typing.List[float] = list()
        self.inputs: typing.List[np.array] = list()
        self.observations: typing.List[np.array] = list()
        self.costs: typing.List[dict] = list()

    def save(self, path):
        pandas.DataFrame(self.costs).to_csv(os.path.join(path, 'cost_map.csv'))
        np.save(os.path.join(path, 'inputs.npy'), np.array(self.inputs))
        np.save(os.path.join(path, 'time.npy'), np.array(self.time))
        np.save(os.path.join(path, 'observations.npy'), np.array(self.observations))

    @staticmethod
    def load_from_path(path):
        res = TrajectoryData()
        res.costs = pandas.read_csv(os.path.join(path, 'cost_map.csv'))
        res.inputs = np.load(os.path.join(path, 'inputs.npy'))
        res.time = np.load(os.path.join(path, 'time.npy'))
        res.observations = np.load(os.path.join(path, 'observations.npy'))
        return res


def get_cost(sim: SimParams, trj: TrajectoryData, negative_unreachable=False) -> typing.Tuple[float, bool]:
    """
    :param sim: simulation parameters
    :param trj: simulation time trajectory data
    :param negative_unreachable: assign a negative cost to unreachable points (for visualization purposes)
    :return: (cost, success)
    """
    relative_task = sim.target_object_pose - sim.initial_object_pose
    control_rate = 1.0 / sim.controller_params["dynamics"]["dt"]
    n_steps = sim.interaction_time * control_rate
    normalizer_cost = n_steps * (relative_task ** 2) * sim.controller_params["cost"]["object_weight"]

    # when loaded from path, trj.costs is a pandas dataframe
    cost_dataframe: pandas.DataFrame = trj.costs

    if 'object_cost' in cost_dataframe:
        cost_vector = cost_dataframe['object_cost'].values
        cost = float(np.nansum(cost_vector))
    else:
        cost = 1.0
    success = False
    if 'could NOT reach reference' in sim.simulation_result:
        if negative_unreachable:
            # for visualization purposes
            cost = -1.0
        else:
            cost = 1.0
    elif 'interaction NOT successful' in sim.simulation_result:
        cost /= normalizer_cost
        # the cost of non-successful interactions should never be less than 1.0
        cost = max(cost, 1.0)
    elif 'interaction successful' in sim.simulation_result:
        cost /= normalizer_cost
        success = True

    return cost, success
