"""
This script replays the rendering of a given simulation
See main for options
"""
import random

import sapien.core as sapien
import os

from tools.sim_tools.sim_data import TrajectoryData, TransformData, SimParams
from tools.sim_tools.sim_engine import SimulationEngine
from tools.sim_tools.sim_functions import set_params, debug_set_marker
from tools.utils.data import get_job_folders

from tools_mppi.mppi import Observation
from tools.utils.configuration import Configuration


def simulation_replay(data_path, rate=1.0, keep_open=True):
    sim_params = SimParams.load_from_path(data_path)
    sim_params.show_gui = True
    sim_params.robot_model = Configuration.src2robot_simulation_raisim
    sim_params = set_params(sim_params, replay=True)
    sim_params.render_resolution = ((1920, 1080),)
    trj = TrajectoryData.load_from_path(data_path)
    transforms = TransformData.load_from_path(data_path)

    engine = SimulationEngine(params=sim_params, data=None)
    engine.load_controller()

    # for convenience
    viewer = engine.viewer
    scene = engine.scene

    # render scene and call camera.take_picture so that I can get visual data
    ref2world = transforms.handle2world @ transforms.ref2handle
    # visualize a marker on the chosen interaction pose
    # ref_box = debug_set_marker(engine.scene,
    #                            pose=sapien.Pose.from_transformation_matrix(ref2world),
    #                            color=[0., 0., 1.], name='ee_ref')
    sim_length = len(trj.observations)
    for i in range(0, sim_length, int(rate)):
        if viewer.closed: break

        obs = trj.observations[i]
        engine.object.set_joint_position(Observation.object_qpos(obs))
        engine.robot.set_qpos(Observation.robot_qpos(obs))

        scene.update_render()
        viewer.render()

    if keep_open:
        print("END OF REPLAY: close the viewer to exit the script")
        while not viewer.closed:
            scene.update_render()
            viewer.render()
    viewer.close()


if __name__ == "__main__":
    # ---- EXAMPLE USAGE ----
    # replay one simulation job
    job_path = 'demo_jobs/job_open'  # path relative to the src folder
    simulation_replay(Configuration.get_abs(job_path), rate=10, keep_open=True)

    # replay all jobs in a folder
    proot = 'demo_jobs'  # path relative to the src folder
    proot = Configuration.get_abs(proot)
    points = get_job_folders(dirpath=proot)
    random.shuffle(points)
    for idx, job in enumerate(points):
        simulation_replay(data_path=os.path.join(proot, job), rate=10, keep_open=False)
