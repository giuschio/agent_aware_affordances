"""
Visualize an urdf file in sapien, with the robot model for scale
"""
import numpy as np

import sapien.core as sapien
from sapien.utils.viewer import Viewer

from tools.utils.time_utils import Clock
from tools.utils.configuration import Configuration
from tools.sim_tools.actors import Object, Robot


def main(object_path):
    engine = sapien.Engine()
    engine.set_log_level('error')
    renderer = sapien.VulkanRenderer()
    engine.set_renderer(renderer)
    clock = Clock(sim_rate=2000)

    scene_config = sapien.SceneConfig()
    scene_config.solver_iterations = 40
    scene = engine.create_scene(scene_config)
    scene.set_timestep(clock.sim_timestep)
    # scene.add_ground(0)

    rscene = scene.get_renderer_scene()
    rscene.set_ambient_light([0.5, 0.5, 0.5])
    rscene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    # viewer.set_camera_xyz(x=-2, y=0, z=1)
    # viewer.set_camera_rpy(r=0, p=-0.3, y=0)
    viewer.set_camera_xyz(x=-1, y=-2.5, z=1.5)
    viewer.set_camera_rpy(r=0, p=0, y=-1.57)

    # Load URDF
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True

    material = engine.create_physical_material(0.5, 0.2, 0.5)
    object_path = Configuration.get_abs(object_path)
    m_object = Object(path=object_path, loader=loader, material=material, rendering_only=False)
    m_object.set_joint_position(0)

    robot_file = Configuration.get_abs(Configuration.src2robot_simulation_raisim)
    robot = Robot(robot_file, loader, base_pose=np.array([-2.5, 0, 0]))

    while not viewer.closed:
        robot.set_qvel(np.zeros(12))
        if clock.is_render_update():
            scene.update_render()
            viewer.render()
        scene.step()
        clock.step()


if __name__ == '__main__':
    object_path = 'demo_objects/demo_oven/mobility_original.urdf'
    main(object_path)
