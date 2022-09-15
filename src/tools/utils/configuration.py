import os


class Configuration:
    """
    The absolute path of the src folder is stored in an environment variable
    You need to run setup.sh before you run any code
    """
    src_path = os.getenv('AFFORDANCE_SOURCE', None)

    src2mppi = "tools_mppi"
    # this allows switching the robot model globally across the entire project
    src2robot_folder = "robot_models"
    robot_type = "full"  # options are "full" or "hand"
    if robot_type == "full":
        src2robot = "robot_models/rd.urdf"
        src2robot_controller_raisim = "robot_models/rd_raisim_cylinder.urdf"
        src2robot_simulation_raisim = "robot_models/rd_raisim_cylinder_accurate.urdf"
    elif robot_type == "hand":
        src2robot = "robot_models/hd.urdf"
        src2robot_controller_raisim = "robot_models/hd_raisim_cylinder.urdf"
        src2robot_simulation_raisim = "robot_models/hd_raisim_cylinder_accurate.urdf"
    else:
        raise ValueError("Unknown robot_type: ", robot_type)

    pointcloud_dimension: int = 10000
    random_seed: int = 1000  # for reproducibility

    @staticmethod
    def get_abs(src2path):
        """
        Get absolute path from source-relative path
        :param src2path: path from source to file
        :return:
        """
        return os.path.join(Configuration.src_path, src2path)
