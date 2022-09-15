## System Requirements
#### Hardware
This code was tested on a laptop equipped with an Intel Core i7-7700HQ quad-core processor (32 GB RAM) and an NVIDIA GTX 1050 Ti graphics card (4 GB RAM).

#### Software
Please also note that in order to run this code a RAISIM installation is required. The installation requires a license, which can be requested free of charge for academic purposes.

- Ubuntu 20.04   
- ROS Noetic   
- Python 3.8 (native on Ubuntu 20.04)   
- Vulkan tools (sudo apt install vulkan-tools on ubuntu)   
- Nvidia cuda toolkit - CUDA 10.1   
  ```bash
    sudo apt install nvidia-cuda-toolkit=10.1.243-3
  ```

## Install Instructions
Clone this repo:
  ```bash
    git clone git@github.com:giuschio/agent_aware_affordances.git
  ```
Run the setup_python.sh script will create a python virtual environment and install the necessary requirements. Please make sure your system satisfies the system requirements before running this script.
  ```bash
    bash setup_python.sh
  ```
Run the setup_project.sh script to set up the required folder structure for the project.
  ```bash
    bash setup_project.sh
  ```
#### RAISIM Installation and sampling-based controller
- Install and build the sampling_based_control package from [this link](https://github.com/ethz-asl/sampling_based_control/tree/gis_pybind_manipulation) (branch gis_pybind_manipulation, follow 
the README to install dependencies, including RAISIM). 
- Copy the generated .so file from ~/catkin_ws/devel/.private/mppi_manipulation/lib/pymppi_manipulation.cpython-38-x86_64-linux-gnu.so to src/tools_mppi.    

Note: You might need to add your catkin_ws/devel/lib path to your LD_LIBRARY_PATH.


## Usage
#### Objects and Robot Model
We include a demo model in the repo (in _demo_models_ directory). You can download further object models from the [PartNet website](https://sapien.ucsd.edu/browse). 
We provide a script (_executables/0_partnet_preprocessing) to convert PartNet objects to the conventions required in this repo.
We provide two robot models: a disembodied gripper and a full robot. The robot model can be selected in _tools/utils/configuration_ (WARNING: This choice will propagate to the entire repo).
#### Data Collection and Training
- **executables/1_collect_data_run**   
Run multiple simulation jobs to collect training data (given a set of objects). Simulation results are saved in "human readable" format (i.e. one folder for each simulation, with rbg images and plaintext vectors where possible).
- **executables/2_consolidate_dataset_run**    
Pre-process the collected training data (downsamples the pointcloud, calculates the interaction score...) and saves the resulting dataset in a consolidated format (i.e. numpy arrays and hdf5 dataframes). The infolder and outfolder can be changed in the if __name__ == "__main__" block.
- **executables/3_training_job_run**    
Given training and validation consolidated datasets, trains a network model. All epoch checkpoints and logs are saved to _training_logs_yyyymmdd_hhmmss_.

#### Evaluation
- **executables/simulation_job_replay**   
Replay a simulation job offline (demo jobs are provided in the _demo_jobs_ directory)
- **executables/simulation_job_run**    
Run a single simulation job with adjustable parameters (network, object, initial conditions, task...). See comments for parameter function.
- **executables/1_collect_data_run**    
This scripts runs multiple simulation jobs (given a set of objects and a task). It can either use random interaction points to collect training data (or establish a random baseline) or use a pretrained network to evaluate the performance. Parameters can be adjected (see comments).
- **executables/4_print_results_summary**    
Evaluate and print statistics (success rate, reach rate...) of a collection of jobs.
#### Notes
- file and folder paths should be specified relative to the src folder