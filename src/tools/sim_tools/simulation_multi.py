"""
This file contains all functions that deal with multi-processing of simulation jobs
Made using the documentation found at: https://docs.python.org/3/library/multiprocessing.html
"""
import copy
import os

import torch.multiprocessing as multiprocessing
import typing

from time import sleep, time as get_time
from tqdm import tqdm

from executables.core.simulation_job_core import simulation_job
from tools.utils.configuration import Configuration


def run_process(simulations_queue: multiprocessing.Queue,
                shared_lock: multiprocessing.Lock,
                process_number: int,
                network_checkpoint: typing.Optional[dict] = None,
                results_queue: typing.Optional[multiprocessing.Queue] = None):
    while not simulations_queue.empty():
        job = simulations_queue.get()
        job.process_number = process_number
        try:
            simulation_job(sim_params=job,
                           network_checkpoint=network_checkpoint,
                           lock=shared_lock)
        except Exception as e:
            print(e)
        if results_queue is not None:
            results_queue.put((job.process_number, job.job_number))


def get_processes(n_processes,
                  simulations_queue,
                  shared_lock,
                  network_checkpoint=None,
                  results_queue=None):
    process_list = list()
    for i in range(n_processes):
        # each process has its own instance of the checkpoint dictionary
        network_checkpoint_copy = copy.deepcopy(network_checkpoint)
        kwargs = dict(simulations_queue=simulations_queue,
                      shared_lock=shared_lock,
                      process_number=i,
                      network_checkpoint=network_checkpoint_copy,
                      results_queue=results_queue)
        process_list.append(multiprocessing.Process(target=run_process, kwargs=kwargs, daemon=True))
        del network_checkpoint_copy
    return process_list


def collect_data_util(simulations_queue: multiprocessing.Queue,
                      n_processes: int,
                      pbar: tqdm,
                      time_per_job: float = 6.0,
                      network_checkpoint: typing.Optional[str] = None):
    # as in https://medium.com/swlh/protect-your-shared-resource-using-multiprocessing-locks-in-python-21fc90ad5af1
    shared_lock = multiprocessing.Lock()
    results_queue = multiprocessing.Queue()
    process_list = get_processes(n_processes=n_processes,
                                 simulations_queue=simulations_queue,
                                 shared_lock=shared_lock,
                                 network_checkpoint=network_checkpoint,
                                 results_queue=results_queue)
    del network_checkpoint
    n_jobs = simulations_queue.qsize()
    timeout = time_per_job * n_jobs
    for process in process_list:
        process.start()

    start_time = get_time()
    n_todos = n_jobs
    while True:
        # update the progress bar
        n_todos_new = simulations_queue.qsize()
        pbar.update(n_todos - n_todos_new)
        n_todos = n_todos_new

        if get_time() - start_time > timeout:
            # kill everything is timeout was reached
            print("\nsomething got stuck, killing this chunk\n")
            for process in process_list:
                process.kill()
            break
        if simulations_queue.empty():
            # all jobs have been taken
            break
        sleep(2.0)

    f_time = get_time()
    while results_queue.qsize() < n_jobs:
        sleep(2.0)
        if get_time() - f_time > time_per_job * 6:
            print('global timeout reached')
            # global timeout
            break
    # all jobs have returned OR something hanged for more than 60 seconds
    sleep(2.0)

    for idx, process in enumerate(process_list):
        process.join(timeout=1)
        # process.terminate()
        # sleep(1.0)
        # process.close()

    return results_queue


def collect_data(simulations_list: typing.List,
                 n_processes: int,
                 chunk_size: int = 500,
                 time_per_job: float = 6.0,
                 network_checkpoint: typing.Optional[str] = None):
    # divide the simulations list into several queues
    n_jobs = len(simulations_list)
    n_queues = max(int(n_jobs / chunk_size), 1)
    sim_queues = [multiprocessing.Queue() for k in range(n_queues)]
    for job_n, job_params in enumerate(simulations_list):
        idx = min(job_n // chunk_size, n_queues - 1)
        sim_queues[idx].put(job_params)

    pbar = tqdm(total=n_jobs, smoothing=0.0, ncols=100)
    for queue_n, queue in enumerate(sim_queues):
        collect_data_util(simulations_queue=queue,
                          n_processes=n_processes,
                          pbar=pbar,
                          time_per_job=time_per_job,
                          network_checkpoint=network_checkpoint)
    pbar.close()


def get_objects(folder_path) -> typing.List[str]:
    abs_assets_path = Configuration.get_abs(folder_path)
    object_names = os.listdir(abs_assets_path)
    for i in range(len(object_names)):
        object_names[i] = os.path.join(folder_path, object_names[i])
    return object_names
