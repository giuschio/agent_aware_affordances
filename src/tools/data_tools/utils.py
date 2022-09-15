import os
import copy
import multiprocessing
import pandas

from tools.sim_tools.sim_data import SimParams
from tools.utils.configuration import Configuration
from tools.utils.data import get_job_folders


def rad_to_deg(rad):
    return rad / 1.57 * 90


def binomial_to_normal(n, p):
    m, s = n * p, (n * p * (1 - p)) ** 0.5
    return m, s


def confidence_interval(n, p, factor=2.):
    n = float(n)
    m, s = binomial_to_normal(n=n, p=p)
    confidence = (factor * s) / n
    return confidence


def load_jobs(jobs_folder):
    points = get_job_folders(dirpath=jobs_folder)
    points_f = [os.path.join(jobs_folder, point) for point in points]
    res = list()
    with multiprocessing.Pool(processes=8) as pool:
        for i, sim in enumerate(pool.imap_unordered(SimParams.load_from_path, points_f)):
            res.append(sim)
    return res


def print_jobs_summary(jobs_folder, condition=lambda sim: True):
    evals_base = {'total': 0, 'ik fail': 0, 'reach': 0, 'success': 0, 'success to target': 0}
    evals = {'OVERALL': copy.deepcopy(evals_base)}
    loaded_sims = load_jobs(jobs_folder)
    for sim in loaded_sims:
        if condition(sim) is not True:
            continue

        object_name = sim.object_name
        if object_name not in evals:
            evals[object_name] = copy.deepcopy(evals_base)

        for name in [object_name, 'OVERALL']:
            evals[name]['ik fail'] += int('pinocchio reachable' not in sim.simulation_result)
            evals[name]['reach'] += int('could NOT reach reference' not in sim.simulation_result)
            evals[name]['success'] += int('interaction successful' in sim.simulation_result)
            evals[name]['success to target'] += int(abs(sim.final_object_pose - sim.target_object_pose) < sim.task_completion_tolerance)
            evals[name]['total'] += 1

    res_frame = {}
    for object_name, object_eval in evals.items():
        tmp = {}
        if object_name != 'OVERALL': continue
        if float(object_eval['total']) < 1.0: continue
        for key, value in object_eval.items():
            p = (float(value) / float(object_eval['total']))
            confidence = confidence_interval(n=object_eval['total'], p=p)
            res = "{:0.1f}% \u00B1 {:0.1f}%".format(p * 100., confidence * 100.)

            if key == 'total':
                res = str(int(value))
            tmp[key] = res
        res_frame[object_name] = copy.deepcopy(tmp)

    df = pandas.DataFrame(res_frame).T
    df.fillna("-_-", inplace=True)
    print(df)


def print_detailed_summary(jobs_folder, interaction_type=None):
    jobs_folder = Configuration.get_abs(jobs_folder)
    print("printing summary for jobs in folder: ")
    print("\t" + jobs_folder)
    if interaction_type is None: interaction_type = ['open', 'close']

    def open_from_zero(sim: SimParams):
        return sim.initial_object_pose == 0.0

    def close_from_full(sim: SimParams):
        return sim.initial_object_pose > 1.56

    def open_normal(sim: SimParams):
        return sim.initial_object_pose != 0.0 and sim.target_object_pose > sim.initial_object_pose

    def close_normal(sim: SimParams):
        return sim.initial_object_pose < 1.57 and sim.target_object_pose < sim.initial_object_pose

    if 'open' in interaction_type:
        print("OPEN FROM ZERO")
        print_jobs_summary(jobs_folder, open_from_zero)
        print("\nOPEN NORMAL")
        print_jobs_summary(jobs_folder, open_normal)
    if 'close' in interaction_type:
        print("\nCLOSE FROM FULL")
        print_jobs_summary(jobs_folder, close_from_full)
        print("\nCLOSE NORMAL")
        print_jobs_summary(jobs_folder, close_normal)
