import numpy as np


class Laplace:
    """
    Tools for Laplace distribution
    """

    @staticmethod
    def get_percentiles(median: np.array, bandwidth: np.array, percentile: float) -> np.array:
        """
        For each median[i] and bandwidth[i], return the percentile
        """
        robust = np.copy(median)
        if percentile < 0.5:
            robust = robust + bandwidth * np.log(2 * percentile)
        else:
            robust = robust - bandwidth * np.log(2 - 2 * percentile)
        return robust

    @staticmethod
    def cdf(median: np.array, bandwidth: np.array, x: float = 1.0) -> np.array:
        """
        For each median[i] and bandwidth[i], return the cdf(x)
        """
        aff_s = median < x
        c = 1 - 0.5 * np.exp(np.divide(median - x, bandwidth))

        aff_ns = median >= x
        # not success = predicted cost m is more than query x=1 (m > x)
        c_ns = 0.5 * np.exp(np.divide(x - median, bandwidth))

        res = np.zeros_like(bandwidth)
        res[aff_s] = c[aff_s]
        res[aff_ns] = c_ns[aff_ns]

        return res


def divide_trials(task_success_map):
    probabilities = list(task_success_map.values())
    tasks = list(task_success_map.keys())
    n_tasks = len(tasks)

    A = np.diag([p for p in probabilities])
    b = np.array([100 for i in range(n_tasks)])
    x = np.linalg.solve(a=A, b=b)

    task_sample_probabilities = (x / np.sum(x)).tolist()
    task_sample_map = dict()
    for i in range(n_tasks):
        task_sample_map[tasks[i]] = task_sample_probabilities[i]

    return task_sample_map


def sample_interaction_index(utility: np.ndarray, sampling_strategy: str):
    # use the utility to sample an interaction point
    if sampling_strategy == 'eps-greedy':
        ip_idx = np.argmax(utility)
    elif sampling_strategy == 'eps-95':
        n_pixels = len(utility)
        # ordered_indexes[0] is the smallest (i.e. least utility)
        ordered_indexes = np.argsort(utility).flatten()
        # get the top 5% of pixels (according to utility)
        indexes_to_sample = ordered_indexes[int(n_pixels * 0.95):]
        # uniformly sample interaction point from the top 5% of filtered utility
        ip_idx = np.random.choice(indexes_to_sample)
    elif sampling_strategy == 'thompson':
        normalizer = np.nansum(utility)
        if normalizer == 0:
            ip_idx = np.random.choice(a=len(utility))
        else:
            ip_idx = np.random.choice(a=len(utility), p=utility / normalizer)
    else:
        raise ValueError('sampling mode not recognized')

    return ip_idx
