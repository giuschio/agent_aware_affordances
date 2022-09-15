import matplotlib
import numpy as np
import re
import typing

from pathlib import Path
from random import random
from matplotlib import pyplot as plt
from time import time as get_time


class StreamingMovingAverage:
    def __init__(self, window_size):
        self.__window_size = window_size
        self.__values = []
        self.__sum = 0

        self.__smoothed_values = []

    def add(self, value):
        self.__values.append(value)
        self.__sum += value
        if len(self.__values) > self.__window_size:
            self.__sum -= self.__values.pop(0)

        self.__smoothed_values.append(self.average)

    @property
    def average(self):
        return float(self.__sum) / len(self.__values)

    def to_list(self):
        return self.__smoothed_values


def discrete_colormat(array, levels, colors):
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend='max')
    array_n = norm(array)
    colors = cmap(array_n)
    colors = np.reshape(colors, (len(array_n), 4))
    colors = colors[:, :3]
    return colors


def histogram_equalization(array):
    # based on code here: https://levelup.gitconnected.com/introduction-to-histogram-equalization-for-digital-image-enhancement-420696db9e43
    # STEP 1: Normalized cumulative histogram
    # cast to int
    array_int = np.floor(array * 256).astype('int')
    # flatten image array and calculate histogram via binning
    histogram_array = np.bincount(array_int, minlength=256)
    # normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels
    # normalized cumulative histogram
    chistogram_array = np.cumsum(histogram_array)
    # STEP 2: Pixel mapping lookup table
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
    # STEP 3: Transformation
    # flatten image array into 1D list
    img_list = list(array_int)
    # transform pixel values to equalize
    eq_img_list = [transform_map[p] for p in img_list]
    # reshape and write back into img_array
    array_int = np.array(eq_img_list)
    # cast to float
    array = array_int.astype('float') / 256
    return array


def exponential_mapping(array, c=0.5):
    return array ** c


def heatmap(array, mask=None, cmap_name='plasma_r', equalize=2):
    # another cmap_name could be viridis_r
    array = array.flatten()
    if mask is None:
        mask = np.linspace(0, len(array))
    res = np.ones_like(array)
    array = array[mask]
    # linear normalization
    if equalize < 0:
        array = np.clip(array, 0., 1.)
    if equalize >= 0:
        array = array / array.max()
    if equalize >= 1:
        array = (array - array.min()) / (array.max() - array.min())
    if equalize >= 2:
        array = histogram_equalization(array)
        array = exponential_mapping(array)

    res[mask] = array
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = cmap(res)
    colors = np.reshape(colors, (len(res), 4))
    colors = colors[:, :3]
    return colors


def sample_from_bounds(var: typing.Union[typing.List[float], float, None],
                       bounds: typing.List) -> typing.Union[typing.List[float], float]:
    """
    Sample var only if it has not been assigned yet. This allows the user to hardcode var if
    necessary during testing
    :param var: variable to assign
    :param bounds: bounds [min_0, max_0, min_1, max_1, ..., min_n, max_n]
    :return: var or length n
    """
    if var is None:
        var = list()
        length = len(bounds)
        if length % 2 == 1 or length < 2: raise ValueError("Length of <bounds> should be even (i.e. min and max)")
        for i in range(0, len(bounds), 2):
            lo, hi = bounds[i], bounds[i + 1]
            if lo > hi: raise ValueError("Min higher than max")
            var.append(lo + random() * (hi - lo))

        if len(var) == 1:
            var = var[0]
    return var


def sample_object_state(var: typing.Optional[float],
                        bounds: typing.List,
                        pdf: typing.List[float]) -> float:
    """
    Sample var only if it has not been assigned yet. This allows the user to hardcode var if
    necessary during testing
    :param bounds: bounds for random sampling
    :param var: variable to assign
    :param pdf: probability distribution pdf = [p_closed, p_open, p_in_between]
    :return: var
    """
    if var is None:
        random_state = sample_from_bounds(var=var, bounds=bounds)
        elements = [bounds[0], bounds[1], random_state]
        var = float(np.random.choice(elements, 1, p=pdf))
    return var


class Logger2:
    def __init__(self, log_title="", fname=None, log_to_screen=False):
        self.__log_to_screen = log_to_screen
        if fname is None:
            fname = log_title + "_log.txt"
        self.__fname = fname
        self.__write("Log for: " + log_title)
        self.__start_time = get_time()

    def append(self, msg: str, level='info'):
        self.__write("\t" + level.upper() + " - {:0.2f}".format(get_time()-self.__start_time) + ": " + msg)

    def __write(self, msg):
        if self.__log_to_screen:
            print(msg)
        with open(self.__fname, "a") as myfile:
            myfile.write(msg + "\n")


def dir_exists(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def plot_losses(train_losses: np.ndarray, val_losses: np.ndarray):
    epochs = np.arange(start=0, stop=len(train_losses), step=1)
    titles = ['affordance loss', 'orientation loss', 'actionability loss', 'optimized loss', 'total loss']
    for i in range(5):
        plt.plot(epochs, train_losses[:, i], 'b', label='train_loss')
        plt.plot(epochs, val_losses[:, i], 'r--', label='val_loss')
        plt.xlabel("epoch number")
        plt.title(titles[i])
        plt.legend()
        plt.show()


def human_sort(m_list: typing.List[str]):
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        """
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        """
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    m_list.sort(key=natural_keys)
    return m_list
