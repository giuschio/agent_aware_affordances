import numpy as np


def zero_center_distance(points: np.ndarray) -> np.ndarray:
    """
    :param points: pointcloud
    :return:
    """
    # calculate average distance (x coordinate)
    dist = np.average(points[:, 0])
    points[:, 0] /= dist
    return points


def load_if_available(checkpoint, target, key):
    if key in checkpoint:
        target.load_state_dict(checkpoint[key])


class EarlyStopping:
    """
    Early stopping utility used during network training
    """
    def __init__(self, patience=5, min_improvement=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_improvement: minimum improvement (percentage) (i.e. between 0 and 1)
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.counter = 0
        self.best_loss = None
        self.best_epoch = None

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.best_epoch = None

    def __call__(self, current_val_loss, current_epoch_number):
        """
        :param current_val_loss:
        :return: (bool): whether or not to stop training
        """
        if self.best_loss is None:
            self.best_loss = current_val_loss
            self.best_epoch = current_epoch_number
            msg = "initializing early stopping with current loss {:0.3f}".format(self.best_loss)
        elif current_val_loss / self.best_loss < (1 - self.min_improvement):
            msg = "loss has improved from {:0.3f} to {:0.3f}".format(self.best_loss, current_val_loss)
            self.best_loss = current_val_loss
            self.best_epoch = current_epoch_number
            # reset counter if validation loss improves
            self.counter = 0
        else:
            self.counter += 1
            msg = "loss has not improved for {:5d} epochs".format(self.counter)

        return self.counter >= self.patience, msg
