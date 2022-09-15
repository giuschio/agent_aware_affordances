import numpy as np
import random
import torch
import typing

from datetime import datetime
from os.path import join as pj
from time import time as get_time

from torch.optim.lr_scheduler import ReduceLROnPlateau

from tools_pytorch.network_full import Pipeline
from tools_pytorch.utils import EarlyStopping
from tools.utils.misc import Logger2 as Logger, dir_exists
from tools.utils.configuration import Configuration
from tools.data_tools.data_loader import DataLoader, check_batch_size


class Trainer:
    def __init__(self, train_folder, val_folder, output_folder=None):
        self.train_folder = train_folder
        self.val_folder = val_folder
        if output_folder is None: output_folder = "training_logs_" + datetime.today().strftime('%Y%m%d_%H%M%S')
        self.output_folder = output_folder

        self.batch_size = 10
        self.min_epochs_per_mode = 5
        self.starting_mode = None
        self.orientation_encoding = 'quat'       # or 'r6
        self.cost_representation = 'continuous'  # or 'success'

    def training_job(self,
                     n_epochs: int,
                     previous_checkpoint_path: typing.Optional[str] = None,
                     extra_seed: int = 0):
        # --- REPRODUCIBILITY OPTIONS
        seed = Configuration.random_seed + extra_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False

        # --- START TRAINING
        dir_exists(self.output_folder)
        log = Logger(log_title="network training",
                     fname=pj(self.output_folder, "training_log.txt"),
                     log_to_screen=True)
        validation_loader = DataLoader(self.val_folder)
        training_loader = DataLoader(self.train_folder)

        log.append("train folder " + str(self.train_folder))
        log.append("val folder " + str(self.val_folder))
        log.append("previous checkpoint: " + str(previous_checkpoint_path))
        log.append("orientation encoding: " + str(self.orientation_encoding))
        log.append("cost representation: " + str(self.cost_representation))
        log.append("current random seed: " + str(seed))

        validation_loader.rotation_representation = self.orientation_encoding
        training_loader.rotation_representation = self.orientation_encoding
        validation_loader.cost_representation = self.cost_representation
        training_loader.cost_representation = self.cost_representation
        network = Pipeline(orientation_encoding=self.orientation_encoding,
                           cost_representation=self.cost_representation)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        starting_epoch = 0
        mode = 1
        if previous_checkpoint_path is not None:
            checkpoint = torch.load(previous_checkpoint_path)
            network.load_state_dict(checkpoint['network_state'])
            optimizer.load_state_dict(checkpoint['adam_state'])
            starting_epoch = checkpoint['epoch'] + 1
            mode = checkpoint['mode']
        if self.starting_mode is not None:
            # allow overwriting of starting mode
            mode = self.starting_mode

        # default batch size
        batch_size_t = check_batch_size(training_loader, self.batch_size)
        batch_size_v = check_batch_size(validation_loader, self.batch_size)

        train_losses = list()
        val_losses = list()
        # Early stopping: improve by at least 5% in 12 epochs
        early_stop = EarlyStopping(min_improvement=0.05,
                                   patience=12)
        # Scheduler: improve by at least 2% in 5 epochs, otherwise divide lr by 5
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      mode='min',
                                      threshold_mode='rel',
                                      threshold=0.02,
                                      factor=0.2,
                                      min_lr=1e-6,
                                      patience=5)
        start_time = get_time()
        checkpoint_paths = dict()
        for epoch in range(starting_epoch, n_epochs + starting_epoch):
            training_loader.reset()
            validation_loader.reset(shuffle=False)
            log.append("epoch number " + str(epoch) + ", mode: " + str(mode))
            tot_secs_elapsed = round(get_time() - start_time)
            mins, secs = tot_secs_elapsed // 60, tot_secs_elapsed % 60
            log.append("time elapsed: " + str(mins) + " minutes, " + str(secs) + " seconds")
            # --- TRAINING -----
            network.train()
            epoch_losses_train = list()
            while training_loader.epoch_done is False:
                # catch edge case of invalid loader with zero data points
                if batch_size_t < 1: break
                loss_affordance, loss_action, loss_actionability = network(training_loader, batch_size_t)
                loss = torch.clone(loss_affordance)
                if mode > 1: loss += loss_action
                if mode > 2: loss += loss_actionability
                if loss > 0.001:
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                epoch_losses_train.append([float(loss_affordance), float(loss_action), float(loss_actionability),
                                           float(loss), float(loss_affordance) + float(loss_action) + float(loss_actionability)])

            if len(epoch_losses_train) == 0: epoch_losses_train.append([0., 0., 0., 0.])
            epoch_losses_train = np.array(epoch_losses_train)
            train_losses.append(np.mean(epoch_losses_train, axis=0))
            log.append("\ttraining points " + str(len(epoch_losses_train) * batch_size_t))
            log.append("\tmean training losses: " + str(np.round(train_losses[-1], decimals=3)))

            # # --- VALIDATION ---------
            epoch_losses_val = list()
            with torch.no_grad():
                network.eval()
                while validation_loader.epoch_done is False:
                    # catch edge case of invalid loader with zero data points
                    if batch_size_v < 1: break
                    # network forward pass
                    loss_affordance, loss_action, loss_actionability = network(validation_loader, batch_size_v)
                    loss = torch.clone(loss_affordance)
                    if mode > 1: loss += loss_action
                    if mode > 2: loss += loss_actionability

                    epoch_losses_val.append([float(loss_affordance), float(loss_action), float(loss_actionability),
                                             float(loss), float(loss_affordance) + float(loss_action) + float(loss_actionability)])

            if len(epoch_losses_val) == 0: epoch_losses_val.append([0., 0., 0., 0.])
            epoch_losses_val = np.array(epoch_losses_val)
            val_losses.append(np.mean(epoch_losses_val, axis=0))
            log.append("\tvalidation points " + str(len(epoch_losses_val) * batch_size_v))
            log.append("\tmean validation losses: " + str(np.round(val_losses[-1], decimals=3)))

            current_val_loss = float(val_losses[-1][3])

            # # save models
            checkpoint = dict(network_state=network.state_dict(),
                              adam_state=optimizer.state_dict(),
                              epoch=epoch, mode=mode)
            # save the epoch checkpoint, in case something crashes and we need to re-start
            checkpoint_paths[epoch] = pj(self.output_folder, "training_result_" + str(epoch) + ".pt")
            torch.save(checkpoint, checkpoint_paths[epoch])
            np.save(file=pj(self.output_folder, "training_losses.npy"), arr=np.array(train_losses))
            np.save(file=pj(self.output_folder, "validation_losses.npy"), arr=np.array(val_losses))
            log.append("\tsaved epoch checkpoint")

            # LR scheduler
            scheduler.step(current_val_loss)
            if scheduler.state_dict()['num_bad_epochs'] == 0 and scheduler.state_dict()['best'] != current_val_loss:
                log.append("\t----REDUCED LEARNING RATE-----")
            # Early Stopping
            stop_loop, msg = early_stop(current_val_loss, current_epoch_number=epoch)
            log.append("\tES: " + msg)
            if stop_loop:
                log.append("\tearly stopping... ")
                break

            # only one epoch in mode 2
            if mode == 2:
                mode = 3
                early_stop.reset()

        training_loader.close()
        validation_loader.close()
        log.append("--------- TRAINING RUN END ----------")
        log.append("--------- BEST EPOCH IS EPOCH {:5d} ----------".format(early_stop.best_epoch))

        return checkpoint_paths[early_stop.best_epoch]


def train_model(dataset_path, cost_encoding='success', extra_seed=0):
    """
    Train a network pipeline
    :param dataset_path: path to the root dataset folder, containing training (train_p) and validation (val_p) datasets
    :param cost_encoding: 'success' or 'continuous' -> default is 'success'
    :param extra_seed: random seed (0 -> use default random seed, for reproducibility, > 0 -> add to the random seed)
    :return:
    """
    dataset_path = Configuration.get_abs(dataset_path)
    # train the affordance module
    trainer = Trainer(train_folder=pj(dataset_path, 'train_p'), val_folder=pj(dataset_path, 'val_p'))
    trainer.starting_mode = 1
    trainer.orientation_encoding = 'quat'
    trainer.cost_representation = cost_encoding
    checkpoint_path = trainer.training_job(n_epochs=60, previous_checkpoint_path=None, extra_seed=extra_seed)

    # train everything jointly
    trainer = Trainer(train_folder=pj(dataset_path, 'train_p'), val_folder=pj(dataset_path, 'val_p'))
    trainer.starting_mode = 2
    trainer.orientation_encoding = 'quat'
    trainer.cost_representation = cost_encoding
    trainer.training_job(n_epochs=60, previous_checkpoint_path=checkpoint_path, extra_seed=extra_seed)
