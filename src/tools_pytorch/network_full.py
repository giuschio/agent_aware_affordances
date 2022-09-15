"""
Created by Giulio on Dec 20th 2021
"""
import numpy as np
import torch
import typing
import warnings

# https://github.com/erikwijmans/Pointnet2_PyTorch
from torch import nn as nn

from tools_pytorch.network_components import FeatureNet, AffordanceNet, ActionNet, ActionabilityNet
from tools.data_tools.data_loader import DataLoader
from tools_pytorch.losses import l1_loss as affordance_loss_L1
from tools_pytorch.losses import cross_entropy_loss as affordance_loss_cross_entropy
from tools_pytorch.losses import l1_loss as actionability_loss
from tools_pytorch.losses import rotation_loss, rotation_loss_w2a


class Pipeline(nn.Module):
    def __init__(self,
                 feat_dim=128,
                 n_orientation_proposals=100,
                 orientation_encoding='quat',
                 cost_representation='continuous'):
        super().__init__()
        self.dtype = torch.float32
        self.testing = False
        # device = torch.device("cpu")
        self.device = torch.device("cuda:0")
        self.feat_dim = feat_dim
        orientation_dim = 6 if orientation_encoding == 'r6' else 4
        self.orientation_dim = orientation_dim
        self.cost_representation = cost_representation

        self.feature_extractor = FeatureNet(feat_dim).to(device=self.device, dtype=self.dtype)
        self.sigmoid_output = bool(self.cost_representation == 'success')
        self.affordance_net = AffordanceNet(self.feat_dim,
                                            orientation_dimension=self.orientation_dim,
                                            sigmoid_output=self.sigmoid_output).to(device=self.device, dtype=self.dtype)
        self.actionability_net = ActionabilityNet(feat_dim).to(device=self.device, dtype=self.dtype)
        self.action_net = ActionNet(feat_dim, n_orientation_proposals, orientation_dimension=orientation_dim).to(device=self.device,
                                                                                                                 dtype=self.dtype)

    def to_torch(self, *args):
        res = [torch.from_numpy(var).to(device=self.device, dtype=self.dtype) for var in args]
        return res

    @staticmethod
    def check_nan(*args):
        res = [np.isnan(np.sum(var)) for var in args]
        return True in res

    def score_orientation_proposals(self, orientation_proposals, point_feats, interaction_xyz_gt, tasks_gt, best_of=5):
        # copy the affordance net
        affordance_net_copy = AffordanceNet(self.feat_dim,
                                            orientation_dimension=self.orientation_dim,
                                            sigmoid_output=self.sigmoid_output).to(device=self.device, dtype=self.dtype)
        affordance_net_copy.load_state_dict(self.affordance_net.state_dict())
        affordance_net_copy.eval()
        with torch.no_grad():
            orient_cost_pred, orient_cost_uncert = affordance_net_copy(pixel_feats=point_feats,
                                                                       xyzs=interaction_xyz_gt,
                                                                       orientations=orientation_proposals,
                                                                       tasks=tasks_gt)
            n_points = orient_cost_pred.shape[0]
            argsorted_cost_preds = torch.argsort(orient_cost_pred, dim=1).squeeze(2)
            # now we have orientation_proposals.shape = (B, N_ORIENT_PROPS, 4) and argsorted_cost_predictions.shape = (B, N_ORIENT_PROPS, 1)
            # where B is the number of interaction points (i.e. for each B interaction point, we have N_ORIENT_PROPS proposals of dimension 4
            # best_costs.shape = (B=n_points, best_of, 1)
            best_costs = torch.stack([orient_cost_pred[idx, argsorted_cost_preds[idx, :best_of]] for idx in range(n_points)])
            best_orientations = torch.stack([orientation_proposals[idx, argsorted_cost_preds[idx, :best_of], :] for idx in range(n_points)])
        return best_orientations, best_costs

    def forward(self,
                loader: DataLoader,
                batch_size: int):
        """
        Network forward pass
        :param loader: data loader instance
        :param batch_size:
        :return: loss_affordance, loss_action, loss_actionability
        """
        # ---- GET INPUT DATA
        # data from loader
        points_np, point_indexes_np, orientations_np, costs_np, tasks_np = loader.get_batch(dimension=batch_size)
        if self.check_nan(point_indexes_np, orientations_np, costs_np, tasks_np):
            warnings.warn("NaN value detected, skipping batch.")
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        points_gt, orientations_gt, tasks_gt = self.to_torch(points_np, orientations_np, tasks_np)
        tasks_gt = torch.reshape(tasks_gt, (batch_size, 1))

        # ---- FEATURE EXTRACTION
        all_feats = self.feature_extractor(points_gt)
        # select only the points we are interested in
        point_feats = torch.stack([all_feats[i, point_indexes_np[i], :] for i in range(batch_size)])
        interaction_xyz_gt = torch.stack([points_gt[i, point_indexes_np[i], :] for i in range(batch_size)])

        # ---- PREDICT ACTIONABILITY
        actionability_prediction, actionability_uncertainty = self.actionability_net(pixel_feats=point_feats,
                                                                                     xyzs=interaction_xyz_gt,
                                                                                     tasks=tasks_gt)
        # ---- ORIENTATION PROPOSALS
        orientation_proposals = self.action_net(pixel_feats=point_feats, xyzs=interaction_xyz_gt, tasks=tasks_gt)
        # ---- GROUND TRUTH COST PREDICTION
        cost_prediction, cost_uncertainty = self.affordance_net(pixel_feats=point_feats,
                                                                xyzs=interaction_xyz_gt,
                                                                orientations=torch.reshape(orientations_gt, (batch_size, 1, -1)),
                                                                tasks=tasks_gt)

        # ---- SCORING OF ORIENTATION PROPOSALS
        best_orientations, best_costs = self.score_orientation_proposals(orientation_proposals=orientation_proposals,
                                                                         point_feats=point_feats,
                                                                         interaction_xyz_gt=interaction_xyz_gt,
                                                                         tasks_gt=tasks_gt,
                                                                         best_of=5)

        # ---- CALCULATE LOSSES
        # cost and orientation gt data directly from dataset
        costs_gt = torch.from_numpy(costs_np).to(device=self.device, dtype=self.dtype)
        costs_gt = torch.reshape(costs_gt, (batch_size,))
        orientation_gt = orientations_gt.detach()

        # estimate the actionability ground truth as the minimum predicted cost out of all orientation proposals for each point
        # now for each B point, calculate actionability -> resulting vector is of shape (B, 1)
        actionability_gt = torch.mean(best_costs.detach(), dim=1)
        affordance_loss = affordance_loss_cross_entropy if self.cost_representation == 'success' else affordance_loss_L1
        loss_affordance = affordance_loss(predictions=cost_prediction.squeeze(),
                                          variance=cost_uncertainty.squeeze(),
                                          gt_labels=costs_gt.squeeze())

        # select only the ones where the gt interaction was successful
        successful_indexes = np.argwhere(costs_np < 0.99).squeeze()
        # orientation loss: the network should predict at least n orientations near a successful one
        topk = self.action_net.n_orientation_props
        orientation_loss = rotation_loss_w2a if self.orientation_dim == 6 else rotation_loss
        loss_action = orientation_loss(orientation_proposals=orientation_proposals[successful_indexes],
                                       ground_truth_orientation=orientation_gt[successful_indexes],
                                       n_orientation_proposals=self.action_net.n_orientation_props,
                                       n_topk=topk)

        loss_actionability = actionability_loss(predictions=actionability_prediction,
                                                variance=actionability_uncertainty,
                                                gt_labels=actionability_gt)
        if self.testing:
            # during testing
            return loss_affordance, loss_action, loss_actionability, cost_prediction, costs_gt
        else:
            # during training
            return loss_affordance, loss_action, loss_actionability

    def get_actionability_whole_pcd(self,
                                    points_np: np.ndarray,
                                    tasks_np: np.ndarray):
        # ---- GET INPUT DATA
        batch_size = 1
        points_gt, tasks_gt = self.to_torch(points_np, tasks_np)
        points_gt = torch.reshape(points_gt, (1, -1, 3))
        tasks_gt = torch.reshape(tasks_gt, (batch_size, 1))

        # ---- FEATURE EXTRACTION
        all_feats = self.feature_extractor(points_gt)
        point_feats = all_feats[0, :, :]
        interaction_xyz_gt = points_gt[0, :, :]

        # ---- PREDICT ACTIONABILITY
        act_pred, act_unc = self.actionability_net(pixel_feats=point_feats,
                                                   xyzs=interaction_xyz_gt,
                                                   tasks=torch.repeat_interleave(tasks_gt, point_feats.shape[0], dim=0))
        act_pred_np = act_pred.detach().cpu().numpy()
        act_unc_np = act_unc.detach().cpu().numpy()
        point_feats_np = point_feats.detach().cpu().numpy()
        return act_pred_np, act_unc_np, point_feats_np

    def get_affordance_whole_pcd(self,
                                 points_np: np.ndarray,
                                 orientations_np: np.ndarray,
                                 tasks_np: np.ndarray):
        """
        Predict affordance over an entire pointcloud (for each point one orientation and one task can be specified)
        This is used during online sampling, so that we can sample with only the warmstarted affordance module
        :param points_np: pointcloud (n_points, 3)
        :param orientations_np: (n_points, 4)
        :param tasks_np: (n_points, 1)
        :return: aff_pred_np, aff_unc_np, point_feats_np
        """
        # ---- GET INPUT DATA
        batch_size = points_np.shape[0]
        points_gt, orientations_gt, tasks_gt = self.to_torch(points_np, orientations_np, tasks_np)
        # only one pointcloud, so for the featurenet the batch size is 1
        points_gt = torch.reshape(points_gt, (1, -1, 3))

        # ---- FEATURE EXTRACTION
        all_feats = self.feature_extractor(points_gt)
        point_feats = all_feats[0, :, :]
        interaction_xyz_gt = points_gt[0, :, :]

        # ---- PREDICT AFFORDANCE
        # input dim for AffordanceNet are: pixel_feats (B, F), xyzs: (B, 3), orientations: (B, n_orientation_props, 4), tasks: (B, 1)
        tasks_gt = torch.reshape(tasks_gt, (batch_size, 1))
        orientations_gt = torch.reshape(orientations_gt, (batch_size, 1, -1))
        # forward pass
        aff_pred, aff_unc = self.affordance_net(pixel_feats=point_feats,
                                                xyzs=interaction_xyz_gt,
                                                orientations=orientations_gt,
                                                tasks=tasks_gt)

        aff_pred_np = aff_pred.detach().cpu().numpy()
        aff_unc_np = aff_unc.detach().cpu().numpy()
        point_feats_np = point_feats.detach().cpu().numpy()
        return aff_pred_np, aff_unc_np, point_feats_np

    def get_interaction_orientation(self,
                                    point_feat_np,
                                    xyz_np,
                                    task_np):
        point_feat, xyz, task = self.to_torch(point_feat_np, xyz_np, task_np)
        point_feat = torch.reshape(point_feat, (1, -1))
        xyz = torch.reshape(xyz, (1, -1))
        task = torch.reshape(task, (1, -1))
        # ---- ORIENTATION PROPOSALS
        orientation_proposals = self.action_net(pixel_feats=point_feat, xyzs=xyz, tasks=task)

        # ---- SCORING OF ORIENTATION PROPOSALS
        best_orientations, best_costs = self.score_orientation_proposals(orientation_proposals=orientation_proposals,
                                                                         point_feats=point_feat,
                                                                         interaction_xyz_gt=xyz,
                                                                         tasks_gt=task,
                                                                         best_of=self.action_net.n_orientation_props)
        return best_costs.detach().cpu().numpy(), best_orientations.detach().cpu().numpy()
