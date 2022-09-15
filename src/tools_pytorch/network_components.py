"""

The code in this file is based on the code published alongside the paper
Where2Act: From Pixels to Actions for Articulated 3D Objects(citation in the project README).
The MIT License from the original code is below.

Copyright 2022 Kaichun Mo

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch

from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule

from torch import nn as nn
from torch.nn import functional as F


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    """
    Feature extractor
    """

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formatted as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class FeatureNet(nn.Module):
    """
    Feature extractor wrapper
    """

    def __init__(self, point_feature_dimension):
        super(FeatureNet, self).__init__()
        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': point_feature_dimension})

    # input: points_and_features: B x N x 3 (float), with the 0th point to be the query point
    # output: extracted_features: B x point_feature_dimensions x N
    def forward(self, points):
        r"""
            Forward pass of the network
            Parameters
            ----------
            points: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formatted as (x, y, z, features...)
        """
        # found the repeat command in the w2a code (action_scoring_only.py line 128)
        points = points.repeat(1, 1, 2)
        extracted_features = self.pointnet2(points)
        # get to dimensions (B, N, FEAT_DIM)
        extracted_features = torch.transpose(extracted_features, 1, 2)
        return extracted_features


class Constants:
    XYZ_DIM = 3
    # ORIENTATION_DIM = 4
    TASK_DIM = 1
    RANDOM_VECTOR_DIM = 7
    FEAT_DIM = 128


class AffordanceNet(nn.Module, Constants):
    """
    Predict cost/utility given:
      - interaction point feature from the feature extractor (B x point_feature_dimension)
      - interaction point xyz coordinates in camera space (to predict reach) (B x 3)
      - interaction orientation as quaternion (B x 4)
      - relative task (B x 1)
    Output is:
      - cost estimate
      - estimated uncertainty
    """

    def __init__(self, point_feature_dimension, orientation_dimension=4, sigmoid_output=False):
        super(AffordanceNet, self).__init__()
        # dimension is feature + orientation + task
        self.orientation_dimension = orientation_dimension
        self.sigmoid_output = sigmoid_output
        input_dim = point_feature_dimension + orientation_dimension + self.XYZ_DIM + self.TASK_DIM
        self.mlp1 = nn.Linear(input_dim, point_feature_dimension)
        self.mlp2 = nn.Linear(point_feature_dimension, 2)

    # input: pixel_feats (B, F), xyzs: (B, 3), orientations: (B, 4), tasks: (B, 1)
    # output: (B, 1), (B, 1) (cost and uncertainty)
    def forward_util(self, pixel_feats, xyzs, orientations, tasks):
        net = torch.cat([pixel_feats, xyzs, orientations, tasks], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net).squeeze(-1)
        affordance = net[:, 0]
        if self.sigmoid_output: affordance = torch.sigmoid(affordance)
        # uncertainty prediction must always be positive
        uncertainty = torch.square(net[:, 1])
        return affordance, uncertainty

    # input: pixel_feats (B, F), xyzs: (B, 3), orientations: (B, n_orientation_props, 4), tasks: (B, 1)
    # (more than one orientation proposal for each point)
    # output: (B, n_orientation_props, 1), (B, n_orientation_props, 1) (cost and uncertainty)
    def forward(self, pixel_feats, xyzs, orientations, tasks):
        # get inputs to the correct shape: orientation_proposals is (B, n_orientation_props, 4)
        n_orientation_props = orientations.shape[1]
        flat_orientation_proposals = torch.reshape(orientations, (-1, self.orientation_dimension))  # (B*n_orientation_props, 4)
        rept_point_feats = torch.repeat_interleave(pixel_feats, n_orientation_props, dim=0)
        rept_xyzs = torch.repeat_interleave(xyzs, n_orientation_props, dim=0)
        rept_tasks = torch.repeat_interleave(tasks, n_orientation_props, dim=0)
        # orient_cost_pred.shape is (B*n_orientation_props, 1)
        affordance, uncertainty = self.forward_util(pixel_feats=rept_point_feats,
                                                    xyzs=rept_xyzs,
                                                    orientations=flat_orientation_proposals,
                                                    tasks=rept_tasks)

        # get orient_cost_pred.shape to (B, n_orientation_props, 1)
        affordance = torch.reshape(affordance, (-1, n_orientation_props, 1))
        uncertainty = torch.reshape(uncertainty, (-1, n_orientation_props, 1))
        return affordance, uncertainty


# input sz bszx3x2
def r6_to_matrix(d6s):
    bsz = d6s.shape[0]
    selection = d6s[:, :, 0]
    b1 = F.normalize(selection, p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)


class ActionNet(nn.Module, Constants):
    """
    Generate a good interaction orientation given:
      - interaction point feature from the feature extractor (B x point_feature_dimension)
      - interaction point xyz coordinates in camera space (to account for reach) (B x 3)
      - relative task (B x 1)
    Output is:
      - orientation proposal as quaternion (B x 4)
    """

    def __init__(self, point_feature_dimension, n_orientation_proposals=100, orientation_dimension=4):
        super(ActionNet, self).__init__()
        self.orientation_dimension = orientation_dimension
        # dimension is feature + orientation + task
        input_dim = point_feature_dimension + self.XYZ_DIM + self.RANDOM_VECTOR_DIM + self.TASK_DIM
        self.mlp1 = nn.Linear(input_dim, point_feature_dimension)
        self.mlp2 = nn.Linear(point_feature_dimension, orientation_dimension)
        self.n_orientation_props = n_orientation_proposals

    # input is (B*n_orientation_props, FEAT_DIM), (B*n_orientation_props, 3), (B*n_orientation_props, 1)
    def forward_util(self, pixel_feats, xyzs, tasks):
        random_vector = torch.normal(0, 1.0, size=(tasks.shape[0], self.RANDOM_VECTOR_DIM),
                                     dtype=tasks.dtype, device=tasks.device)
        # # use a ones vector during debugging to see if reshaping is done correctly
        # random_vector = torch.ones((tasks.shape[0], self.RANDOM_VECTOR_DIM),
        #                            dtype=tasks.dtype, device=tasks.device)
        net = torch.cat([pixel_feats, xyzs, tasks, random_vector], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net).squeeze(-1)
        # normalize to unit length to make sure it's a valid rotation quaternion
        if self.orientation_dimension == 4:
            # if it's a quaternion
            net = F.normalize(net, p=2, dim=1)
        elif self.orientation_dimension == 6:
            # r6 encoding of rotation
            # here the convention is (qx, px, qy, py, qz, pz)
            net = net.reshape(-1, 3, 2)
            net = r6_to_matrix(net)[:, :, :2]

            # from here we have (qx, qy, qz, px, py, pz)
            net = net.permute(0, 2, 1)
            net = net.reshape(-1, 6)
        return net

    # input is (B, FEAT_DIM), (B, 3), (B, 1)
    # output is (B, n_orientation_props, 4)
    # where B is now the number of points for which we want to calculate orientation proposals
    def forward(self, pixel_feats, xyzs, tasks):
        # each point in the batch is a different interaction point
        # for each interaction point, generate multiple orientation proposals
        batch_size = tasks.shape[0]
        pixel_feats_r = torch.repeat_interleave(pixel_feats, self.n_orientation_props, dim=0)
        xyzs_r = torch.repeat_interleave(xyzs, self.n_orientation_props, dim=0)
        tasks_r = torch.repeat_interleave(tasks, self.n_orientation_props, dim=0)

        orientation_proposals = self.forward_util(pixel_feats=pixel_feats_r,
                                                  xyzs=xyzs_r,
                                                  tasks=tasks_r)  # (B*n_orientation_props, 4)
        # reshape so that every point is separate
        # orientation_proposals[i, :, :] are the orientation proposals corresponding to one specific point
        orientation_proposals = torch.reshape(orientation_proposals, (batch_size, self.n_orientation_props, -1))
        return orientation_proposals


class ActionabilityNet(nn.Module, Constants):
    """
    Predict cost/utility given:
      - interaction point feature from the feature extractor (B x point_feature_dimension)
      - interaction point xyz coordinates in camera space (to predict reach) (B x 3)
      - relative task (B x 1)
    Output is:
      - actionability estimate
      - estimated uncertainty
    """

    def __init__(self, point_feature_dimension):
        super(ActionabilityNet, self).__init__()
        # dimension is feature + orientation + task
        input_dim = point_feature_dimension + self.XYZ_DIM + self.TASK_DIM
        self.mlp1 = nn.Linear(input_dim, point_feature_dimension)
        self.mlp2 = nn.Linear(point_feature_dimension, 2)

    # pixel_feats is (B, FEAT_DIM), xyzs is (B, 3), tasks is (B, 1)
    # actionability.shape is (B, 1) where B is the number of points in the batch
    def forward(self, pixel_feats, xyzs, tasks):
        net = torch.cat([pixel_feats, xyzs, tasks], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net).squeeze(-1)

        actionability = torch.reshape(net[:, 0], (-1, 1))
        # uncertainty prediction must always be positive
        uncertainty = torch.reshape(torch.square(net[:, 1]), (-1, 1))
        return actionability, uncertainty
