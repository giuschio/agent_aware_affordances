import torch


def gaussian_density_loss(predictions, variance, gt_labels):
    """
    Based on this: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    Not available in pytorch 1.7
    :param predictions: cost predictions
    :param variance: variance predictions
    :param gt_labels: cost ground truth labels
    :return: scalar loss
    """
    eps = 1e-2 * torch.ones_like(variance)
    error = predictions - gt_labels
    variance_n = torch.maximum(eps, variance)
    loss_v = 0.5 * (torch.log(variance_n) + torch.div(torch.pow(error, 2), variance_n))
    return torch.mean(loss_v)


def laplacian_density_loss(predictions, variance, gt_labels):
    """
    :param predictions: cost predictions
    :param variance: variance predictions
    :param gt_labels: cost ground truth labels
    :return: scalar loss
    """
    eps = 1e-2 * torch.ones_like(variance)
    error = predictions - gt_labels
    variance_n = torch.maximum(eps, variance)
    loss_v = torch.log(variance_n) + torch.div(torch.abs(error), variance_n)
    return torch.mean(loss_v)


def l2_loss(predictions, variance, gt_labels):
    if predictions.shape != gt_labels.shape:
        raise ValueError("predictions.shape and gt_labels.shape are different")
    return torch.mean(torch.square(predictions - gt_labels))


def l1_loss(predictions, variance, gt_labels):
    if predictions.shape != gt_labels.shape:
        raise ValueError("predictions.shape and gt_labels.shape are different")
    return torch.mean(torch.abs(predictions - gt_labels))


def cross_entropy_loss(predictions, variance, gt_labels):
    if predictions.shape != gt_labels.shape:
        raise ValueError("predictions.shape and gt_labels.shape are different")
    loss = torch.nn.BCELoss()
    return loss(predictions, gt_labels)


def rotation_loss(orientation_proposals,
                  ground_truth_orientation,
                  n_orientation_proposals,
                  n_topk):
    # orientation_proposals.shape = (B, N_ORIENT_PROPS, 4), orientation_gt.shape = (B, 4)
    # loss from paper: "On the Merits of Joint Space and Orientation Representations in Learning the Forward Kinematics in SE(3)"
    # we follow the SAPIEN quaternion convention q = (w,x,y,z)
    op = orientation_proposals
    gt = torch.reshape(ground_truth_orientation, shape=(-1, 1, 4))
    gt = torch.repeat_interleave(input=gt, repeats=n_orientation_proposals, dim=1)
    tmp = torch.multiply(op, gt)
    err = 2.0 * torch.acos(torch.sum(tmp, dim=2))
    min_err, _ = torch.topk(err, k=n_topk, largest=False, dim=1)
    res = torch.mean(min_err)
    return res


def rotation_loss_w2a(orientation_proposals,
                      ground_truth_orientation,
                      n_orientation_proposals,
                      n_topk):
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
    # orientation_proposals.shape = (B, N_ORIENT_PROPS, 6), orientation_gt.shape = (B, 6)
    op = orientation_proposals
    gt = torch.reshape(ground_truth_orientation, shape=(-1, 1, 6))
    gt = torch.repeat_interleave(input=gt, repeats=n_orientation_proposals, dim=1)
    tmp = torch.abs(op - gt)
    res = torch.mean(tmp)
    return res
