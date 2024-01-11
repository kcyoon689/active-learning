# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/AL-SSL/


# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import voc300 as cfg
from ..box_utils import match, log_sum_exp
import numpy as np


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets, semis=[]):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        losses_l = torch.empty(size=(num, 1), device=loc_data.device)
        for idx in range(num):
            single_loc_p = loc_data[idx][pos_idx[idx]]
            single_loc_t = loc_t[idx][pos_idx[idx]]
            single_loss_l = F.smooth_l1_loss(single_loc_p, single_loc_t, size_average=False)
            losses_l[idx] = single_loss_l

        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(pos.size()[0], pos.size()[1])
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        real_labels, pseudo_labels = [False] * len(semis), [False] * len(semis)
        for i, el in enumerate(semis):
            if el == torch.FloatTensor([1.]):
                real_labels[i] = True
            elif el == torch.FloatTensor([2.]):
                pseudo_labels[i] = True

        losses_c = torch.empty(size=(len(semis), 1), device=conf_data.device)
        for idx in range(len(semis)):
            if real_labels[idx]:
                single_conf_p_labels = conf_data[idx][(pos_idx[idx]+neg_idx[idx]).gt(0)]
                targets_weighted_labels = conf_t[idx][(pos[idx] + neg[idx]).gt(0)]
            elif pseudo_labels[idx]:
                single_conf_p_labels = conf_data[idx][(pos_idx[idx]).gt(0)]
                targets_weighted_labels = conf_t[idx][(pos[idx]).gt(0)]
            else:
                raise ValueError('Unlabeld data is included in supervised dataset')

            single_conf_p_labels = single_conf_p_labels.view(-1, self.num_classes)
            single_loss_c = F.cross_entropy(single_conf_p_labels, targets_weighted_labels, size_average=False)
            losses_c[idx] = single_loss_c

        pos_idx_labels = pos_idx[real_labels]
        neg_idx = neg_idx[real_labels]
        conf_data_real = conf_data[real_labels]
        conf_t_real = conf_t[real_labels]
        pos_real = pos[real_labels]
        neg_real = neg[real_labels]
        pos_idx_pseudo = pos_idx[pseudo_labels]
        conf_data_pseudo = conf_data[pseudo_labels]
        conf_t_pseudo = conf_t[pseudo_labels]
        pos_pseudo = pos[pseudo_labels]

        conf_p_labels = conf_data_real[(pos_idx_labels+neg_idx).gt(0)].view(-1, self.num_classes)
        conf_p_pseudo = conf_data_pseudo[(pos_idx_pseudo).gt(0)].view(-1, self.num_classes)
        targets_weighted_labels = conf_t_real[(pos_real + neg_real).gt(0)]
        targets_weighted_pseudo = conf_t_pseudo[(pos_pseudo).gt(0)]

        loss_c_real = F.cross_entropy(conf_p_labels, targets_weighted_labels, size_average=False)
        if targets_weighted_pseudo.shape[0] > 0:
            loss_c_pseudo = F.cross_entropy(conf_p_pseudo, targets_weighted_pseudo, size_average=False)
            loss_c = loss_c_real + loss_c_pseudo
        else:
            loss_c = loss_c_real

        N = num_pos.data.sum()
        losses_l /= N
        losses_c /= N
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c, torch.squeeze(losses_l), torch.squeeze(losses_c)
