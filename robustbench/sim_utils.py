import argparse
import dataclasses
import json
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import nn
import numpy as np


from robustbench.metric import clustering_by_representation


def clean_accuracy_source(model: nn.Module,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          batch_size: int = 100,
                          class_num: int = 10,
                          device: torch.device = None):
    if device is None:
        device = x.device

    n_batches = math.ceil(x.shape[0] / batch_size)
    targets = []
    out_reprs = []

    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) * batch_size]


            ret = model([x_curr])
            if isinstance(ret, tuple):
                _, output_z = ret
            else:
                output_z = ret

            targets.append(y_curr.detach().cpu())
            out_reprs.append(output_z.detach().cpu())

    targets = torch.concat(targets, dim=-1).numpy()
    out_reprs = torch.vstack(out_reprs).detach().cpu().numpy()


    acc, nmi, ari, _, p, fscore, kmeans_pre, kmeans_center = clustering_by_representation(out_reprs, targets, class_num)

    result = {}
    result['consist-acc'] = acc
    result['consist-nmi'] = nmi
    result['consist-ari'] = ari
    result['consist-p'] = p
    result['consist-fscore'] = fscore

    return result, kmeans_pre, kmeans_center


def clean_accuracy_target(source_center: np.array,
                          source_result: torch.Tensor,
                          model: nn.Module,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          batch_size: int = 100,
                          class_num: int = 10,
                          views: int = 1,
                          up_alpha: float = 0.1,
                          device: torch.device = None):
    if device is None:
        device = x.device
    if source_result.dim()==1:
        source_result=source_result.unsqueeze(1)
    history_views=source_result.shape[1]

    n_batches = math.ceil(x.shape[0] / batch_size)

  
    out_reprs_student = []
    out_reprs_teacher = []
    targets = []


    avg_losses = {
        'loss_recon': 0.,
        'loss_distill': 0.,
        'loss_str': 0.,
        'loss_teacher_con': 0.
    }


    with torch.no_grad():
        for counter in range(n_batches):
            start = counter * batch_size
            end = (counter + 1) * batch_size

            # 1. 数据切片
            x_curr = x[start:end].to(device)
            y_curr = y[start:end].to(device)

            # # 切片上一轮标签 (Batch_Size, )
            # y_last_curr = source_result[start:end].to(device)
            #取出当前历史标签库 [B,V]
            history_pre_batch=source_result[start:end].to(device)
            #初始化全局预测标签投票箱
            vote_sum=torch.zeros(history_pre_batch.shape[0],history_pre_batch.shape[0]).to(device)

            for i in range(history_views):
                pres=history_pre_batch[:,i]
                adj=(pres.unsqueeze(0)==pres.unsqueeze(1)).float()
                vote_sum+=adj

            vote_rate=vote_sum/history_views

            consensus_S=(vote_rate>0.5).float()


            input = ([x_curr], None, source_center, consensus_S)


            z_s, z_t, loss_dict = model(input)


            targets.append(y_curr.detach().cpu())
            out_reprs_student.append(z_s.detach().cpu())
            out_reprs_teacher.append(z_t.detach().cpu())


            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    avg_losses[k] += v.item()
                else:
                    avg_losses[k] += v


    targets = torch.concat(targets, dim=-1).numpy()
    out_reprs_student = torch.vstack(out_reprs_student).detach().cpu().numpy()
    out_reprs_teacher = torch.vstack(out_reprs_teacher).detach().cpu().numpy()


    acc_s, nmi_s, ari_s, _, p_s, fscore_s, kmeans_pre_s, _ = clustering_by_representation(out_reprs_student, targets,
                                                                                          class_num)


    acc_t, nmi_t, _, _, _, _, _, _ = clustering_by_representation(out_reprs_teacher, targets, class_num)

    result = {}

    result['consist-acc'] = acc_s
    result['consist-nmi'] = nmi_s
    result['consist-ari'] = ari_s
    result['consist-p'] = p_s
    result['consist-fscore'] = fscore_s


    result['teacher-acc'] = acc_t
    result['teacher-nmi'] = nmi_t


    for k in avg_losses:
        avg_losses[k] /= n_batches


    return result, kmeans_pre_s, avg_losses