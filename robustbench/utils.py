import argparse
import dataclasses
import json
import math
import os
import warnings
# import gdown
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import requests
import torch
from torch import nn

# from robustbench.model_zoo import model_dicts as all_models
# from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.metric import clustering_by_representation,clustering_accuracy
import numpy as np


def clean_accuracy_source(model: nn.Module,#######################
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   class_num: int=10,
                   device: torch.device = None):

    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    targets = []
    out_reprs = []

    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size]

            _, output_z = model([x_curr])

            targets.append(y_curr.detach().cpu())
            out_reprs.append(output_z.detach().cpu())

    targets = torch.concat(targets, dim=-1).numpy()
    out_reprs = torch.vstack(out_reprs).detach().cpu().numpy()




    acc, nmi, ari, _, p, fscore, kmeans_pre, kmeans_center = clustering_by_representation(out_reprs, targets,class_num)

    result = {}
    result['consist-acc'] = acc
    result['consist-nmi'] = nmi
    result['consist-ari'] = ari
    result['consist-p'] = p
    result['consist-fscore'] = fscore

    return result, kmeans_pre, kmeans_center,out_reprs

def clean_accuracy_target(source_center: np.array,
                          source_result: torch.Tensor,
                          model: nn.Module,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       batch_size: int = 100,
                       class_num: int = 10,
                       device: torch.device = None):

        if device is None:
            device = x.device
        acc = 0.
        n_batches = math.ceil(x.shape[0] / batch_size)
        targets = []
        out_reprs = []
        last_pre = []
        r_loss = 0.
        c_loss = 0.

        with torch.no_grad():
            for counter in range(n_batches):


                x_curr = x[counter * batch_size:(counter + 1) *
                                                batch_size].to(device)

                y_curr = y[counter * batch_size:(counter + 1) *
                                                batch_size].to(device)

                y_last = source_result[counter * batch_size:(counter + 1) *
                                                batch_size].to(device)
                input = ([x_curr],y_last,source_center)
                output_z,rec_loss,consis_loss = model(input)  #

                targets.append(y_curr.detach().cpu())
                out_reprs.append(output_z.detach().cpu())
                last_pre.append(y_last.detach().cpu())
                r_loss += rec_loss
                c_loss += consis_loss

            targets = torch.concat(targets, dim=-1).numpy()
            out_reprs = torch.vstack(out_reprs).detach().cpu().numpy()
            last_pre = torch.concat(last_pre, dim=-1).numpy()

            acc, nmi, ari, _, p, fscore, kmeans_pre, cluster_center = clustering_by_representation(out_reprs, targets, class_num)

            result = {}
            result['consist-acc'] = acc
            result['consist-nmi'] = nmi
            result['consist-ari'] = ari
            result['consist-p'] = p
            result['consist-fscore'] = fscore

            return result, kmeans_pre,r_loss/n_batches, c_loss/n_batches, kmeans_pre, cluster_center


