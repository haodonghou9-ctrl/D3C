import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# from robustbench.model_zoo.enums import BenchmarkDataset
# from robustbench.zenodo_download import DownloadError, zenodo_download
# from robustbench.loaders import CustomImageFolder






def load_multiview(n_examples,shuffle,dataset) -> Tuple[torch.Tensor, torch.Tensor]:




    test_loader = data.DataLoader(dataset, num_workers=8, batch_size=n_examples,
                                  sampler= None,shuffle=shuffle,pin_memory=True,
                                    drop_last=True)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test




