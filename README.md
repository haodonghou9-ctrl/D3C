# D3C: Decoupling Exploration and Retention for Continual Multi-view Clustering

PyTorch implementation of:

**Decoupling Exploration and Retention: A Dual-Stream Framework for Continual Multi-view Clustering**


**ACM Multimedia 2026 (MM '26)**

DOI: `10.1145/3767308.3835617`

---

## Installation

The code was trained and tested with the following environment:

```text
Python>=3.8
PyTorch==1.12.1
torchvision==0.13.1
torchaudio==0.12.1
```
The experiments reported in the paper were conducted on a single
**NVIDIA RTX 3090 GPU with 24 GB memory**.

---

## Repository Structure

```text
D3C/
│
├── MyData/
│   └── ...                         # Dataset files
│
├── source/
│   └── <dataset>_dual/
│       ├── best_source_model.pth
│       ├── source_kmeans_pre.npy
│       ├── source_kmeans_center.npy
│       └── best_source_features.npy
│
├── last_sim_model/
│   └── <dataset>_dual_offline/
│       ├── best_model_view0.pth
│       ├── best_model_view1.pth
│       ├── ...
│       ├── res_v0.npy
│       ├── cen_v0.npy
│       └── ...
│
├── robustbench/
│   └── ...                         # Model, metric and utility modules
│
├── configs.yaml                    # Dataset and training configurations
├── cotta.py                        # Core D3C dual-stream implementation
├── data_load.py                    # Dataset loading
├── main.py                         # Training pipeline
├── my_transforms.py                
├── test.py                         # Evaluation using trained checkpoints
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Datasets
All datasets are loaded from:

```text
./MyData
```

Please place the dataset files under `MyData/` in the format expected by
`data_load.py`.

---

## Configuration

All dataset-specific settings are defined in `configs.yaml`.

The global defaults are:

```yaml
root: "./MyData"
seed: 3407
cuda_device: "0"
method: "Adam"
beta: 0.9
wd: 0.0
```

An example configuration for COIL-20 is:

```yaml
coil-20:
  class_num: 20
  views_total: 3
  num_ex: 480
  in_channel: 1
  crop_size: 64
  c_dim: 20
  batch_size: 128
  lr: 0.0001
  epochs: 50

  contra: 1.0
  consis: 1.0

  network:
    basic_hidden: 32
    latent_ch: 8
    block_size: 8
    ch_mult: [1, 2, 4, 8]
```

---

# Quick Start

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd D3C
```

If model checkpoints are stored using Git LFS:

```bash
git lfs install
git lfs pull
```

---

## 2. Create the Environment

```bash
conda create -n d3c python>=3.8.20
conda activate d3c
pip install -r requirements.txt
```

---

## 3. Prepare the Dataset

Place the dataset under:

```text
./MyData/
```

For example, the default evaluation script currently uses:

```python
TARGET_LIST = ['coil-20']
```

The dataset name must match the corresponding entry in `configs.yaml`.

---

## 4. Evaluate the Pre-trained Model

Run:

```bash
python test.py
```

The script loads the final D3C checkpoint and evaluates the final
**Base/Student model** on all historical views.

For each view:

1. representations are extracted from the final Base model;
2. K-means clustering is performed;
3. clustering metrics are calculated.

The following metrics are reported:

- **ACC** — Clustering Accuracy
- **NMI** — Normalized Mutual Information
- **ARI** — Adjusted Rand Index
- **F-Score**

The complete evaluation results are saved to:

```text
test_report_final_dual.csv
```

---

# Training from Scratch

The complete training pipeline is implemented in:

```text
main.py
```

## 1. Select the Dataset

Modify `TARGET_LIST` in `main.py`.

---

## 2. Run Training

```bash
python main.py
```


## Reproducibility

For reproducibility, the repository contains:

- the D3C implementation;
- dataset-specific configurations;
- environment dependencies;
- warm-up checkpoint support;
- continual-learning checkpoints;
- clustering assignments and centers;
- evaluation scripts.


---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{jiang2026d3c,
  title     = {Decoupling Exploration and Retention: A Dual-Stream Framework for Continual Multi-view Clustering},
  author    = {Guangqi Jiang and Haodong Hou and Yi Liu and Jinjia Peng and Huibing Wang},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3835617}
}
```

---

## Acknowledgements


We thank the authors of the open-source implementations and benchmark datasets that supported this research. 

Part of our code and methodology is inspired by and built upon **AdaptCMVC**. If you find our repository helpful, please also consider citing their work:

> **AdaptCMVC: Robust Adaption to Incremental Views in Continual Multi-view Clustering**
> Jing Wang, Songhe Feng, Kristoffer Knutsen Wickstrøm, and Michael C. Kampffmeyer.
> *2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*

```bibtex
@INPROCEEDINGS{11093621,
  author={Wang, Jing and Feng, Songhe and Wickstrøm, Kristoffer Knutsen and Kampffmeyer, Michael C.},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={AdaptCMVC: Robust Adaption to Incremental Views in Continual Multi-view Clustering}, 
  year={2025},
  volume={},
  number={},
  pages={10285-10294},
  doi={10.1109/CVPR52734.2025.00962}
}

---

---

## Contact

For questions regarding the paper or implementation, please contact us.
