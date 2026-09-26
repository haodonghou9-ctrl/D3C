import os
import yaml
import torch
import numpy as np
import pandas as pd
import argparse


from robustbench.base_model import ConsistencyAE
from robustbench.metric import clustering_by_representation
from data_load import get_val_transformations, get_train_dataset


import cotta


def load_config(path='configs_dual.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_final_model(dataset_name, cfg):
    print(f"\n{'=' * 20} Testing Final Model: {dataset_name} {'=' * 20}")
    device = 'cuda'


    final_view_idx = cfg['views_total'] - 1


    model_dir = f"./last_sim_model/{dataset_name}_dual_offline"
    model_path = os.path.join(model_dir, f"best_model_view{final_view_idx}.pth")

    if not os.path.exists(model_path):
        print(f" Final model not found: {model_path}")
        return []

    print(f" Loading Final Model: {os.path.basename(model_path)}")


    net_cfg = cfg['network']
    base_model = ConsistencyAE(
        basic_hidden_dim=net_cfg['basic_hidden'],
        c_dim=cfg['c_dim'],
        continous=True,
        in_channel=cfg['in_channel'],
        num_res_blocks=3,
        ch_mult=net_cfg['ch_mult'],
        block_size=net_cfg['block_size'],
        latent_ch=net_cfg['latent_ch'],
        temperature=1.0, kld_weight=1.0,
        categorical_dim=cfg['class_num']
    ).to(device)


    model = cotta.CoTTA(
        base_model, optimizer=None,
        steps=1, episodic=False,
        num_classes=cfg['class_num'], CROP_SIZE=cfg['crop_size'],
        feature_dim=cfg['c_dim']
    ).to(device)


    try:
        model.load_state_dict(torch.load(model_path), strict=False)
    except Exception as e:
        print(f"Error loading: {e}")
        return []

    model.eval()


    transform = get_val_transformations(cfg['crop_size'])
    dataset = get_train_dataset(dataset_name, cfg['root'], cfg['views_total'], transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

    results = []


    for data_view_idx in range(cfg['views_total']):
        print(f"   Testing on Data View {data_view_idx} ...")

        all_feats = []
        all_targets = []

        with torch.no_grad():
            for batch_imgs, batch_targets in loader:
                img = batch_imgs[data_view_idx].to(device)


                if hasattr(model, 'student'):
                    _, z = model.student([img])
                else:
                    _, z = model([img])

                all_feats.append(z.cpu())
                all_targets.append(batch_targets)

        all_feats = torch.vstack(all_feats).numpy()
        all_targets = torch.cat(all_targets).numpy()


        all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=1e5, neginf=-1e5)


        acc, nmi, ari, _, _, fscore, _, _ = clustering_by_representation(all_feats, all_targets, cfg['class_num'])
        print(f"     -> ACC={acc:.4f}, NMI={nmi:.4f}")

        results.append({
            "Dataset": dataset_name,
            "Model": "Final_Dual_Model",
            "Data_View": data_view_idx,
            "ACC": acc,
            "NMI": nmi,
            "ARI": ari,
            "F-Score": fscore
        })

    return results


if __name__ == '__main__':
    #'coil-20','OfficeHome'
    TARGET_LIST = [ 'coil-20']
    import random

    full_cfg = load_config('configs.yaml')
    defaults = full_cfg['defaults']
    all_results = []

    for name in TARGET_LIST:
        if name not in full_cfg['datasets']:
            print(f"Skipping {name}")
            continue
        torch.manual_seed(999)
        torch.cuda.manual_seed_all(999)
        np.random.seed(999)
        random.seed(999)
        torch.backends.cudnn.deterministic = True

        cfg = defaults.copy()
        cfg.update(full_cfg['datasets'][name])
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_device']

        res = test_final_model(name, cfg)
        all_results.extend(res)

    if all_results:
        df = pd.DataFrame(all_results)
        cols = ["Dataset", "Model", "Data_View", "ACC", "NMI", "ARI", "F-Score"]
        df = df[cols]
        print("\n================ FINAL DUAL-STREAM REPORT ================")
        print(df)
        df.to_csv("test_report_final_dual.csv", index=False)
        print(f"\nResults saved to test_report_final_dual.csv")
    else:
        print("No results found.")