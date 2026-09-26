import os
import yaml
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse


from robustbench.base_model import ConsistencyAE
from robustbench.data import load_multiview
from robustbench.metric import clustering_by_representation
from robustbench.sim_utils import clean_accuracy_target as accuracy_target
from robustbench.utils import clean_accuracy_source as accuracy_source
from data_load import get_val_transformations, get_train_dataset
import cotta



def load_config(path='configs.yaml'):
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)


def build_base_model(config, device):
    net_cfg = config['network']
    model = ConsistencyAE(
        basic_hidden_dim=net_cfg['basic_hidden'], c_dim=config['c_dim'],
        continous=True, in_channel=config['in_channel'], num_res_blocks=3,
        ch_mult=net_cfg['ch_mult'], block_size=net_cfg['block_size'],
        latent_ch=net_cfg['latent_ch'], temperature=1.0, kld_weight=1.0,
        categorical_dim=config['class_num']
    ).to(device)
    return model


def setup_optimizer(params, cfg):
    if cfg['method'] == 'Adam':
        return optim.Adam(params, lr=cfg['lr'], betas=(cfg['beta'], 0.999), weight_decay=cfg['wd'])
    elif cfg['method'] == 'SGD':
        return optim.SGD(params, lr=cfg['lr'], momentum=0.9, weight_decay=cfg['wd'], nesterov=True)
    else:
        raise NotImplementedError


def update_knowledge_base(model, x_full_view, y_full_view, view_idx, save_dir, class_num, batch_size, device='cuda'):
    print(f"Updating Knowledge Base for View {view_idx} ...")
    n_samples = x_full_view.shape[0]
    n_batches = math.ceil(n_samples / batch_size)
    all_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            x_batch = x_full_view[start:end].to(device)
            if hasattr(model, 'student'):
                _, z = model.student([x_batch])
            else:
                _, z = model([x_batch])
            all_feats.append(z.cpu())
    all_feats = torch.vstack(all_feats).numpy()
    y_numpy = y_full_view.cpu().numpy() if y_full_view.is_cuda else y_full_view.numpy()
    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=1e5, neginf=-1e5)
    acc, nmi, _, _, _, _, new_labels, new_centers = clustering_by_representation(all_feats, y_numpy, class_num)
    print(f"Knowledge Updated! View {view_idx}: ACC={acc:.4f}")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'res_v{view_idx}.npy'), new_labels)
    np.save(os.path.join(save_dir, f'cen_v{view_idx}.npy'), new_centers)
    return new_labels


def run_source_training(model, x_view0, y_gt, save_dir, config, device):
    print(f"\n>>> Phase 1: Source Training (View 0)...")
    os.makedirs(save_dir, exist_ok=True)
    optimizer = setup_optimizer(model.parameters(), config)
    best_acc = 0.
    current_loss=float('inf')
    n_samples = x_view0.shape[0]
    n_batches = math.ceil(n_samples / config['batch_size'])

    for epoch in range(100):
        model.train()
        indices = torch.randperm(n_samples)
        x_shuffled = x_view0[indices]
        total_loss = 0.
        for i in range(n_batches):
            start = i * config['batch_size']
            end = min((i + 1) * config['batch_size'], n_samples)
            img = x_shuffled[start:end].to(device)
            loss, _ = model.get_loss([img], epoch, 0.5, 4)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            result, kmeans_pre, kmeans_center, out_reprs = accuracy_source(model, x_view0, y_gt, config['batch_size'],
                                                                   config['class_num'], device)
            acc = result['consist-acc']
            print(f"Source Epoch {epoch}: Loss={total_loss / n_batches:.4f} | ACC={acc:.4f}")
            if (total_loss / n_batches) <= current_loss:
                current_loss=total_loss / n_batches
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(save_dir, "best_source_model.pth"))
                np.save(os.path.join(save_dir, "source_kmeans_pre.npy"), kmeans_pre)
                np.save(os.path.join(save_dir, "source_kmeans_center.npy"), kmeans_center)


                feature_path = os.path.join(save_dir, "best_source_features.npy")
                np.save(feature_path, out_reprs)
                print(f"  Feature Saved: {feature_path} (Shape: {out_reprs.shape})")
    print(f"Source Done. Best ACC: {best_acc:.4f}\n")


def run_dataset_experiment(dataset_name, cfg):
    cfg['_name'] = dataset_name
    print(f"\n{'=' * 20} Running: {dataset_name} {'=' * 20}")
    device = 'cuda'

    transform = get_val_transformations(cfg['crop_size'])
    dataset = get_train_dataset(dataset_name, cfg['root'], cfg['views_total'], transform)
    x_all, y_all = load_multiview(cfg['num_ex'], False, dataset)

    source_dir = f"./source/{dataset_name}_dual"
    source_model_path = os.path.join(source_dir, "best_source_model.pth")
    base_model = build_base_model(cfg, device)

    if not os.path.exists(source_model_path):
        x_view0 = x_all[0].to(device)
        y_gt = y_all.to(device)
        run_source_training(base_model, x_view0, y_gt, source_dir, cfg, device)
        base_model.load_state_dict(torch.load(source_model_path))
    else:
        print(f"Found source model: {source_model_path}")
        base_model.load_state_dict(torch.load(source_model_path))


    model = cotta.CoTTA(
        base_model, optimizer=None,
        steps=1, episodic=False,
        num_classes=cfg['class_num'], CROP_SIZE=cfg['crop_size'],
        contra=cfg.get('contra', 1.0), consis=cfg.get('consis', 1.0),
        feature_dim=cfg['c_dim'], N=1,alpha_recon=cfg.get('alpha_recon',0.01),lambda_distill=cfg.get('lambda_distill',0.5)
    ).to(device)

    params_T = cotta.collect_params_teacher(model.teacher, model.projector)
    model.optimizer_T = setup_optimizer(params_T, cfg)
    params_S = cotta.collect_params_student(model.student)
    model.optimizer_S = setup_optimizer(params_S, cfg)

    result_dir = f"./last_sim_model/{dataset_name}_dual_offline"
    os.makedirs(result_dir, exist_ok=True)

    init_s_path = os.path.join(source_dir, "source_kmeans_pre.npy")
    c_path = os.path.join(source_dir, "source_kmeans_center.npy")
    init_labels = torch.from_numpy(np.load(init_s_path)).long()
    label_bank = [init_labels]


    for view_idx in range(cfg['views_total']):
        print(f"\n--- Processing View {view_idx} ---")

        x_curr = x_all[view_idx].to(device)
        y_gt = y_all.to(device)
        history_tensor = torch.stack(label_bank, dim=1).to(device)

        if view_idx == 0:
            source_center = np.load(c_path)
        else:
            source_center = np.load(os.path.join(result_dir, f'cen_v{view_idx - 1}.npy'))

        best_acc = 0.
        best_model_path = os.path.join(result_dir, f"best_model_view{view_idx}.pth")

        # total_epochs = 100
        # teacher_epochs = total_epochs // 2
        # student_epochs = total_epochs - teacher_epochs
        teacher_epochs = cfg['epochs']
        student_epochs = cfg['epochs']

        print(f"Plan: {teacher_epochs} eps Teacher -> {student_epochs} eps Student")

        # ---------------------------------
        # Stage 1: Train Teacher Only
        # ---------------------------------
        model.set_stage('teacher')
        t_loss=float('inf')
        for i in range(teacher_epochs):
            result, _, avg_losses = accuracy_target(source_center, history_tensor, model, x_curr, y_gt,
                                                    cfg['batch_size'], cfg['class_num'], view_idx, 0.0, device)
            if (i+1)%5==0:
                print(
                    f"Stage 1 Ep {i+1}: [T]ACC:{result['teacher-acc']:.4f} | Loss: T-Con={avg_losses['loss_teacher_con']:.3f}")
    #             current_teacher_loss = (
    #     cfg.get('consis', 1.0) * avg_losses['loss_teacher_con']
    #     + cfg.get('alpha_recon', 0.01) * avg_losses['loss_recon']
    # )
                if t_loss >=avg_losses['loss_teacher_con']:
                    t_loss=avg_losses['loss_teacher_con']
                    best_acc = result['teacher-acc']
                    torch.save(model.state_dict(), best_model_path)
                    print(f" Best Teacher Saved: {best_acc:.4f}")


        print(f"Stage 1 Finished. Best Teacher ACC was: {best_acc:.4f}")
        print(f">> Reloading Best Teacher Model for Stage 2 Distillation...")


        model.load_state_dict(torch.load(best_model_path))
        best_acc=0.0
        # ---------------------------------
        # Stage 2: Train Student Only
        # ---------------------------------
        s_loss=float('inf')
        model.set_stage('student')
        for i in range(student_epochs):
            result, _, avg_losses = accuracy_target(source_center, history_tensor, model, x_curr, y_gt,
                                                    cfg['batch_size'], cfg['class_num'], view_idx, 0.0, device)

            print(
                f"Stage 2 Ep {i}: [S]ACC:{result['consist-acc']:.4f} | Loss: Distill={avg_losses['loss_distill']:.3f}")

            if s_loss >= avg_losses['loss_distill']:
                s_loss=avg_losses['loss_distill']
                best_acc = result['consist-acc']
                torch.save(model.state_dict(), best_model_path)
                print(f"  New Best Saved: {best_acc:.4f}")

        print(f'View {view_idx} Best ACC: {best_acc:.4f}')


        print(f">> Reloading best model...")
        model.load_state_dict(torch.load(best_model_path))
        new_labels = update_knowledge_base(model, x_curr, y_gt, view_idx, result_dir, cfg['class_num'],
                                           cfg['batch_size'], device)

        new_labels_tensor = torch.from_numpy(new_labels).long()
        if view_idx == 0:
            label_bank = [new_labels_tensor]
            print("View 0 Re-Adapt Done. Overwrote Init S.")
        else:
            label_bank.append(new_labels_tensor)
            print("Appended to Label Bank.")



if __name__ == '__main__':
    import random
    #'coil-20','OfficeHome'
    TARGET_LIST = [ 'coil-20']
    full_cfg = load_config('configs.yaml')
    defaults = full_cfg['defaults']

    for name in TARGET_LIST:
        if name not in full_cfg['datasets']: continue
        cfg = defaults.copy()
        cfg.update(full_cfg['datasets'][name])
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_device']
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])
        torch.backends.cudnn.deterministic = True

        try:
            run_dataset_experiment(name, cfg)
        except Exception as e:
            print(f" Error running {name}: {e}")