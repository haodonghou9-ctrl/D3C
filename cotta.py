from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
import numpy as np
import math


# ================= KAN Linear =================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02,
                 grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features,
                                                                                                       -1).contiguous())
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.spline_scaler = None
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features,
                                 self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + (
                        (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        bases = self.b_splines(x)
        spline_output = F.linear(bases.view(x.size(0), -1), self.spline_weight.view(self.out_features, -1))
        if self.enable_standalone_scale_spline:
            spline_output = spline_output + F.linear(x, self.spline_scaler)
        return base_output + spline_output


def get_tta_transforms(gaussian_std: float = 0.005, CROP_SIZE=32, soft=True):
    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(brightness=[0.8, 1.2], contrast=[0.85, 1.15], saturation=[0.75, 1.25],
                                     hue=[-0.03, 0.03], gamma=[0.85, 1.15]),
        transforms.Pad(padding=int(CROP_SIZE / 8), padding_mode='edge'),
        transforms.RandomAffine(degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.9, 1.1),
                                interpolation=PIL.Image.BILINEAR),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.5]),
        transforms.CenterCrop(size=CROP_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(0.0, 1.0)
    ])
    return tta_transforms


def supervised_contrastive_loss(features, mask, temperature=0.1):
    device = features.device
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    batch_size = features.shape[0]
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
    return -mean_log_prob_pos.mean()


def compute_entropy_weights(z, prototypes, temperature=1.0):
    z = F.normalize(z, dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    dists = 2.0 - 2.0 * torch.matmul(z, prototypes.t())
    probs = F.softmax(-dists / temperature, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
    weights = 1.0 + entropy
    weights = weights / (weights.mean() + 1e-7)
    return weights.detach()


class ProjectionHead(nn.Module):
    def __init__(self, dim_in=20, dim_out=20):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in),
            KANLinear(dim_in, dim_in, grid_size=5, spline_order=3),
            nn.LayerNorm(dim_in),
            KANLinear(dim_in, dim_out, grid_size=5, spline_order=3)
        )

    def forward(self, x):
        return self.net(x)


# ================= CoTTA =================
class CoTTA(nn.Module):
    def __init__(self, model, optimizer=None, steps=1, episodic=False,
                 num_classes=20, CROP_SIZE=32, contra=1.0, consis=1.0, N=2,
                 feature_dim=20,alpha_recon=1,lambda_distill=0.5):
        super().__init__()

        self.student = model
        self.teacher = deepcopy(model)

        self.student.train().requires_grad_(True)
        self.teacher.train().requires_grad_(True)

        self.projector = ProjectionHead(dim_in=feature_dim, dim_out=feature_dim).cuda()
        self.projector.train().requires_grad_(True)

        self.optimizer_T = None
        self.optimizer_S = None

        self.transform = get_tta_transforms(0.005, CROP_SIZE, False)
        self.similarity = nn.CosineSimilarity(dim=2)

        self.steps = steps
        self.episodic = episodic
        self.consis = consis
        self.contra = contra

        self.alpha_recon = alpha_recon
        self.lambda_distill = lambda_distill


        self.current_stage = 'teacher'

    def set_stage(self, stage):
        assert stage in ['teacher', 'student']
        self.current_stage = stage

    def forward(self, x):
        if self.episodic:
            self.reset()

        last_loss_dict = {}
        for _ in range(self.steps):
            _, loss_dict = self.forward_and_adapt(x, None)
            last_loss_dict = loss_dict

        imgs = x[0]
        input_img = imgs[0]
        with torch.no_grad():
            _, z_student = self.student([input_img])
            _, z_teacher = self.teacher([input_img])

        return z_student, z_teacher, last_loss_dict

    def reset(self):
        pass

    @torch.enable_grad()
    def forward_and_adapt(self, x, optimizer):
        imgs = x[0]
        input_img = imgs[0]
        consensus_mask = x[3].float()

        source_center = x[2]
        if not torch.is_tensor(source_center):
            source_center = torch.from_numpy(source_center).float().cuda()
        else:
            source_center = source_center.float().cuda()


        z_out = None
        loss_dict = {
            'loss_recon': 0.0, 'loss_distill': 0.0,
            'loss_str': 0.0, 'loss_teacher_con': 0.0
        }

        # ==========================================
        # Phase 1: Teacher Training
        # ==========================================
        if self.current_stage == 'teacher':
            self.teacher.train()
            self.projector.train()

            # 1.1 Recon & KL
            mu_weak, logvar_weak = self.teacher.encode([input_img])
            z_teacher_weak = self.teacher.cont_reparameterize(mu_weak, logvar_weak)
            recon_weak = self.teacher.decode(z_teacher_weak)[0]
            loss_recon_T = F.mse_loss(recon_weak, input_img, reduction='sum') / input_img.size(0)

            aug_img = self.transform(input_img)
            mu_strong, logvar_strong = self.teacher.encode([aug_img])
            z_teacher_strong = self.teacher.cont_reparameterize(mu_strong, logvar_strong)
            recon_strong = self.teacher.decode(z_teacher_strong)[0]
            loss_recon_T += F.mse_loss(recon_strong, input_img, reduction='sum') / input_img.size(0)

            loss_kl_T = -0.5 * torch.sum(1 + logvar_weak - mu_weak.pow(2) - logvar_weak.exp(), dim=1).mean()

            # 1.2 SupCon (Double Filtering)
            features = torch.cat([self.projector(z_teacher_weak), self.projector(z_teacher_strong)], dim=0)

            label_mask_bool = (consensus_mask > 0.0)

            with torch.no_grad():
                z_norm = F.normalize(z_teacher_weak, dim=1)
                sim_matrix = torch.matmul(z_norm, z_norm.t())
                sim_no_diag = sim_matrix.clone()
                sim_no_diag.fill_diagonal_(-1.0)
                max_sim, _ = sim_no_diag.max(dim=1, keepdim=True)
                sim_thresh_mask = sim_matrix > (0.4 * max_sim)

            final_sim_mask = (label_mask_bool & sim_thresh_mask).float()
            final_sim_mask.fill_diagonal_(1.0)
            mask = final_sim_mask.repeat(2, 2)

            loss_con = supervised_contrastive_loss(features, mask, temperature=0.1)

            # 1.3 Structure
            cos_sim_teacher = self.similarity(z_teacher_weak.unsqueeze(1), z_teacher_weak.unsqueeze(0))
            loss_str_T = F.mse_loss(consensus_mask, cos_sim_teacher)

            total_loss_T = self.consis * loss_con + self.contra * loss_str_T + self.alpha_recon * (
                        loss_recon_T + loss_kl_T)

            if self.optimizer_T is not None:
                self.optimizer_T.zero_grad()
                total_loss_T.backward()
                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.projector.parameters(), max_norm=1.0)
                self.optimizer_T.step()

            z_out = z_teacher_weak
            loss_dict['loss_teacher_con'] = loss_con.item()
            loss_dict['loss_recon'] = loss_recon_T.item()

        # ==========================================
        # Phase 2: Student Training (Distillation)
        # ==========================================
        elif self.current_stage == 'student':

            self.teacher.eval()
            self.student.train()

            with torch.no_grad():
                mu_T, logvar_T = self.teacher.encode([input_img])
                z_target = self.teacher.cont_reparameterize(mu_T, logvar_T)

            mu_S, logvar_S = self.student.encode([input_img])
            z_student = self.student.cont_reparameterize(mu_S, logvar_S)
            recon_S = self.student.decode(z_student)[0]

            # Distill
            entropy_weights = compute_entropy_weights(z_student, source_center)
            loss_distill = 1.0 - F.cosine_similarity(z_student, z_target.detach(), dim=1)
            loss_distill = (loss_distill * entropy_weights).mean()

            # Structure & Recon
            cos_sim_student = self.similarity(z_student.unsqueeze(1), z_student.unsqueeze(0))
            loss_str_S = F.mse_loss(consensus_mask, cos_sim_student)

            loss_recon_S = F.mse_loss(recon_S, input_img, reduction='sum') / input_img.size(0)
            loss_kl_S = -0.5 * torch.sum(1 + logvar_S - mu_S.pow(2) - logvar_S.exp(), dim=1).mean()

            total_loss_S = self.lambda_distill * loss_distill + \
                           self.contra * loss_str_S + \
                           self.alpha_recon * (loss_recon_S + loss_kl_S)

            if self.optimizer_S is not None:
                self.optimizer_S.zero_grad()
                total_loss_S.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.optimizer_S.step()

            z_out = z_student
            loss_dict['loss_distill'] = loss_distill.item()
            loss_dict['loss_recon'] = loss_recon_S.item()
            loss_dict['loss_str'] = loss_str_S.item()

        return z_out, loss_dict



def configure_model(model):
    model.train()
    model.requires_grad_(True)
    return model


def collect_params_teacher(model, projector):
    params = []
    for m in [model, projector]:
        for p in m.parameters():
            if p.requires_grad: params.append(p)
    return params


def collect_params_student(model):
    params = []
    for p in model.parameters():
        if p.requires_grad: params.append(p)
    return params