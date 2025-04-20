import time

import torch
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import v2
from logging import INFO
import numpy as np
import pandas as pd
from GlobalParameters import *
from Dependency.ACGAN import ACGAN
from Dependency.Dataset import DatasetLabeled, DatasetUnlabeled, DatasetPL
from torchvision.utils import save_image, make_grid
from pathlib import Path
import math
from utils.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period=10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# Diffusion model related
class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10, task='fashion'):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.time_dim = 64

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(self.time_dim, 2 * n_feat)
        self.timeembed2 = EmbedFC(self.time_dim, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

        ks = 7 if task in ['fashion', 'mnist'] else 8
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, ks, ks),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed context, time step
        t = timestep_embedding(t, self.time_dim)
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def get_named_beta_schedule(schedule_name='linear', num_diffusion_timesteps=1000) -> np.ndarray:
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta=0.999) -> np.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = torch.linspace(beta1, beta2, T)
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    # for calculating mean from predicted noise in the reverse process
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, n_once=1):
        super(DPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.beta_t)
        self.condition = nn.functional.one_hot(torch.arange(classes_n), num_classes=classes_n).repeat(n_once, 1).to(
            device)
        self.uncondition = torch.zeros(classes_n * n_once, classes_n).to(device)
        model_fn = model_wrapper(
            nn_model,
            noise_schedule,
            model_type="noise",  # or "x_start" or "v" or "score"
            guidance_type="classifier-free",
            condition=self.condition,
            unconditional_condition=self.uncondition,
            guidance_scale=guide_w,
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
        self.dpm_solver = dpm_solver

    def calc_loss(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(0, self.n_T, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(len(c)) + self.drop_prob).to(self.device)
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = 1 - context_mask  # need to flip 0 <-> 1
        c = c * context_mask

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, _ts, c))

    def forward(self, x_t, _ts, c):
        return self.nn_model(x_t, _ts, c)

    def sample_loop(self, sn, size, n_once=100):
        g_s_list, g_l_list = [], []
        for i in range(int(sn // n_once)):
            # g_s, g_l = self.sample(n_once, size)
            g_s, g_l = self.quick_sample(n_once, size)
            g_s_list.append(g_s.to('cpu'))
            g_l_list.append(g_l.to('cpu'))
        g_s_list = torch.cat(g_s_list, dim=0)
        g_l_list = torch.cat(g_l_list, dim=0)
        return g_s_list, g_l_list

    def sample(self, n_sample, size):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        device = self.device

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0, classes_n).to(device)  # context for us just cycles throught the labels
        c_i = nn.functional.one_hot(c_i, num_classes=classes_n).type(torch.float)
        c_i = c_i.repeat((int(n_sample / c_i.shape[0]), 1))

        # don't drop context at test time
        context_mask = torch.zeros(n_sample).to(device)[:, None]

        # double the batch
        c_i = c_i.repeat((2, 1))
        context_mask = context_mask.repeat(2, 1)
        context_mask[n_sample:] = 1.  # makes second half of batch context free
        context_mask = (1 - context_mask)  # need to flip 0 <-> 1
        # for i in range(self.n_T, 0, -1):
        for i in reversed(range(self.n_T)):
            # print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor(i).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            c_i_ = c_i * context_mask
            # split predictions and compute weighting
            eps = self.nn_model(x_i, t_is, c_i_)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )
        return x_i, c_i[:n_sample]

    def quick_sample(self, n_sample, size):
        x_i = torch.randn(n_sample, *size).to(self.device)  # x_T ~ N(0, 1), sample initial noise
        x_i = self.dpm_solver.sample(
            x_i,
            steps=20,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
        return x_i, self.condition


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    # split the mixed inputs into three lists, according to the batch size
    # i.e. split labeled, u1 and u2 (for each) into the same three parts
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        # swap the splitted part with the unlabeled one
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def MixUpData(all_inputs, all_targets):
    l = np.random.beta(beta_param, beta_param)
    l = max(l, 1 - l)
    idx = torch.randperm(all_inputs.size(0), device=all_inputs.device)
    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b
    return mixed_input, mixed_target


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, self.expansion * planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        # out = self.bn2(self.conv2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def vis_gsamples(gen_x, mean, std, worker_id, round_id, vis_path):
    mean, std = torch.tensor(mean)[None], torch.tensor(std)[None]
    g_sample = gen_x * std + mean
    g_sample = g_sample.clip(0, 1)
    # grid = make_grid(g_sample * -1 + 1, nrow=10)
    grid = make_grid(g_sample, nrow=10)
    save_image(grid, str(vis_path / f"image_{worker_id}_round_{round_id}_w{guide_w}.png"))
    return


def iter_loader(loader_iter, loader, iter_id):
    loader_length = len(loader)
    if iter_id % loader_length == 0:
        loader_iter = iter(loader)
        samples, labels, index = next(loader_iter)
    else:
        samples, labels, index = next(loader_iter)
    return loader_iter, samples, labels, index


class NeuralNetwork(nn.Module):
    def __init__(self, device='cuda', model='cnn', task='mnist', worker_id=-1, ftime=None, logger=None):
        super(NeuralNetwork, self).__init__()
        self.worker_id = worker_id
        self.logger = logger
        if task == 'cifar10':
            self.c, self.size = 3, 32
            self.mean, self.std = (0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)
        else:
            self.c, self.size = 1, 28
            self.mean, self.std = (0.5,), (0.5,)

        # model structure
        if model == 'cnn':
            self.embedding = nn.Sequential(
                nn.Conv2d(1, 6, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 25, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Linear(49 * 25, 50, True),
                nn.ReLU(),
                nn.Linear(50, 10, True),
            )
        elif model == 'resnet18':
            assert task == 'cifar10'
            self.embedding = ResNet18(classes_n)

        # task description
        self.model = model
        self.task = task
        self.lr = lr

        if ssl_method in ['DFLSemi']:
            if gen_model == 'dpm':
                drop_prob = 0.15
                dpm = DPM(nn_model=ContextUnet(self.c, 128, n_classes=classes_n, task=task), betas=(1e-4, 0.02),
                          n_T=1000, device=device, drop_prob=drop_prob, n_once=g_size_once//classes_n)
                dpm.to(device)
                self.dpm_optim = torch.optim.Adam(dpm.parameters(), lr=dlr)
            elif gen_model == 'gan':
                img_size = 28 if task in ['fashion', 'mnist'] else 32
                gan = ACGAN(img_dim=self.c, label_dim=classes_n, img_size=img_size, device=device).to(device)
                dpm = gan
                dpm.to(device)

            self.dpm = dpm
            self.dloss_ema = None  # only for log

        # optimizer and model's device
        self.optimizer = tc.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.to(self.device)

        # datasets
        self.labeled_train_loader = None
        self.unlabeled_train_loader = None
        self.test_loader = None
        self.test_samples, self.test_labels, self.test_transform = None, None, None
        self.size_trainingset = 0
        self.size_testingset = 0
        self.g_samples, self.g_labels = None, None
        self.g_test_samples, self.g_test_labels = None, None
        self.r_samples, self.p_labels = [], []
        self.train_transform = None  # transform for diffusion generated samples
        self.cur_params = None
        self.adj_params = None
        self.sigma_c_old, self.sigma_c_new = None, None

        # ckpts
        if not debug and worker_id != -1:
            self.ckpt_path = Path(f'ckpts/{task}/{ftime}')
            self.ckpt_path.mkdir(exist_ok=True, parents=True)
            self.vis_path = Path(f'visualize/dpm/{task}')
            self.vis_path.mkdir(exist_ok=True, parents=True)
        if worker_id != -1:
            self.save_path = Path(f'outputs/{task}/{ftime}')
            self.save_path.mkdir(exist_ok=True, parents=True)
        self.history_loss_train = []
        self.history_acc = []
        self.g_acc = None

    def forward(self, input):
        if self.model == 'cnn':
            x = self.embedding[:6](input)
            x = self.embedding[6:](x.view(x.size(0), -1))
        else:
            x = self.embedding(input)
        return x

    def Train(self, epoch: int = 1, round_id: int = 0):
        if self.worker_id in l_client_ids:
            self.TrainLabeled(epoch, round_id)
        elif self.worker_id in m_client_ids:
            self.TrainMixed(epoch, round_id)
        else:
            self.TrainUnlabeled(epoch, round_id)
        if self.g_samples is not None and self.g_labels is not None:
            assert round_id + 1 >= sample_start_round
            self.TestOnGSamples()

    def TrainDiffusionLoop(self, d_train_loader, loop_num, round_id):
        device = self.device
        logger = self.logger
        if gen_model == 'dpm':
            optim = self.dpm_optim
            dpm = self.dpm
            if pl_ablation != 'none' and self.worker_id in u_client_ids and (round_id + 1) <= sample_start_round:
                return

            if self.worker_id in l_client_ids + m_client_ids:
                d_train_iter = iter(d_train_loader)
            if self.worker_id in m_client_ids + u_client_ids:
                r_all_samples, p_all_labels = torch.cat(self.r_samples), torch.cat(self.p_labels)
            for i in range(loop_num):
                if self.worker_id in l_client_ids + m_client_ids:
                    try:
                        samples, labels, _ = next(d_train_iter)
                    except:
                        d_train_iter = iter(d_train_loader)
                        samples, labels, _ = next(d_train_iter)
                if self.worker_id in l_client_ids:
                    x, c = samples.to(device), labels.to(device)
                elif self.worker_id in m_client_ids:
                    bs, len_r = local_batch_size, len(r_all_samples)
                    rand_idx = torch.randperm(len_r)[:bs]
                    r_samples, p_labels = r_all_samples[rand_idx], p_all_labels[rand_idx]
                    x, c = torch.cat([samples, r_samples]), torch.cat([labels, p_labels])
                    x, c = x.to(device), c.to(device)
                else:
                    bs, len_r = local_batch_size, len(r_all_samples)
                    rand_idx = torch.randperm(len_r)[:bs]
                    r_samples, p_labels = r_all_samples[rand_idx], p_all_labels[rand_idx]
                    x, c = r_samples.to(device), p_labels.to(device)

                optim.zero_grad()
                loss = dpm.calc_loss(x, c)
                loss.backward()
                if self.dloss_ema is None:
                    self.dloss_ema = loss.item()
                else:
                    self.dloss_ema = 0.95 * self.dloss_ema + 0.05 * loss.item()
                optim.step()
            logger.log(INFO, f'diffusion loss: {self.worker_id}: {self.dloss_ema:.4f}')
            return self.dloss_ema
        elif gen_model == 'gan':
            if self.worker_id in l_client_ids + m_client_ids:
                d_train_iter = iter(d_train_loader)
            if self.worker_id in m_client_ids + u_client_ids:
                r_all_samples, p_all_labels = torch.cat(self.r_samples), torch.cat(self.p_labels)
            for i in range(loop_num):
                if self.worker_id in l_client_ids + m_client_ids:
                    try:
                        samples, labels, _ = next(d_train_iter)
                    except:
                        d_train_iter = iter(d_train_loader)
                        samples, labels, _ = next(d_train_iter)
                if self.worker_id in l_client_ids:
                    x, c = samples.to(device), labels.to(device)
                elif self.worker_id in m_client_ids:
                    bs, len_r = local_batch_size, len(r_all_samples)
                    rand_idx = torch.randperm(len_r)[:bs]
                    r_samples, p_labels = r_all_samples[rand_idx], p_all_labels[rand_idx]
                    x, c = torch.cat([samples, r_samples]), torch.cat([labels, p_labels])
                    x, c = x.to(device), c.to(device)
                else:
                    bs, len_r = local_batch_size, len(r_all_samples)
                    rand_idx = torch.randperm(len_r)[:bs]
                    r_samples, p_labels = r_all_samples[rand_idx], p_all_labels[rand_idx]
                    x, c = r_samples.to(device), p_labels.to(device)

                d_loss, g_loss, d_corr_num, num = self.dpm.train_batch(x, c)
            logger.log(INFO, f'GAN loss: {self.worker_id}: d_loss: {d_loss}, g_loss: {g_loss}')
            return g_loss, d_loss

    @torch.no_grad()
    def GenerateSamples(self, s_num, sample_shape):
        g_samples, g_labels = self.dpm.sample_loop(s_num, sample_shape, n_once=g_size_once)
        return g_samples, g_labels

    def TrainLabeled(self, epoch, round_id):
        # load the aggregated model weights
        if round_id != 0:
            self.load_state_dict(self.cur_params)
            # self.to(device)
            for k in self.state_dict().keys():
                if not torch.allclose(self.state_dict()[k], self.cur_params[k]):
                    # print(self.worker_id, round_id, k, self.state_dict()[k].cpu(), self.cur_params[k])
                    # self.load_state_dict(self.cur_params, strict=True)
                    print(self.worker_id, round_id, k)
                # assert torch.allclose(self.state_dict()[k], self.cur_params[k])
        device, logger = self.device, self.logger
        l_iter = iter(self.labeled_train_loader)
        c, size = self.c, self.size
        for epoch_id in range(0, epoch):
            if ssl_method == 'DFLSemi':
                # sample before training at each client
                if (round_id + 1) >= sample_start_round and (round_id + 1) % sample_round_interval == 0:
                    self.g_samples, self.g_labels = self.GenerateSamples(1000, (c, size, size))
                    self.g_test_samples, self.g_test_labels = self.GenerateSamples(100, (c, size, size))
                    # if not debug:
                    #     self.SaveCKPTs(round_id)
                    #     vis_gsamples(self.g_samples, self.mean, self.std, self.worker_id, round_id, self.vis_path)
                if round_id != 0:
                    self.TrainDiffusionLoop(self.labeled_train_loader, train_dpm_loop_num, round_id)

            if self.g_samples is not None:
                g_samples, g_labels = self.g_samples, self.g_labels
                bs, len_r = local_batch_size, len(g_samples)
                rand_idx = torch.randperm(len_r)
            # Train Classifier
            for iter_id in range(epoch_iteration_num):
                l_iter, samples_x, labels_x, index_x = iter_loader(l_iter, self.labeled_train_loader, iter_id)

                samples_x, labels_x = samples_x.to(device), labels_x.to(device)
                if ssl_method in ['MixMatch']:
                    mixed_input, mixed_target = MixUpData(samples_x, labels_x)
                elif ssl_method in ['DFLSemi']:
                    g_samples, g_labels = self.g_samples, self.g_labels
                    if self.g_samples is not None:
                        cur_ridx = rand_idx[iter_id * bs:iter_id * bs + bs]
                        extra_samples, extra_labels = g_samples[cur_ridx].to(device), g_labels[cur_ridx].to(device)
                        all_inputs = torch.cat([samples_x, extra_samples])
                        all_targets = torch.cat([labels_x, extra_labels])
                    else:
                        all_inputs = samples_x
                        all_targets = labels_x
                    mixed_input, mixed_target = MixUpData(all_inputs, all_targets)
                else:
                    mixed_input = samples_x
                    mixed_target = labels_x

                output = self.forward(mixed_input)
                loss = torch.sum(F.log_softmax(output, dim=1) * mixed_target, dim=1)
                loss = -torch.mean(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.history_loss_train.append(float(loss.item()))
        logger.log(INFO, f'Current round final loss: {self.worker_id} {float(loss.item())}')
        return

    @torch.no_grad()
    def GeneratePlLoader(self, round_id):
        if ssl_method in ['CBAFed', 'DFLSemi']:
            sigma_c_old, sigma_c_new = self.sigma_c_old, self.sigma_c_new
        st, device = time.time(), self.device
        if round_id != 0 and neighbor_pl:
            # print('Check params: ', self.worker_id)
            for k in self.state_dict().keys():
                if not torch.allclose(self.state_dict()[k], self.cur_params[k]):
                    self.logger.log(INFO, f'!!!!! params not equal {self.worker_id, round_id, k}')
        sigma_c = {i: 0 for i in range(-1, classes_n)}
        un_iter = iter(self.unlabeled_train_loader)
        all_indexes_u, all_labels_u, all_targets_u = [], [], []
        for samples_u, labels_u, index_u in un_iter:
            p = 0
            su = samples_u[0]
            su = su.to(device)
            p_out = torch.softmax(self.forward(su), dim=-1)
            p += p_out
            if round_id != 0 and neighbor_pl:
                self.load_state_dict(self.adj_params)
                su = samples_u[-1]
                su = su.to(device)
                p_out = torch.softmax(self.forward(su), dim=-1)
                p += p_out
                self.load_state_dict(self.cur_params)
            else:
                su = samples_u[-1]
                su = su.to(device)
                p_out = torch.softmax(self.forward(su), dim=-1)
                p += p_out
            p /= len(samples_u)
            pt = p ** (1 / sharp_temperature)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach().cpu()
            if ssl_method == 'CBAFed':
                pred_c = targets_u.argmax(dim=1)
                if sigma_c_old is None:
                    sigma_c_old = targets_u.new_ones(classes_n) * thre_pl
                fidx = torch.where(targets_u.max(dim=1)[0] > sigma_c_old[pred_c])[0].cpu()
            else:
                fidx = torch.where(targets_u.max(dim=1)[0] > thre_pl)[0].cpu()
            sigma_c[-1] += len(index_u) - len(fidx)
            for fi in fidx:
                sigma_c[targets_u[fi].argmax().tolist()] += 1
            all_indexes_u.append(index_u)
            all_labels_u.append(labels_u)
            all_targets_u.append(targets_u)

        all_indexes_u = torch.cat(all_indexes_u)
        all_labels_u = torch.cat(all_labels_u)
        all_targets_u = torch.cat(all_targets_u)
        all_pred_c = all_targets_u.argmax(dim=1)
        if ssl_method == 'MixMatch':
            # MixMatch no filtering
            filter_idxs = torch.where(all_targets_u.max(dim=1)[0] > 0)[0]
        elif ssl_method == 'FlexMatch':
            thres_c = torch.tensor([thre_pl * sigma_c[i] / max(sigma_c.values()) for i in range(classes_n)])
            filter_idxs = torch.where(all_targets_u.max(dim=1)[0] > thres_c[all_pred_c])[0]
        elif ssl_method == 'CBAFed':
            # hyperparameters from the Fig. 4 in the original paper
            T_base = 0.85
            T_lower = 0.07
            T_upper = 0.95

            local_labels = torch.tensor([sigma_c[ci] for ci in range(classes_n)]).float()
            if self.worker_id in m_client_ids:
                cur_labels = self.labeled_train_loader.dataset.labels.argmax(dim=-1)
                for cl in cur_labels:
                    local_labels[cl] += 1
            filter_idxs = torch.where(all_targets_u.max(dim=1)[0] > sigma_c_old[all_pred_c])[0]
            one_hot_labels = torch.zeros_like(all_targets_u[filter_idxs])
            one_hot_labels.scatter_(1, all_targets_u[filter_idxs].argmax(dim=-1).view(-1, 1), 1)
            all_targets_u[filter_idxs] = one_hot_labels
            # include second
            uns_idxs = torch.where(all_targets_u.max(dim=1)[0] <= sigma_c_old[all_pred_c])[0]
            uns_p = all_targets_u[uns_idxs]
            uns_p[:, uns_p.argmax(dim=-1)] = 0
            max_uns_p = uns_p.max(dim=-1)[0]
            second_idxs = uns_idxs[torch.where(max_uns_p < T_lower)[0]]
            second_one_hot_labels = torch.zeros_like(all_targets_u[second_idxs])
            second_one_hot_labels.scatter_(1, uns_p[max_uns_p < T_lower].argmax(dim=-1).view(-1, 1), 1)
            all_targets_u[second_idxs] = second_one_hot_labels
            filter_idxs = torch.cat([filter_idxs, second_idxs])
            for cl in uns_p.argmax(dim=-1):
                local_labels[cl] += 1
            if torch.sum(local_labels) > 0:
                local_labels /= torch.sum(local_labels)
            class_confident = local_labels + T_base - local_labels.std()
            class_confident[class_confident >= T_upper] = T_upper
            self.sigma_c_old = class_confident
            all_pred_c = all_targets_u.argmax(dim=1)
        else:
            # DFLSemi
            if pl_ablation == 'none':
                self.sigma_c_new = dict(filter(lambda x: x[0] != -1, sigma_c.items()))
                if (round_id + 1) > 1 and with_ni:
                    max_sigma_c = max(max(sigma_c.values()), max(sigma_c_old.values()))
                else:
                    max_sigma_c = max(sigma_c.values())
                thres_c = torch.tensor([thre_pl * sigma_c[i] / max_sigma_c for i in range(classes_n)])
                filter_idxs = torch.where(all_targets_u.max(dim=1)[0] > thres_c[all_pred_c])[0]
            elif pl_ablation == 'vanilla':
                filter_idxs = torch.where(all_targets_u.max(dim=1)[0] > 0)[0]
            elif pl_ablation == 'apl':
                max_sigma_c = max(sigma_c.values())
                thres_c = torch.tensor([thre_pl * sigma_c[i] / max_sigma_c for i in range(classes_n)])
                filter_idxs = torch.where(all_targets_u.max(dim=1)[0] > thres_c[all_pred_c])[0]
        if len(filter_idxs) > 0:
            pl_acc = torch.sum(all_labels_u[filter_idxs].argmax(dim=1) == all_pred_c[filter_idxs]) / len(filter_idxs)
        else:
            pl_acc = 0
        self.logger.log(INFO, f"Current epoch acc of pseudo labels worker {self.worker_id}: {pl_acc}")
        sample_idxes = all_indexes_u[filter_idxs]
        if len(filter_idxs) < local_batch_size:
            print('Generate pl loader cost time: ', time.time() - st)
            return None
        cur_epoch_pl_data = DatasetPL(samples=self.unlabeled_train_loader.dataset.samples[sample_idxes],
                                      labels=all_targets_u[filter_idxs],
                                      ori_indexes=sample_idxes,
                                      transform=self.train_transform,
                                      classes=self.unlabeled_train_loader.dataset.classes)
        cur_epoch_pl_loader = data.DataLoader(cur_epoch_pl_data, batch_size=local_batch_size, shuffle=True,
                                              num_workers=0, pin_memory=True, drop_last=True)
        print('Generate pl loader cost time: ', time.time() - st)
        return cur_epoch_pl_loader

    @torch.no_grad()
    # generate pseudo labels for each batch (in device)
    def GeneratePlBatch(self, samples_u, labels_u, index_u, round_id):
        if ssl_method in ['FlexMatch', 'DFLSemi']:
            sigma_c, sigma_c_new = self.sigma_c_old, self.sigma_c_new
        device = self.device
        labels_u = labels_u.to(device)
        p = 0
        for ku in samples_u:
            ku = ku.to(device)
            p_out = torch.softmax(self.forward(ku), dim=-1)
            p += p_out
        p /= len(samples_u)
        pt = p ** (1 / sharp_temperature)
        targets_u = pt / pt.sum(dim=1, keepdim=True)
        targets_u = targets_u.detach()
        if ssl_method == 'MixMatch':
            fidx = torch.where(targets_u.max(dim=1)[0] > 0)[0]
        elif ssl_method == 'FlexMatch':
            # FlexMatch
            fidx = torch.where(targets_u.max(dim=1)[0] > thre_pl)[0]
            # calculate thres_c and update sigma_c
            sigma_c_new[-1] += len(index_u) - len(fidx)
            for fi in fidx:
                sigma_c_new[targets_u[fi].argmax().tolist()] += 1
            self.sigma_c_new = sigma_c_new
            if (round_id + 1) != 1:
                thres_c = torch.tensor([thre_pl * sigma_c[i] / max(sigma_c.values()) for i in range(classes_n)]).to(
                    device)
                pred_c = targets_u.argmax(dim=1)
                fidx = torch.where(targets_u.max(dim=1)[0] > thres_c[pred_c])[0]
        elif ssl_method == 'DFLSemi':
            if pl_ablation == 'vanilla':
                fidx = torch.where(targets_u.max(dim=1)[0] > 0)[0]
            elif pl_ablation == 'apl':
                fidx = torch.where(targets_u.max(dim=1)[0] > thre_pl)[0]
                # calculate thres_c and update sigma_c
                sigma_c_new[-1] += len(index_u) - len(fidx)
                for fi in fidx:
                    sigma_c_new[targets_u[fi].argmax().tolist()] += 1
                self.sigma_c_new = sigma_c_new
                if (round_id + 1) != 1:
                    thres_c = torch.tensor([thre_pl * sigma_c[i] / max(sigma_c.values()) for i in range(classes_n)]).to(
                        device)
                    pred_c = targets_u.argmax(dim=1)
                    fidx = torch.where(targets_u.max(dim=1)[0] > thres_c[pred_c])[0]

        samples_u = [ku.to(device)[fidx] for ku in samples_u]
        labels_u = labels_u[fidx]
        index_u = index_u[fidx.to('cpu')]
        targets_u = targets_u[fidx]
        total_label_num = len(labels_u)  # the really predicted number of labels
        if len(targets_u) > 0:
            correct_label_num = len(torch.where(targets_u.argmax(dim=1) == labels_u.argmax(dim=1))[0])
        else:
            correct_label_num = 0
        nu = len(samples_u)
        targets_u = targets_u.repeat([nu, 1])
        samples_u = torch.cat(samples_u, dim=0)
        return samples_u, labels_u, index_u, targets_u, correct_label_num, total_label_num

    def TrainMixed(self, epoch: int = 1, round_id: int = 0):
        if ssl_method in ['FlexMatch', 'DFLSemi']:
            self.sigma_c_old = self.sigma_c_new
            self.sigma_c_new = {i: 0 for i in range(-1, classes_n)}
        # load the aggregated model weights
        if round_id != 0:
            self.load_state_dict(self.cur_params)
            # self.to(device)
        device, logger = self.device, self.logger
        l_iter = iter(self.labeled_train_loader)
        ul_iter = iter(self.unlabeled_train_loader) if self.unlabeled_train_loader else None
        pl_iter = None
        if fixed_pl:
            cur_epoch_pl_loader = self.GeneratePlLoader(round_id)
            if cur_epoch_pl_loader is not None:
                pl_iter = iter(cur_epoch_pl_loader)

        c, size = self.c, self.size
        c_label_num, total_label_num = 0, 0
        for epoch_id in range(0, epoch):
            if ssl_method == 'DFLSemi':
                # sample before training at each client
                if (round_id + 1) >= sample_start_round and (round_id + 1) % sample_round_interval == 0:
                    self.g_samples, self.g_labels = self.GenerateSamples(1000, (c, size, size))
                    self.g_test_samples, self.g_test_labels = self.GenerateSamples(100, (c, size, size))
                    # if not debug:
                    #     self.SaveCKPTs(round_id)
                if round_id != 0:
                    self.TrainDiffusionLoop(self.labeled_train_loader, train_dpm_loop_num, round_id)

            if self.g_samples is not None:
                g_samples, g_labels = self.g_samples, self.g_labels
                bs, len_r = local_batch_size, len(g_samples)
                rand_idx = torch.randperm(len_r)
            # real samples and pseudo labels for diffusion training
            self.r_samples, self.p_labels = [], []
            for iter_id in range(epoch_iteration_num):
                l_iter, samples_x, labels_x, index_x = iter_loader(l_iter, self.labeled_train_loader, iter_id)
                samples_x, labels_x = samples_x.to(device), labels_x.to(device)

                # only use labeled data
                if ssl_method == 'Supervise' or (fixed_pl and pl_iter is None):
                    mixed_input, mixed_target = samples_x, labels_x
                else:
                    # generating pseudo label
                    if fixed_pl:
                        pl_iter, samples_u, targets_u, index_u = iter_loader(pl_iter, cur_epoch_pl_loader, iter_id)
                        samples_u, targets_u = samples_u.to(device), targets_u.to(device)
                    else:
                        tmp_data = iter_loader(ul_iter, self.unlabeled_train_loader, iter_id)
                        ul_iter, samples_u, labels_u, index_u = tmp_data
                        tmp_data = self.GeneratePlBatch(samples_u, labels_u, index_u, round_id)
                        samples_u, labels_u, index_u, targets_u, cn, tn = tmp_data
                        c_label_num += cn
                        total_label_num += tn

                    if ssl_method in ['MixMatch']:
                        # mixup
                        all_inputs = torch.cat([samples_x, samples_u], dim=0)
                        all_targets = torch.cat([labels_x, targets_u], dim=0)
                        mixed_input, mixed_target = MixUpData(all_inputs, all_targets)
                    elif ssl_method in ['DFLSemi']:
                        if epoch_iteration_num - iter_id <= 50:
                            self.r_samples += [samples_u.to('cpu')]
                            self.p_labels += [targets_u.to('cpu')]
                        all_inputs, all_targets = [samples_x, samples_u], [labels_x, targets_u]
                        if self.g_samples is not None:
                            cur_ridx = rand_idx[iter_id * bs:iter_id * bs + bs]
                            extra_samples, extra_labels = g_samples[cur_ridx].to(device), g_labels[cur_ridx].to(device)
                            all_inputs = all_inputs + [extra_samples]
                            all_targets = all_targets + [extra_labels]
                        all_inputs, all_targets = torch.cat(all_inputs, dim=0), torch.cat(all_targets, dim=0)
                        mixed_input, mixed_target = MixUpData(all_inputs, all_targets)
                    else:
                        # Pseudo
                        mixed_input = [samples_x] + [samples_u]
                        mixed_input = torch.cat(mixed_input)
                        mixed_target = torch.cat([labels_x, targets_u], dim=0)

                output = self.forward(mixed_input)
                loss = torch.sum(F.log_softmax(output, dim=1) * mixed_target, dim=1)
                loss = -torch.mean(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.history_loss_train.append(float(loss.item()))
        logger.log(INFO, f'Current round final loss: {self.worker_id} {float(loss.item())}')
        if ssl_method != 'Supervise' and total_label_num != 0:
            logger.log(INFO, f"Acc of pseudo labels worker {self.worker_id}: {c_label_num / total_label_num}")
        return

    def TrainUnlabeled(self, epoch: int = 1, round_id: int = 0):
        if ssl_method in ['FlexMatch', 'DFLSemi']:
            self.sigma_c_old = self.sigma_c_new
            self.sigma_c_new = {i: 0 for i in range(-1, classes_n)}
        # load the aggregated model weights
        if round_id != 0:
            self.load_state_dict(self.cur_params)
        device, logger = self.device, self.logger
        ul_iter = iter(self.unlabeled_train_loader) if self.unlabeled_train_loader else None
        pl_iter = None
        if fixed_pl:
            cur_epoch_pl_loader = self.GeneratePlLoader(round_id)
            if cur_epoch_pl_loader is not None:
                pl_iter = iter(cur_epoch_pl_loader)

        c, size = self.c, self.size
        if ssl_method == 'Supervise' or (fixed_pl and pl_iter is None):
            return
        c_label_num, total_label_num = 0, 0
        for epoch_id in range(0, epoch):
            if ssl_method == 'DFLSemi':
                # sample before training at each client
                if (round_id + 1) >= sample_start_round and (round_id + 1) % sample_round_interval == 0:
                    self.g_samples, self.g_labels = self.GenerateSamples(1000, (c, size, size))
                    self.g_test_samples, self.g_test_labels = self.GenerateSamples(100, (c, size, size))
                if round_id != 0:
                    self.TrainDiffusionLoop(None, train_dpm_loop_num, round_id)
            self.r_samples, self.p_labels = [], []

            if self.g_samples is not None:
                g_samples, g_labels = self.g_samples, self.g_labels
                bs, len_r = local_batch_size, len(g_samples)
                rand_idx = torch.randperm(len_r, device=device).to('cpu')
            for iter_id in range(epoch_iteration_num):
                if fixed_pl:
                    pl_iter, samples_u, targets_u, index_u = iter_loader(pl_iter, cur_epoch_pl_loader, iter_id)
                    samples_u, targets_u = samples_u.to(device), targets_u.to(device)
                else:
                    tmp_data = iter_loader(ul_iter, self.unlabeled_train_loader, iter_id)
                    ul_iter, samples_u, labels_u, index_u = tmp_data
                    tmp_data = self.GeneratePlBatch(samples_u, labels_u, index_u, round_id)
                    samples_u, labels_u, index_u, targets_u, cn, tn = tmp_data
                    c_label_num += cn
                    total_label_num += tn
                if ssl_method == 'MixMatch':
                    all_inputs, all_targets = samples_u, targets_u
                    mixed_input, mixed_target = MixUpData(all_inputs, all_targets)
                elif ssl_method in ['DFLSemi']:
                    if epoch_iteration_num - iter_id <= 50:
                        self.r_samples += [samples_u.to('cpu')]
                        self.p_labels += [targets_u.to('cpu')]
                    all_inputs, all_targets = samples_u, targets_u
                    if self.g_samples is not None:
                        cur_ridx = rand_idx[iter_id * bs:iter_id * bs + bs]
                        extra_samples, extra_labels = g_samples[cur_ridx].to(device), g_labels[cur_ridx].to(device)
                        all_inputs = torch.cat([all_inputs, extra_samples])
                        all_targets = torch.cat([all_targets, extra_labels])
                    mixed_input, mixed_target = MixUpData(all_inputs, all_targets)
                else:
                    # Pseudo
                    mixed_input, mixed_target = samples_u, targets_u

                if len(mixed_target) == 0:
                    loss = torch.tensor(0.0)
                else:
                    output = self.forward(mixed_input)
                    loss = torch.sum(F.log_softmax(output, dim=1) * mixed_target, dim=1)
                    loss = -torch.mean(loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        self.history_loss_train.append(float(loss.item()))
        logger.log(INFO, f'Current round final loss: {self.worker_id} {float(loss.item())}')
        if ssl_method != 'Supervise' and total_label_num != 0:
            logger.log(INFO, f"Acc of pseudo labels worker {self.worker_id}: {c_label_num / total_label_num}")
        return

    @torch.no_grad()
    def SaveCKPTs(self, round_id):
        torch.save(self.dpm.state_dict(), str(self.ckpt_path / f'dpm_{self.worker_id}_{round_id + 1}.pth'))
        return

    @torch.no_grad()
    def TestOneBatch(self, round_idx, worker_id=None):
        device, logger = self.device, self.logger
        worker_id = self.worker_id if worker_id is None else worker_id
        batch_size = 512
        samples, labels = self.test_samples[:batch_size], self.test_labels[:batch_size]
        samples = self.test_transform((samples / 255.0).float())
        samples, labels = samples.to(device), labels.to(device)
        output = tc.argmax(self.forward(samples), axis=-1)
        num_correct = len(tc.where(output == labels)[0])
        acc = num_correct / batch_size
        logger.log(INFO, f'ACC on one test batch: {num_correct} / {batch_size} = {acc * 100:.3f}%')
        self.history_acc.append(acc)
        return acc

    @torch.no_grad()
    def TestOnGSamples(self):
        device, logger = self.device, self.logger
        t_samples, t_labels = self.g_test_samples, self.g_test_labels
        length = len(self.g_test_samples)
        batch_size = 512
        iter_num = length // batch_size + 1
        num_correct = 0
        transform = v2.Compose([v2.RandomHorizontalFlip(0.5), v2.RandomCrop(self.size, padding=pad_size)])
        for i in range(iter_num):
            samples, labels = t_samples[i * batch_size:(i + 1) * batch_size], t_labels[
                                                                              i * batch_size:(i + 1) * batch_size]
            samples = transform(samples)
            samples, labels = samples.to(device), labels.to(device)
            output = tc.argmax(self.forward(samples), axis=-1)
            num_correct += len(tc.where(output == labels.argmax(axis=-1))[0])
        acc = num_correct / length
        self.g_acc = acc
        logger.log(INFO, f'Worker {self.worker_id}: '
                         f'ACC on generated samples: {num_correct} / {length} = {acc * 100:.3f}%')
        return acc

    def Test(self):
        device, logger = self.device, self.logger
        test_size = self.size_testingset
        num_correct = 0
        for sample_batch, label_batch, _ in self.test_loader:
            sample_batch, label_batch = sample_batch.to(device), label_batch.to(device)
            output = tc.argmax(self.forward(sample_batch), axis=-1)
            num_correct += len(tc.where(output == label_batch)[0])
        acc = num_correct / test_size
        logger.log(INFO, f'ACC on test set: {num_correct} / {test_size} = {acc * 100:.3f}%')
        self.history_acc.append(acc)
        return acc

    def SetTrainSet(self):
        def _get_data():
            if task == 'mnist':
                dataset_train = datasets.MNIST('./samples/mnist', train=True, download=True)
                data_train = dataset_train.data.reshape(-1, c, size, size)
                label_train = dataset_train.targets
            elif task == 'fashion':
                dataset_train = datasets.FashionMNIST('./samples/fashion', train=True, download=True)
                data_train = dataset_train.data.reshape(-1, c, size, size)
                label_train = dataset_train.targets
            elif task == 'cifar10':
                dataset_train = datasets.CIFAR10('./samples/cifar10', train=True, download=True)
                data_train = dataset_train.data.transpose(0, 3, 1, 2)  # BHWC->BCHW
                data_train = torch.from_numpy(data_train.reshape(-1, c, size, size))
                label_train = torch.tensor(dataset_train.targets)
            return data_train, label_train

        csv_path = Path(f'./DFLSemi_data/dataset_{task}/{dist_wm_id:02d}')
        logger = self.logger
        c, size, mean, std = self.c, self.size, self.mean, self.std
        train_transform = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomCrop(size, padding=pad_size),
            v2.Normalize(mean, std)
        ])
        self.train_transform = train_transform
        all_data_train, all_label_train = _get_data()
        if self.worker_id in l_client_ids:
            labeled_idxes = pd.read_csv(str(csv_path / f'client_{self.worker_id:02d}_{args.label_ratio:.1f}'
                                                       f'_{non_iid_alpha:.2f}_labeled.csv'), header=None).squeeze()
            labeled_samples = all_data_train[labeled_idxes]
            labeled_labels = all_label_train[labeled_idxes]
            labeled_train_data = DatasetLabeled(labeled_samples, labeled_labels, train_transform, True, classes_n)
            self.labeled_train_loader = data.DataLoader(labeled_train_data, batch_size=local_batch_size,
                                                        shuffle=True, num_workers=loader_worker, drop_last=True,
                                                        persistent_workers=True)
            logger.log(INFO, f'{self.worker_id}, length of labeled data loader, {len(self.labeled_train_loader)}')
        elif self.worker_id in m_client_ids:
            labeled_idxes = pd.read_csv(str(csv_path / f'client_{self.worker_id:02d}_{args.label_ratio:.1f}_'
                                                       f'{non_iid_alpha:.2f}_labeled.csv'), header=None).squeeze()
            labeled_samples, labeled_labels = all_data_train[labeled_idxes], all_label_train[labeled_idxes]
            labeled_train_data = DatasetLabeled(labeled_samples, labeled_labels, train_transform, True, classes_n)
            self.labeled_train_loader = data.DataLoader(labeled_train_data, batch_size=local_batch_size,
                                                        shuffle=True, num_workers=loader_worker, drop_last=True,
                                                        persistent_workers=True)
            logger.log(INFO, f'{self.worker_id}, length of labeled data loader, {len(self.labeled_train_loader)}')
            if f'{args.label_ratio:.1f}' == '100.0':
                return
            unlabel_idxes = pd.read_csv(str(csv_path / f'client_{self.worker_id:02d}_{args.label_ratio:.1f}_'
                                                       f'{non_iid_alpha:.2f}_unlabeled.csv'), header=None).squeeze()
            unlabel_samples, unlabel_labels = all_data_train[unlabel_idxes], all_label_train[unlabel_idxes]
            unlabeled_train_data = DatasetUnlabeled(unlabel_samples, unlabel_labels, train_transform, classes_n)
            self.unlabeled_train_loader = data.DataLoader(unlabeled_train_data, batch_size=local_batch_size,
                                                          shuffle=True, num_workers=loader_worker, drop_last=True,
                                                          persistent_workers=True)
            logger.log(INFO, f'{self.worker_id}, length of unlabeled data loader, {len(self.unlabeled_train_loader)}')
        else:
            if f'{args.label_ratio:.1f}' == '100.0':
                return
            unlabel_idxes = pd.read_csv(str(csv_path / f'client_{self.worker_id:02d}_{args.label_ratio:.1f}_'
                                                       f'{non_iid_alpha:.2f}_unlabeled.csv'), header=None).squeeze()
            unlabel_samples = all_data_train[unlabel_idxes]
            unlabel_labels = all_label_train[unlabel_idxes]
            unlabeled_train_data = DatasetUnlabeled(unlabel_samples, unlabel_labels, train_transform, classes_n)
            self.unlabeled_train_loader = data.DataLoader(unlabeled_train_data, batch_size=local_batch_size,
                                                          shuffle=True, num_workers=loader_worker, drop_last=True,
                                                          persistent_workers=True)
            logger.log(INFO, f'{self.worker_id}, length of unlabeled data loader, {len(self.unlabeled_train_loader)}')

    def SetTestSet(self):
        def _get_data():
            if task == 'mnist':
                dataset_test = datasets.MNIST('./samples/mnist', train=False, download=True)
                data_test = dataset_test.data.reshape(-1, c, size, size)
                label_test = dataset_test.targets
            elif task == 'fashion':
                dataset_test = datasets.FashionMNIST('./samples/fashion', train=False, download=True)
                data_test = dataset_test.data.reshape(-1, c, size, size)
                label_test = dataset_test.targets
            elif task == 'cifar10':
                dataset_test = datasets.CIFAR10('./samples/cifar10', train=False, download=True)
                data_test = dataset_test.data.transpose(0, 3, 1, 2)  # BHWC->BCHW
                data_test = torch.from_numpy(data_test.reshape(-1, c, size, size))
                label_test = torch.tensor(dataset_test.targets)
            return data_test, label_test

        c, size, mean, std = self.c, self.size, self.mean, self.std
        transform = v2.Normalize(mean, std)
        samples, labels = _get_data()
        self.test_samples, self.test_labels, self.test_transform = samples, labels, transform
        labeled_test_data = DatasetLabeled(samples, labels, transform, False, classes_n)
        self.test_loader = data.DataLoader(labeled_test_data, batch_size=500,
                                           shuffle=False, num_workers=loader_worker, persistent_workers=True)
        self.size_testingset = len(samples)
        return


def prepare_input(resolution):
    x = torch.FloatTensor(1, 1, 28, 28)
    c = torch.FloatTensor(1, 10)
    t = torch.FloatTensor(1)
    # mask = torch.FloatTensor(1)
    # return dict(x=x, c=c, t=t, context_mask=mask)
    return dict(x=x, c=c, t=t)


def prepare_input_cifar10(resolution):
    x = torch.FloatTensor(1, 3, 32, 32)
    return dict(x=x)


if __name__ == '__main__':
    from thop import profile

    x = torch.FloatTensor(1, 1, 3, 32, 32)
    c = torch.FloatTensor(1, 10)
    t = torch.FloatTensor(1)
    # N = ContextUnet(3, 128, 10, task='cifar10')
    # macs, params = profile(N, inputs=(x, t, c))
    # # print(macs)
    # print(params)

    N = ResNet18(10)
    macs, params = profile(N, inputs=x)
    # print(macs)
    print(params)

    # Classifier = ResNet18(10)
    # macs, params = get_model_complexity_info(Classifier, input_res=(1,), input_constructor=prepare_input_cifar10, as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # # print('Params and FLOPs are {}M and {}G'.format(params/1e6, flops/1e9))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # macs, params = get_model_complexity_info(N, input_res=(1,), input_constructor=prepare_input, as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('Params and FLOPs are {}M and {}G'.format(params/1e6, flops/1e9))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
