import argparse
import random
import torch
import numpy as np
from time import time
import logging
import wandb

from torch.utils.data import DataLoader
from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str, default="./data/ToolBench/toolgen-mean-embeddings-output-sentences-llama2.npy",
                        help="Input data path.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64], help='hidden sizes of every layer')
    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")

    return parser.parse_args()

def train_and_evaluate(config=None):
    """Fix random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize wandb
    wandb.init(config=config)
    config = wandb.config

    logging.basicConfig(level=logging.DEBUG)

    """Build dataset and model"""
    data = EmbDataset(config.data_path)
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=config.num_emb_list,
        e_dim=config.e_dim,
        layers=config.layers,
        dropout_prob=config.dropout_prob,
        bn=config.bn,
        loss_type=config.loss_type,
        quant_loss_weight=config.quant_loss_weight,
        beta=config.beta,
        kmeans_init=config.kmeans_init,
        kmeans_iters=config.kmeans_iters,
        sk_epsilons=config.sk_epsilons,
        sk_iters=config.sk_iters,
    )
    data_loader = DataLoader(
        data, num_workers=config.num_workers,
        batch_size=config.batch_size, shuffle=True,
        pin_memory=True
    )
    trainer = Trainer(config, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    # Log metrics to wandb
    wandb.log({"Best Loss": best_loss, "Best Collision Rate": best_collision_rate})
    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)


if __name__ == '__main__':
    args = parse_args()
    wandb.init(mode="offline")
    sweep_config = {
        'method': 'random',  # Random search
        'metric': {'name': 'Best Loss', 'goal': 'minimize'},
        'parameters': {
            'lr': {'values': [1e-3, 5e-4, 1e-4]},
            'batch_size': {'values': [2048, 5096]},
            'dropout_prob': {'values': [0.0, 0.1]},
            'num_emb_list': {
            'values': [
                [256, 256, 256],
                [512, 256, 256],
                [256, 512, 256],
                [256, 512, 512],
                [256, 256, 512],
                [512, 512, 512],  # baseline larger setup
                [512, 1024, 512], # middle layer larger
                [1024, 512, 512], # first layer larger
                [1024, 1024, 512],# first two layers larger
                [512, 1024, 1024],# last two layers larger
                [1024, 512, 1024],# alternate pattern
                [1024, 1024, 1024] # all layers larger
            ]
        },
            'e_dim': {'values': [32, 64]},
            'quant_loss_weight': {'values': [0.5, 1.0, 2.0]},
            'beta': {'values': [0.1, 0.25, 0.5]},
            'sk_epsilons': {'values': [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [0.1, 0.1, 0.1]]},
            'kmeans_init': {'values': [True, False]},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="RQVAE-Hyperparameter-Tuning")
    wandb.agent(sweep_id, function=train_and_evaluate)