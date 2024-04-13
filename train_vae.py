import math
import os

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from config import Config
from data_loader import SitcomPoseDataset
from model import VariationalAutoencoder

device = 'cuda:0'

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def train_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_dataloader, clip):
    ## args
    # - criterion: 학습 손실을 계산하는 데 사용되는 손실 함수
    # - clip: for gradient clipping, prevent explosion

    losses = []
    model.train()

    for i, batch in enumerate(train_dataloader):

        ## -- forward step
        img, img_crop, img_zoom, one_hot_pose_vector, scale_deformation = batch

        # make model output - vae
        scale_deformation_recon, latent_mu, latent_logvar = model(scale_deformation.to(device), one_hot_pose_vector.to(device), img.to(device), img_crop.to(device), img_zoom.to(device))

        # compute loss using loss function
        loss, _ = criterion(scale_deformation_recon.to(device), scale_deformation.to(device), latent_mu.to(device), latent_logvar.to(device))

        loss_val = loss.item()
        losses.append(loss_val)

        wandb.log({"Train Loss": math.log10(loss_val)})

        ## -- optimization step
        optimizer.zero_grad()
        # gradient 계산
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # optimizer - update model parameter
        optimizer.step()
        # update the learning rate
        # lr_scheduler.step()

    return np.mean(losses)

def eval_epoch(model, dataloader, criterion):
    # - criterion: 검증 손실을 계산하는 데 사용되는 손실 함수
    model.eval()
    val_losses = []
    mse_losses = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            
            ## -- forward step
            img, img_crop, img_zoom, one_hot_pose_vector, scale_deformation = batch

            # make model output - vae
            scale_deformation_recon, latent_mu, latent_logvar = model(scale_deformation.to(device), one_hot_pose_vector.to(device), img.to(device), img_crop.to(device), img_zoom.to(device))

            # compute loss using loss function
            loss, mse_loss = criterion(scale_deformation_recon.to(device), scale_deformation.to(device), latent_mu.to(device), latent_logvar.to(device))

            loss_val = loss.item()
            val_losses.append(loss_val)
            mse_losses.append(mse_loss.item())
    
    return np.mean(val_losses), np.mean(mse_losses)

def train():

    cfg = Config()
    model = VariationalAutoencoder(cfg).to(device)

    torch.autograd.set_detect_anomaly(True)

    def vae_loss(recon_x, x, mu, logvar):

        recon_loss = F.mse_loss(recon_x.view(-1, cfg.sd_dim), x.view(-1, cfg.sd_dim), reduction='mean')
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + cfg.variational_beta * kldivergence, recon_loss

    criterion = vae_loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)
    num_epochs = cfg.num_epochs

    data_path = './affordance_data'
    data_list = []

    with open(os.path.join(data_path, 'trainlist.txt'), 'r') as f:
        data_list = list(f.readlines())
    train_dataset = SitcomPoseDataset(data_path, data_list)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    with open(os.path.join(data_path, 'testlist.txt'), 'r') as f:
        data_list = list(f.readlines())
    test_dataset = SitcomPoseDataset(data_path, data_list)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    train_loss_avg = []
    print("Training ...")

    for epoch in range(num_epochs):

        current_lr = get_lr(optimizer)
        print(f'Epoch {epoch+1}/{num_epochs}, current lr={current_lr}')

        train_loss = train_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, cfg.CLIP)
        train_loss_avg.append(train_loss)
        lr_scheduler.step()

        # validation step
        if num_epochs % cfg.validation_term == 0 and num_epochs != 0:
            valid_loss, mse_loss = eval_epoch(model, test_loader, criterion)
            wandb.log({"Validation Loss": valid_loss, "MSE Loss": mse_loss})
            print(f'Validation Loss: {valid_loss:.3f}, MSE Loss: {mse_loss:.3f}')

            # validation 후 - model save
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f'model_{epoch}_{int(valid_loss)}_{int(mse_loss)}.pt'))

if __name__ == '__main__':
    wandb.init(
        project="AIKU-VAE"
    )
    train()