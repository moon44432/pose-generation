import os

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import wandb

from config import Config
from data_loader import SitcomPoseDataset
from model import PoseClassifier

device = 'cuda:1'

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
        img, img_crop, img_zoom, one_hot_pose_vector, _, _, _, _ = batch

        # make model output - pose classification
        output = model(img.to(device), img_crop.to(device), img_zoom.to(device))

        # compute loss using loss function
        loss = criterion(output, one_hot_pose_vector.to(device))

        loss_val = loss.item()
        losses.append(loss_val)

        wandb.log({"Train Loss": loss_val})

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

    gt_labels = []
    pred_probs = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            
            img, img_crop, img_zoom, one_hot_pose_vector, _, _, _, _ = batch

            # make model output - pose classification
            output = model(img.to(device), img_crop.to(device), img_zoom.to(device))

            # compute loss using loss function
            loss = criterion(output, one_hot_pose_vector.to(device))

            loss_val = loss.item()
            val_losses.append(loss_val)

            if len(gt_labels) == 0:
                gt_labels.append(one_hot_pose_vector)
            else:
                gt_labels[0] = torch.cat((gt_labels[0], one_hot_pose_vector), dim=0)

            if len(pred_probs) == 0:
                pred_probs.append(output)
            else:
                pred_probs[0] = torch.cat((pred_probs[0], output), dim=0)

    top_accuracy = [0]
    for i in range(1, 6, 1):
        top_accuracy.append(top_k_accuracy(gt_labels[0].to(device), pred_probs[0].to(device), i))

    return np.mean(val_losses), top_accuracy

def top_k_accuracy(gt_labels, predicted_probabilities, k):
    gt_classes = torch.argmax(gt_labels, dim=1)
    _, top_k_predictions = predicted_probabilities.topk(k, dim=1)  # Get top-k predictions
    correct_predictions = top_k_predictions.eq(gt_classes.view(-1, 1).expand_as(top_k_predictions))  # Check if true label is within top-k predictions
    top_k_acc = correct_predictions.float().sum(1).mean().item()  # Calculate top-k accuracy
    return top_k_acc

def train():
    cfg = Config()
    model = PoseClassifier(cfg).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.learning_rate, betas=(cfg.adam_beta2, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
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

        # lr_scheduler.step()

        # validation step
        if num_epochs % cfg.validation_term == 0 and num_epochs != 0:
            valid_loss, top_accuracy = eval_epoch(model, test_loader, criterion)
            wandb.log({"Validation Loss": valid_loss})
            for i in range(1, 6, 1):
                wandb.log({f"Top-{i} Accuracy": top_accuracy[i]})
            print(f'Validation Loss: {valid_loss:.3f}, Top Accuracy: {top_accuracy[1:]}')

            # validation 후 - model save
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f'model_{epoch}_{int(valid_loss)}_{top_accuracy[1:]}.pt'))

if __name__ == '__main__':
    wandb.init(
        project="AIKU-classifier"
    )
    train()