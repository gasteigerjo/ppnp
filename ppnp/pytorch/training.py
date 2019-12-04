from typing import Type, Tuple
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from ..data.sparsegraph import SparseGraph
from ..preprocessing import gen_seeds, gen_splits, normalize_attributes
from .earlystopping import EarlyStopping, stopping_args
from .utils import matrix_to_torch


def get_dataloaders(idx, labels_np, batch_size=None):
    labels = torch.LongTensor(labels_np)
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    datasets = {phase: TensorDataset(ind, labels[ind]) for phase, ind in idx.items()}
    dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                   for phase, dataset in datasets.items()}
    return dataloaders


def train_model(
        name: str, model_class: Type[nn.Module], graph: SparseGraph, model_args: dict,
        learning_rate: float, reg_lambda: float,
        idx_split_args: dict = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114},
        stopping_args: dict = stopping_args,
        test: bool = False, device: str = 'cuda',
        torch_seed: int = None, print_interval: int = 10) -> Tuple[nn.Module, dict]:
    labels_all = graph.labels
    idx_np = {}
    idx_np['train'], idx_np['stopping'], idx_np['valtest'] = gen_splits(
            labels_all, idx_split_args, test=test)
    idx_all = {key: torch.LongTensor(val) for key, val in idx_np.items()}

    logging.log(21, f"{model_class.__name__}: {model_args}")
    if torch_seed is None:
        torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed)
    logging.log(22, f"PyTorch seed: {torch_seed}")

    nfeatures = graph.attr_matrix.shape[1]
    nclasses = max(labels_all) + 1
    model = model_class(nfeatures, nclasses, **model_args).to(device)

    reg_lambda = torch.tensor(reg_lambda, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataloaders = get_dataloaders(idx_all, labels_all)
    early_stopping = EarlyStopping(model, **stopping_args)
    attr_mat_norm_np = normalize_attributes(graph.attr_matrix)
    attr_mat_norm = matrix_to_torch(attr_mat_norm_np).to(device)

    epoch_stats = {'train': {}, 'stopping': {}}

    start_time = time.time()
    last_time = start_time
    for epoch in range(early_stopping.max_epochs):
        for phase in epoch_stats.keys():

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0
            running_corrects = 0

            for idx, labels in dataloaders[phase]:
                idx = idx.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    log_preds = model(attr_mat_norm, idx)
                    preds = torch.argmax(log_preds, dim=1)

                    # Calculate loss
                    cross_entropy_mean = F.nll_loss(log_preds, labels)
                    l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
                    loss = cross_entropy_mean + reg_lambda / 2 * l2_reg

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Collect statistics
                    running_loss += loss.item() * idx.size(0)
                    running_corrects += torch.sum(preds == labels)

            # Collect statistics
            epoch_stats[phase]['loss'] = running_loss / len(dataloaders[phase].dataset)
            epoch_stats[phase]['acc'] = running_corrects.item() / len(dataloaders[phase].dataset)

        if epoch % print_interval == 0:
            duration = time.time() - last_time
            last_time = time.time()
            logging.info(f"Epoch {epoch}: "
                         f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                         f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                         f"early stopping loss = {epoch_stats['stopping']['loss']:.2f}, "
                         f"early stopping acc = {epoch_stats['stopping']['acc'] * 100:.1f} "
                         f"({duration:.3f} sec)")

        if len(early_stopping.stop_vars) > 0:
            stop_vars = [epoch_stats['stopping'][key]
                         for key in early_stopping.stop_vars]
            if early_stopping.check(stop_vars, epoch):
                break
    runtime = time.time() - start_time
    runtime_perepoch = runtime / (epoch + 1)
    logging.log(22, f"Last epoch: {epoch}, best epoch: {early_stopping.best_epoch} ({runtime:.3f} sec)")

    # Load best model weights
    model.load_state_dict(early_stopping.best_state)

    train_preds = get_predictions(model, attr_mat_norm, idx_all['train'])
    train_acc = (train_preds == labels_all[idx_all['train']]).mean()

    stopping_preds = get_predictions(model, attr_mat_norm, idx_all['stopping'])
    stopping_acc = (stopping_preds == labels_all[idx_all['stopping']]).mean()
    logging.log(21, f"Early stopping accuracy: {stopping_acc * 100:.1f}%")

    valtest_preds = get_predictions(model, attr_mat_norm, idx_all['valtest'])
    valtest_acc = (valtest_preds == labels_all[idx_all['valtest']]).mean()
    valtest_name = 'Test' if test else 'Validation'
    logging.log(22, f"{valtest_name} accuracy: {valtest_acc * 100:.1f}%")
    
    result = {}
    result['predictions'] = get_predictions(model, attr_mat_norm, torch.arange(len(labels_all)))
    result['train'] = {'accuracy': train_acc}
    result['early_stopping'] = {'accuracy': stopping_acc}
    result['valtest'] = {'accuracy': valtest_acc}
    result['runtime'] = runtime
    result['runtime_perepoch'] = runtime_perepoch

    return model, result


def get_predictions(model, attr_matrix, idx, batch_size=None):
    if batch_size is None:
        batch_size = idx.numel()
    dataset = TensorDataset(idx)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    preds = []
    for idx, in dataloader:
        idx = idx.to(attr_matrix.device)
        with torch.set_grad_enabled(False):
            log_preds = model(attr_matrix, idx)
            preds.append(torch.argmax(log_preds, dim=1))
    return torch.cat(preds, dim=0).cpu().numpy()
