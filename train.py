"""Training script for Stacked Denoising Autoencoder."""
import os
import copy
import argparse
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import DAE, StackDAE
from utils import StateData, select_device, clean_output, split_dataset
from noise import get_noise_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def train_dae_layer(model, train_loader, val_loader, device, noise_fn, opt, layer):
    """Train a single DAE layer with validation, early stopping, and logging.

    Args:
        model: DAE model
        train_loader: training data loader
        val_loader: validation data loader (can be None)
        device: torch device
        noise_fn: callable noise function
        opt: parsed arguments
        layer: current layer number (1-indexed)

    Returns:
        Tensor: last training output
    """
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Learning rate scheduler with optional warmup
    if opt.warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < opt.warmup_epochs:
                return (epoch + 1) / opt.warmup_epochs
            return 0.2 ** ((epoch - opt.warmup_epochs) // 6)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)

    # Early stopping state
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    output = None

    for ep in range(opt.epoch):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_batches = 0

        for i, data in enumerate(train_loader):
            data = data.to(device)
            noise_data = noise_fn(data, float(opt.noise_r) / 100)
            noise_data = noise_data.to(device)

            optimizer.zero_grad()
            output = model(noise_data)
            error = loss_fn(output, data)
            error.backward()
            optimizer.step()

            train_loss += error.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        scheduler.step()

        # --- Validation ---
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    noise_data = noise_fn(data, float(opt.noise_r) / 100)
                    noise_data = noise_data.to(device)
                    output = model(noise_data)
                    error = loss_fn(output, data)
                    val_loss += error.item()
                    val_batches += 1
            avg_val_loss = val_loss / max(val_batches, 1)

        # --- Logging ---
        lr = optimizer.param_groups[0]['lr']
        if avg_val_loss is not None:
            logger.info(
                'Layer %d | Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f | LR: %.6f',
                layer, ep + 1, opt.epoch, avg_train_loss, avg_val_loss, lr)
        else:
            logger.info(
                'Layer %d | Epoch %d/%d | Train Loss: %.6f | LR: %.6f',
                layer, ep + 1, opt.epoch, avg_train_loss, lr)

        # --- TensorBoard ---
        if opt.writer is not None:
            global_step = ep + (layer - 1) * opt.epoch
            opt.writer.add_scalar(f'layer_{layer}/train_loss', avg_train_loss, ep)
            opt.writer.add_scalar(f'layer_{layer}/learning_rate', lr, ep)
            if avg_val_loss is not None:
                opt.writer.add_scalar(f'layer_{layer}/val_loss', avg_val_loss, ep)

        # --- Early stopping ---
        if opt.patience > 0 and avg_val_loss is not None:
            if avg_val_loss < best_val_loss - opt.min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= opt.patience:
                    logger.info('Layer %d | Early stopping at epoch %d (patience=%d)',
                                layer, ep + 1, opt.patience)
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

    return output


def train(opt):
    """Train a Stacked DAE with layer-wise pretraining.

    Args:
        opt: parsed command-line arguments
    """
    device = select_device()
    stack_num = opt.stack_num
    in_dim = opt.in_dim
    end_dim = opt.out_dim

    final_net = StackDAE(in_dim, end_dim, stack_num)

    # Use same dimension list as StackDAE to avoid float drift
    dims = StackDAE._compute_dims(in_dim, end_dim, stack_num)

    # Get noise function
    noise_fn = get_noise_fn(opt.noise_type)
    logger.info('Using noise type: %s (ratio: %d%%)', opt.noise_type, opt.noise_r)

    # Set up TensorBoard writer
    opt.writer = None
    if opt.tensorboard:
        log_dir = opt.log_dir or 'runs/sdae_training'
        opt.writer = SummaryWriter(log_dir=log_dir)
        logger.info('TensorBoard logging to: %s', log_dir)

    # Load full dataset
    full_dataset = StateData(data_size=opt.data_size)

    # Train/val split
    if opt.val_ratio > 0:
        train_dataset, val_dataset = split_dataset(full_dataset, opt.val_ratio)
        logger.info('Dataset: %d train, %d validation', len(train_dataset), len(val_dataset))
    else:
        train_dataset = full_dataset
        val_dataset = None
        logger.info('Dataset: %d train (no validation)', len(train_dataset))

    train_loader_raw = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize, shuffle=True)
    val_loader_raw = None
    if val_dataset is not None:
        val_loader_raw = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batchSize, shuffle=False)

    # Layer-wise pretraining
    stacked_enc_net = nn.Sequential()
    stacked_dec_net = nn.Sequential()
    train_loader = train_loader_raw
    val_loader = val_loader_raw

    for i in range(stack_num):
        cur_in_dim = dims[i]
        cur_out_dim = dims[i + 1]
        logger.info('--- Training layer %d/%d (dim: %d -> %d) ---',
                     i + 1, stack_num, cur_in_dim, cur_out_dim)

        model = DAE(cur_in_dim, cur_out_dim).to(device)

        train_dae_layer(model, train_loader, val_loader,
                        device, noise_fn, opt, layer=i + 1)

        stacked_enc_net.add_module("encoder_%d" % i, model.encoder)
        stacked_dec_net.add_module("decoder_%d" % i, model.decoder)

        # Generate clean outputs for next layer
        train_enc_dataset = clean_output(train_loader_raw, stacked_enc_net, device)
        train_loader = torch.utils.data.DataLoader(
            train_enc_dataset, batch_size=opt.batchSize, shuffle=True)

        val_loader = None
        if val_loader_raw is not None:
            val_enc_dataset = clean_output(val_loader_raw, stacked_enc_net, device)
            val_loader = torch.utils.data.DataLoader(
                val_enc_dataset, batch_size=opt.batchSize, shuffle=False)

    # Assemble final stacked model
    stacked_enc_dict = stacked_enc_net.state_dict()
    stacked_dec_dict = stacked_dec_net.state_dict()
    new_state_dict = OrderedDict()

    for k, v in stacked_enc_dict.items():
        new_state_dict['stack_enc.' + k] = v
    for k, v in stacked_dec_dict.items():
        new_state_dict['stack_dec.' + k] = v

    final_net.load_state_dict(new_state_dict)

    # Evaluate final reconstruction loss
    final_net.to(device).eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        loader = val_loader_raw if val_loader_raw is not None else train_loader_raw
        for data in loader:
            data = data.to(device)
            reconstruction, _ = final_net(data)
            total_loss += loss_fn(reconstruction, data).item()
            total_batches += 1
    final_loss = total_loss / max(total_batches, 1)
    logger.info('Final reconstruction loss: %.6f', final_loss)

    if opt.writer is not None:
        opt.writer.add_scalar('final/reconstruction_loss', final_loss, 0)

    # Save model
    os.makedirs('model', exist_ok=True)
    checkpoint = {
        'in_dim': opt.in_dim,
        'out_dim': opt.out_dim,
        'stack_num': opt.stack_num,
        'noise_type': opt.noise_type,
        'noise_r': opt.noise_r,
        'model': final_net.state_dict(),
        'final_loss': final_loss,
    }
    save_path = opt.save_path or 'model/chekp.pt'
    torch.save(checkpoint, save_path)
    logger.info('Model saved to %s', save_path)

    if opt.writer is not None:
        opt.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stacked Denoising Autoencoder')

    # Data
    parser.add_argument('--data_size', type=int, default=200000, help='size of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='validation split ratio (0 to disable)')

    # Model
    parser.add_argument('--in_dim', type=int, default=48, help='input dimension')
    parser.add_argument('--out_dim', type=int, default=3, help='output (feature) dimension')
    parser.add_argument('--stack_num', type=int, default=4, help='number of encoder/decoder layers')

    # Training
    parser.add_argument('--epoch', type=int, default=15, help='training epochs per layer')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--batchSize', type=int, default=640, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='data loading workers')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='linear warmup epochs (0 to disable)')

    # Noise
    parser.add_argument('--noise_type', type=str, default='salt_and_pepper',
                        choices=['salt_and_pepper', 'gaussian', 'masking'],
                        help='noise type for denoising')
    parser.add_argument('--noise_r', type=int, default=10, help='noise ratio (percent)')

    # Early stopping
    parser.add_argument('--patience', type=int, default=0,
                        help='early stopping patience (0 to disable)')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                        help='minimum improvement for early stopping')

    # Logging
    parser.add_argument('--tensorboard', action='store_true',
                        help='enable TensorBoard logging')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='TensorBoard log directory')

    # Output
    parser.add_argument('--save_path', type=str, default=None,
                        help='model save path (default: model/chekp.pt)')

    opt = parser.parse_args()
    train(opt)
