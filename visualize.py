"""Visualization of SDAE features and reconstructions."""
import os
import argparse
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import StackDAE
import utils


def visualize(data_size=10000, model_path='model/chekp.pt', output_dir='result_SDAE'):
    """Plot 3D feature space and reconstructed results.

    Args:
        data_size (int): number of data points to visualize
        model_path (str): path to trained model checkpoint
        output_dir (str): directory to save result images
    """
    os.makedirs(output_dir, exist_ok=True)
    batchSize = 100
    device = utils.select_device()
    dataset = utils.StateData(data_size=data_size)
    visu_loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    reconstruct_dim = checkpoint['in_dim']
    feature_dim = checkpoint['out_dim']
    stack_num = checkpoint['stack_num']

    model = StackDAE(reconstruct_dim, feature_dim, stack_num)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    reconstruct_stack = []
    feature_stack = []
    with torch.no_grad():
        for data in visu_loader:
            data = data.to(device)
            reconstruction, features = model(data)
            reconstruct_stack.append(reconstruction.cpu())
            feature_stack.append(features.cpu())

    reconstruct_stack = torch.cat(reconstruct_stack).numpy()
    feature_stack = torch.cat(feature_stack).numpy()

    print('%d data points, input size: %d, feature size: %d' %
          (feature_stack.shape[0], reconstruct_dim, feature_stack.shape[1]))

    fig1 = plt.figure()
    ax = Axes3D(fig1)
    ax.scatter(feature_stack[:, 0], feature_stack[:, 1], feature_stack[:, 2])
    fig1.savefig(os.path.join(output_dir, 'result_feature.png'))
    plt.close(fig1)

    fig2 = plt.figure()
    ax = Axes3D(fig2)
    ax.scatter(reconstruct_stack[:, -8], reconstruct_stack[:, -7], reconstruct_stack[:, -6])
    fig2.savefig(os.path.join(output_dir, 'result_reconstruct.png'))
    plt.close(fig2)

    return feature_stack


def visualize_orig(data_size=10000, output_dir='result_SDAE'):
    """Plot positions of the last joint from original observation data.

    Args:
        data_size (int): number of data points
        output_dir (str): directory to save result images
    """
    os.makedirs(output_dir, exist_ok=True)
    batchSize = 100
    dataset = utils.StateData(data_size)
    visu_loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)

    output_stack = []
    for data in visu_loader:
        output_stack.append(data)

    output_stack = torch.cat(output_stack).numpy()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(output_stack[:, -8], output_stack[:, -7], output_stack[:, -6])
    fig.savefig(os.path.join(output_dir, 'result_orig.png'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, default=10000,
                        help='size of data used for visualization')
    parser.add_argument('--model_path', type=str, default='model/chekp.pt',
                        help='path to trained model')
    parser.add_argument('--output_dir', type=str, default='result_SDAE',
                        help='output directory for plots')
    opt = parser.parse_args()
    visualize(opt.data_size, opt.model_path, opt.output_dir)
