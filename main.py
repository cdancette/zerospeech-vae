# This example is taken from https://github.com/pytorch/examples/blob/master/vae/main.py
from __future__ import print_function
import argparse
import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
#from torchvision.utils import save_image

import h5features
import h5py
import random
import numpy as np

from model import VAE
from data import get_train_test_datasets
from pathlib import Path

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 40), reduction='sum')

    # MSE norm
    MSE = F.mse_loss(recon_x, x.view(-1, 40))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data, in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__=='__main__':

	parser = argparse.ArgumentParser(description='VAE MNIST Example')

	parser.add_argument('-f', '--features-path', required=True)
	parser.add_argument('-s', '--embedding-size', required=True, type=int)
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
	                    help='input batch size for training (default: 128)')
	parser.add_argument('-m', '--model-path', help="path to save model", required=True)
	parser.add_argument('--epochs', type=int, default=50, metavar='N',
	                    help='number of epochs to train (default: 10)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='enables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
	                    help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
	                    help='how many batches to wait before logging training status')
	parser.add_argument('--learning-rate', type=float, default=1e-4,
	                    help='learning rate')


	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	model_path = Path(args.model_path)
	model_path.mkdir(exist_ok=True, parents=True)

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if args.cuda else "cpu")

	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

	train_dataset, test_dataset = get_train_test_datasets(args.features_path, 0.7)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

	model = VAE(input_size=40, num_components=args.embedding_size).to(device)
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	for epoch in range(1, args.epochs + 1):
		train(epoch)
		test(epoch)
		torch.save(model.state_dict(), model_path / ("model-%s.pt" % epoch))
		#with torch.no_grad():
		#    sample = torch.randn(64, args.embedding_size).to(device)
		#    sample = model.decode(sample).cpu()
