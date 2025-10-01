import torch
import torch.nn as nn
from torch.nn import functional as F
from engression.utils import vectorize
from engression.loss_func import energy_loss_two_sample

def train(model, optimizer, dataloader, epochs=50, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss, total_loss1, total_loss2 = 0, 0, 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            xhat_0, xhat_1 = model(data, double=True)
            loss, loss1, loss2 = energy_loss_two_sample(data, xhat_0, xhat_1, verbose=True)
            total_loss += loss.item() / len(dataloader)
            total_loss1 += loss1.item() / len(dataloader)
            total_loss2 += loss2.item() / len(dataloader)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f} , Loss1: {:.6f}, Loss2: {:.6f}'.format(
                epoch , epochs,
                100. * epoch / epochs, total_loss, total_loss1, total_loss2))
    print('done!')

def sto_train(model, optimizer, dataloader, epochs=50, device='cpu'):
    model.train()
    for epoch in range(epochs):
        total_loss, total_loss1, total_loss2 = 0, 0, 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            data = vectorize(data)
            target = vectorize(target)
            xhat_0, xhat_1 = model(data), model(data)
            loss, loss1, loss2 = energy_loss_two_sample(target, xhat_0, xhat_1, verbose=True)
            total_loss += loss.item() / len(dataloader)
            total_loss1 += loss1.item() / len(dataloader)
            total_loss2 += loss2.item() / len(dataloader)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f} , Loss1: {:.6f}, Loss2: {:.6f}'.format(
                epoch , epochs,
                100. * epoch / epochs, total_loss, total_loss1, total_loss2))
    print('done!')

def erm_train(model, optimizer, dataloader, epochs=50, device='cpu'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            xhat = model(data)
            loss = F.mse_loss(xhat, target.unsqueeze(1))
            total_loss += loss.item() / len(dataloader)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch , epochs,
                100. * epoch / epochs, total_loss))
    print('done!')
