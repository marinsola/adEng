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
        for batch_idx, x_and_y in enumerate(dataloader):
            data, target = x_and_y[:, :-1].to(device), x_and_y[:, -1].to(device)
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

def check_test(model, test_loader, device, returns=False):
    model.eval()
    tss, rss = 0, 0
    for i, data in enumerate(test_loader):
        x_test, y_test = data[:, :-1], data[:, -1]
        x_test = x_test.to(device)
        try:
            yhat = model.predict(x_test).cpu().detach()
        except:
            yhat = model(x_test).cpu().detach()
        err = F.mse_loss(y_test.cpu(), yhat.cpu()).item()
        tss += F.mse_loss(y_test.cpu(), y_test.mean().cpu() * torch.ones_like(y_test).cpu()).item()
        rss += err
    if returns:
        return 1 - rss/tss, rss/len(test_loader)
    else:
        print(f'R^2 : {1 - rss/tss}')
        print(f'MSE : {rss/len(test_loader)}')

def estimate_metrics(model, test_loader, device, returns=True, n_iter=100, R2=True, MSE=True, fullreturn=False):
    r2_est, mse_est = [], []
    for i in range(n_iter):
        r2, mse = check_test(model, test_loader, device, returns=returns)
        r2_est.append(r2)
        mse_est.append(mse)
    if fullreturn:
        return r2_est, mse_est
    else:
        return np.mean(r2_est), np.mean(mse_est)
