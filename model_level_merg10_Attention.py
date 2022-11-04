"""This cell is train_cifar0.py
This file is using CIFAR100, 
Resnet18 or shufflenet, which can be changed in the load model section
"""
#%% load dependency
import os
import numpy as np
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
from datetime import datetime

from modules import *
from utils import *
from torch.utils.data import Dataset, DataLoader

print('starting date time ', datetime.now())

#%%
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)

#%%
id0 = 'resnet18_mnist'

t0 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
t0.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # original
t0.load_state_dict(torch.load(f'teachers/{id0}.pt'))

t1 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
t1.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # original
# t1.load_state_dict(torch.load(f'teachers/{id0}.pt'))

t2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
t2.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # original
# t2.load_state_dict(torch.load(f'teachers/{id0}.pt'))

t0.eval()
for param0 in t0.parameters():
    param0.requires_grad = False

t1.eval()
for param1 in t1.parameters():
    param1.requires_grad = False

t2.eval()
for param1 in t2.parameters():
    param1.requires_grad = False

t0 = t0.cuda()
t1 = t1.cuda()
t2 = t2.cuda()

#%%
oc_sd=0.01
loga = (torch.randn(3)*oc_sd).cuda().requires_grad_()
print('initial loga', loga)
def hard_concrete(loga, batch_size=128):
    beta, gamma, zeta, eps = 2/3, -0.1, 1.1, 1e-20
    u = torch.rand(batch_size, loga.shape[0], device=loga.device)
    s = torch.sigmoid((torch.log(u+eps) - torch.log(1 - u+eps) + loga) / beta)
    sbar = s * (zeta - gamma) + gamma
    z = hard_sigmoid(sbar)
    return z

#%%
best_validation_accuracy = 0. # used to pick the best-performing model on the validation set
train_accs = []
val_accs = []

opt = {'epochs':15}
optimizer = torch.optim.RAdam([loga],
                lr= 0.001,
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
loss_func = nn.CrossEntropyLoss()

loss_tr, loss_val = [], []
lamb = 1
for epoch in range(opt['epochs']):
    # print training info
    print(f"### Epoch {epoch}:")
    total_train_examples = 0
    num_correct_train = 0

    # for batch_index, (inputs, gt_label) in tqdm(enumerate(train_loader), total=len(train_dataset)//train_batchsize):
    for batch_index, (inputs, gt_label) in enumerate(train_loader):
        inputs = inputs.cuda()
        gt_label = gt_label.cuda()
        optimizer.zero_grad()
        z = hard_concrete(loga, batch_size=gt_label.shape[0])
        pred0 = t0(inputs)*z[:,0:1]
        pred1 = t1(inputs)*z[:,1:2]
        pred2 = t2(inputs)*z[:,2:]
        l0 = loss_func(pred0, gt_label)
        l1 = loss_func(pred1, gt_label)
        l2 = loss_func(pred1, gt_label)
        loss = l0 + l1 + l2 + lamb*(z.mean() -1/3)
        loss.backward()
        loss_tr.append(loss.cpu().detach().item())
        optimizer.step()
        
        g_ind = z.argmax(dim=1)
        pred = torch.stack((pred0, pred1, pred2), dim=1)
        predictions = pred[range(z.shape[0]),g_ind]
        _, predicted_class = predictions.max(1)
        total_train_examples += predicted_class.size(0)
        num_correct_train += predicted_class.eq(gt_label).sum().item()

    # get results
    train_acc = num_correct_train / total_train_examples
    print("Training accuracy: {}".format(train_acc))
    train_accs.append(train_acc)

    # evaluation
    total_val_examples = 0
    num_correct_val = 0

    with torch.no_grad(): # don't save parameter gradients/changes since this is not for model training
        for batch_index, (inputs, gt_label) in enumerate(test_loader):
            inputs = inputs.cuda()
            gt_label = gt_label.cuda()
            z = hard_concrete(loga, batch_size=gt_label.shape[0])
            pred0 = t0(inputs)*z[:,0:1]
            pred1 = t1(inputs)*z[:,1:2]
            pred2 = t2(inputs)*z[:,2:]
            l0 = loss_func(pred0, gt_label)
            l1 = loss_func(pred1, gt_label)
            l2 = loss_func(pred1, gt_label)

            g_ind = z.argmax(dim=1)
            pred = torch.stack((pred0, pred1, pred2), dim=1)
            predictions = pred[range(z.shape[0]),g_ind]
            _, predicted_class = predictions.max(1)
            total_val_examples += predicted_class.size(0)
            num_correct_val += predicted_class.eq(gt_label).sum().item()

        # get validation results
        val_acc = num_correct_val / total_val_examples
        print("Validation accuracy: {}".format(val_acc))
        val_accs.append(val_acc)
        print('after validation', loga)

    # Finally, save model if the validation accuracy is the best so far
    if val_acc > best_validation_accuracy:
        best_validation_accuracy = val_acc
        print("Validation accuracy improved; saving model.")
        print(loga)
        

#%% plot train and val results
epochs_list = list(range(opt['epochs']))
plt.figure()
plt.plot(epochs_list, train_accs, '--x', label='training set accuracy')
plt.plot(epochs_list, val_accs, '-.v', label='validation set accuracy')
plt.xlabel('epoch')
plt.ylabel('prediction accuracy')
# plt.ylim(0.5, 1)
plt.title('Classifier training evolution:\nprediction accuracy over time')
plt.legend()
plt.savefig('train_val.png')
# plt.savefig(f'train_val{id}.png')
plt.show()

print('done')
print('End date time ', datetime.now())