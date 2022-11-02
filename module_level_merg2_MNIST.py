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
batch_size = 256
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

#%%
id0 = 'resnet18_mnist'
def get_random_model(seed, load=False):
    torch.manual_seed(seed)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # original
    if load : model.load_state_dict(torch.load(f'teachers/{id0}.pt'))
    for param in model.parameters():
        param.requires_grad = False
    return model
t0 = get_random_model(0, load=True) # t0 is trained, t1-t3 are not trained
for i in range(1,4):
    exec(f"t{i} = get_random_model({i})")

"swap t1 1st half, t2 second half with trained model"
l = 60 #len(list(t0.parameters()))//2
temp_dict = t1.state_dict().copy()
for i, k in enumerate(t0.state_dict().keys()):
    if i < 60:
        if k in temp_dict.keys():
            temp_dict[k] = t0.state_dict()[k]
t1.load_state_dict(temp_dict, strict=False)

temp_dict = t2.state_dict().copy()
for i, k in enumerate(t0.state_dict().keys()):
    if i >= 60:
        if k in temp_dict.keys():
            temp_dict[k] = t0.state_dict()[k]
t2.load_state_dict(temp_dict, strict=False)

for i in range(1,4):
    exec(f"t{i} = t{i}.cuda()")

"cut 3 models into 6 modules"
import copy
def split_model(m):
    h1 = copy.deepcopy(m)
    h2 = copy.deepcopy(m)
    h1.layer3 = nn.Sequential()
    h1.layer4 = nn.Sequential()
    h1.avgpool = nn.Sequential()
    h1.fc = nn.Sequential()

    h2.layer1 = nn.Sequential()
    h2.layer2 = nn.Sequential()
    h2.conv1 = nn.Sequential()
    h2.bn1 = nn.Sequential()
    h2.relu = nn.Sequential()
    h2.maxpool = nn.Sequential()
    return h1, h2
m11, m12 = split_model(t1)
m21, m22 = split_model(t2)
m31, m32 = split_model(t3)
def replace_relu(m):
    m.layer3[0].relu = nn.ReLU()
    m.layer3[1].relu = nn.ReLU()
    m.layer4[0].relu = nn.ReLU()
    m.layer4[1].relu = nn.ReLU()
    return m
m12 = replace_relu(m12)
m22 = replace_relu(m22)
m32 = replace_relu(m32)

#%%
oc_sd=0.01
loga = (torch.randn(4)*oc_sd).cuda().requires_grad_()
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

opt = {'epochs':30}
optimizer = torch.optim.RAdam([loga],
                lr= 1e-3,
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
loss_func = nn.CrossEntropyLoss()

loss_tr, loss_val = [], []
lamb = 5
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
        half0 = m11(inputs).reshape(gt_label.shape[0], 128,7,7)*z[:,0:1, None, None] \
            + m21(inputs).reshape(gt_label.shape[0], 128,7,7)*z[:,1:2, None, None]
            
        pred0, pred1 = m12(half0)*z[:,2:3], m22(half0)*z[:,3:4]
        l0, l1 = loss_func(pred0, gt_label), loss_func(pred1, gt_label)
        loss = l0 + l1 + lamb*(z[:2].mean()+z[2:].mean() - 1)
        loss.backward()
        loss_tr.append(loss.cpu().detach().item())
        optimizer.step()
        # if batch_index%10 == 0:
        #     print("laga", loga)      
        g_ind = z[:, 2:].argmax(dim=1)
        pred = torch.stack((pred0, pred1), dim=1)
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
            half0 = m11(inputs)*z[:,0:1] + m21(inputs)*z[:,1:2] 
            half0r = half0.reshape(gt_label.shape[0],128,7,7)
            pred0, pred1 = m12(half0r)*z[:,2:3], m22(half0r)*z[:,3:4]

            g_ind = z[:, 2:].argmax(dim=1)
            pred = torch.stack((pred0, pred1), dim=1)
            predictions = pred[range(z.shape[0]),g_ind]
            _, predicted_class = predictions.max(1)
            total_val_examples += predicted_class.size(0)
            num_correct_val += predicted_class.eq(gt_label).sum().item()

        # get validation results
        val_acc = num_correct_val / total_val_examples
        print("Validation accuracy: {}".format(val_acc))
        val_accs.append(val_acc)
        print('after validation', loga.detach().cpu())

    # Finally, save model if the validation accuracy is the best so far
    if val_acc > best_validation_accuracy:
        best_validation_accuracy = val_acc
        print("Validation accuracy improved; saving model.")
        # print(loga.detach().cpu())
        

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