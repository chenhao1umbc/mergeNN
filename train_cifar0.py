#%% load dependency
import os
import numpy as np
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
from datetime import datetime
print('starting date time ', datetime.now())

#%% prepare data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True)

validation_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False,
    num_workers=4, pin_memory=True)

#%% load model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
# model.conv1 = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model = model.cuda()

#%% train_net
id = 'CIFAR0' # for diff. runs
best_validation_accuracy = 0. # used to pick the best-performing model on the validation set
train_accs = []
val_accs = []

opt = {'epochs':200}
optimizer = torch.optim.RAdam(model.parameters(),
                lr= 0.001,
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
loss_func = nn.CrossEntropyLoss()

for epoch in range(opt['epochs']):
    model.train()

    # print training info
    print("### Epoch {}:".format(epoch))
    total_train_examples = 0
    num_correct_train = 0

    for batch_index, (inputs, gt_label) in enumerate(train_loader):
        inputs = inputs.cuda()
        gt_label = gt_label.cuda()
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_func(predictions, gt_label)
        loss.backward()
        optimizer.step()

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
    model.eval()
    with torch.no_grad(): # don't save parameter gradients/changes since this is not for model training
        for batch_index, (inputs, gt_label) in enumerate(validation_loader):
            inputs = inputs.cuda()
            gt_label = gt_label.cuda()
            predictions = model(inputs)

            _, predicted_class = predictions.max(1)
            total_val_examples += predicted_class.size(0)
            num_correct_val += predicted_class.eq(gt_label).sum().item()

        # get validation results
        val_acc = num_correct_val / total_val_examples
        print("Validation accuracy: {}".format(val_acc))
        val_accs.append(val_acc)

    # Finally, save model if the validation accuracy is the best so far
    if val_acc > best_validation_accuracy:
        best_validation_accuracy = val_acc
        print("Validation accuracy improved; saving model.")
        torch.save(model.state_dict(), f'teachers/teacher{id}.pt')


#%% plot train and val results
epochs_list = list(range(opt['epochs']))
plt.figure()
plt.plot(epochs_list, train_accs, 'b-', label='training set accuracy')
plt.plot(epochs_list, val_accs, 'r-', label='validation set accuracy')
plt.xlabel('epoch')
plt.ylabel('prediction accuracy')
plt.ylim(0.5, 1)
plt.title('Classifier training evolution:\nprediction accuracy over time')
plt.legend()
plt.savefig(f'train_val{id}.png')
plt.show()

print('done')
print('End date time ', datetime.now())