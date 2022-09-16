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
neg, pos = torch.load('./data/LCD/neg_pos.pt') # neg and pos each has [1156,32,64,64]
# neg = torch.cat((neg[:,:8],neg[:,8:16],neg[:,16:24], neg[:,24:]), dim=0)
# pos = torch.cat((pos[:,:8],pos[:,8:16],pos[:,16:24], pos[:,24:]), dim=0) # shape [1156*4, 8, 64, 64]
if True:  # if False means split by objects
    idx = torch.randperm(pos.shape[0])
    neg_all = neg[idx][:,None]
    pos_all = pos[idx][:,None]
num_train = 900
num_val = 156
num_test = 100
split1 = num_val+num_train
split2 = num_val+num_train + num_test

train_dataset = Data.TensorDataset(torch.cat((pos_all[:num_train], neg_all[:num_train]), dim=0), \
                torch.cat((torch.ones(num_train, dtype=int), torch.zeros(num_train,  dtype=int)), dim=0))
val_dataset = Data.TensorDataset(torch.cat((pos_all[num_train:split1],neg_all[num_train:split1]), dim=0), \
                torch.cat((torch.ones(num_val, dtype=int), torch.zeros(num_val, dtype=int)), dim=0))
test_dataset = Data.TensorDataset(torch.cat((pos_all[split1:split2], neg_all[split1:split2]), dim=0), \
                torch.cat((torch.ones(num_test, dtype=int), torch.zeros(num_test, dtype=int)), dim=0))
train_batchsize = 64
eval_batchsize = 64
train_loader = DataLoader(train_dataset, train_batchsize, shuffle=True)                                      
validation_loader = DataLoader(val_dataset, eval_batchsize)
test_loader = DataLoader(test_dataset, eval_batchsize)

#%% train_net
from modules import *
class Resnet3D_m(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        #(maxpool): Cifar does not have it.
        self.layer1 = nn.Sequential(
            BasicBlock3D(16, 16),
            BasicBlock3D(16, 16),
            )
        self.layer2 = nn.Sequential(
            BasicBlock3D(16, 16, stride=2, downsample=Downsample3D(16, 16)),
            BasicBlock3D(16, 16),
            )
        self.layer3 = nn.Sequential(
            BasicBlock3D(16, 32, stride=2, downsample=Downsample3D(16, 32)),
            BasicBlock3D(32, 32),
            )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = Resnet3D_m().cuda()
id = 'res3d_m0' # for diff. runs
fig_loc = './data/results/figures/'
mod_loc = './data/results/models/'
if not(os.path.isdir(fig_loc + f'/{id}/')): 
    print('made a new folder')
    os.makedirs(fig_loc + f'{id}/')
    os.makedirs(mod_loc + f'{id}/')
fig_loc = fig_loc + f'{id}/'
mod_loc = mod_loc + f'{id}/'

best_validation_accuracy = 0. # used to pick the best-performing model on the validation set
train_accs = []
val_accs = []

opt = {'epochs':100}
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

    # for batch_index, (inputs, gt_label) in tqdm(enumerate(train_loader), total=len(train_dataset)//train_batchsize):
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
        torch.save(model.state_dict(), mod_loc+f'model_{id}.pt')

        epochs_list = list(range(epoch+1))
        plt.figure()
        plt.plot(epochs_list, train_accs, '-o', label='training set accuracy')
        plt.plot(epochs_list, val_accs, '--x', label='validation set accuracy')
        plt.xlabel('epoch')
        plt.ylabel('prediction accuracy')
        plt.ylim(0.5, 1)
        plt.title('Classifier training evolution:\nprediction accuracy over time')
        plt.legend()
        plt.savefig(fig_loc+f'train_val{id}_{epoch}.png')
        plt.show()


#%% plot train and val results
epochs_list = list(range(opt['epochs']))
plt.figure()
plt.plot(epochs_list, train_accs, '-o', label='training set accuracy')
plt.plot(epochs_list, val_accs, '--x', label='validation set accuracy')
plt.xlabel('epoch')
plt.ylabel('prediction accuracy')
plt.ylim(0.5, 1)
plt.title('Classifier training evolution:\nprediction accuracy over time')
plt.legend()
plt.savefig(fig_loc+f'train_val{id}.png')
plt.show()

print('done')
print('End date time ', datetime.now())