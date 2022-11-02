"""Merge the LCD 2d Resnet18
"""

#%%
from modules import *
from utils import *
from torch.utils.data import Dataset, DataLoader


#%% prepare data
neg, pos = torch.load('./data/LCD/neg_pos.pt') # check prep_data.py for more info
neg_all, pos_all = neg.reshape(-1,1,64,64), pos.reshape(-1,1,64,64)
if True:  # if False means split by objects
    idx = torch.randperm(pos_all.shape[0])
    neg_all = neg_all[idx]
    pos_all = pos_all[idx]  #36992, 64, 64]
num_train = 29600
num_val = 3696
num_test = 3696
split1 = num_val+num_train
split2 = num_val+num_train + num_test

train_dataset = Data.TensorDataset(torch.cat((pos_all[:num_train], neg_all[:num_train]), dim=0), \
                torch.cat((torch.ones(num_train, dtype=int), torch.zeros(num_train,  dtype=int)), dim=0))
val_dataset = Data.TensorDataset(torch.cat((pos_all[num_train:split1],neg_all[num_train:split1]), dim=0), \
                torch.cat((torch.ones(num_val, dtype=int), torch.zeros(num_val, dtype=int)), dim=0))
test_dataset = Data.TensorDataset(torch.cat((pos_all[split1:split2], neg_all[split1:split2]), dim=0), \
                torch.cat((torch.ones(num_test, dtype=int), torch.zeros(num_test, dtype=int)), dim=0))
train_batchsize = 128
eval_batchsize = 64
train_loader = DataLoader(train_dataset, train_batchsize, shuffle=True)                                      
validation_loader = DataLoader(val_dataset, eval_batchsize)
test_loader = DataLoader(test_dataset, eval_batchsize)

#%%
id0, id1 = 'LCD2', 'LCD3'

teacher0 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
teacher0.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # original
# teacher0.load_state_dict(torch.load(f'teachers/teacher{id0}.pt'))

teacher1 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
teacher1.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # original
# teacher1.load_state_dict(torch.load(f'teachers/teacher{id1}.pt'))

model = Judge(teacher0, teacher1).cuda()

#%%
model.compress()

