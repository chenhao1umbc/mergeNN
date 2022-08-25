#%%
import torch
from modules import *
from utils import *
from torch.utils.data import Dataset, DataLoader


#%%
teacher0 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
teacher0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
teacher0.load_state_dict(torch.load(f'teachers/teacher{11}.pt'))

teacher1 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
teacher1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
teacher1.load_state_dict(torch.load(f'teachers/teacher{10}.pt'))

#%%
class Connect_conv(nn.Module):
    def __init__(self, conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
        super().__init__()
        """middle conv is conv in layers/blocks"""
        in_channels = conv1.in_channels
        out_channels = conv1.out_channels
        self.conv = nn.Conv2d(in_channels*2, out_channels*2, kernel_size=conv1.kernel_size, stride=conv1.stride,
                        padding=conv1.padding, bias=False)
        self.conv.weight.data *= 0
        self.conv.weight.data[:out_channels, :in_channels] = conv1.weight.data.detach().clone()
        self.conv.weight.data[out_channels:, in_channels:] = conv2.weight.data.detach().clone()
    def forward(self, x):
        return self.conv

class Connect_bn(nn.Module):
    def __init__(self, n_channel, bn1, bn2):
        super().__init__()
        self.out = nn.BatchNorm2d(n_channel)
        self.out.weight.data[:n_channel//2] = bn1.weight.data.clone()
        self.out.weight.data[n_channel//2:] = bn2.weight.data.clone()
        self.out.bias.data[:n_channel//2] = bn1.bias.data.clone()
        self.out.bias.data[n_channel//2:] = bn2.bias.data.clone()
    def forward(self, x):    
        return self.out

class Layer_l0(nn.Module):
    def __init__(self, tch0_layer, tch1_layer):
        super().__init__()
        self.relu = nn.ReLU()

        n_channel = tch0_layer.conv1.out_channels * 2
        self.main_gate = L0GateLayer2d(n_channel)

        # downsample
        if tch0_layer.downsample_conv is not None:
            self.downsample_conv = connect_middle_conv(tch0_layer.downsample_conv, tch1_layer.downsample_conv)
            self.downsample_bn = nn.BatchNorm2d(n_channel)
        else:
            self.downsample_conv = None

        # first block
        self.block0 = nn.Sequential(
            Connect_conv(tch0_layer[0].conv1, tch1_layer[0].conv1), #self.conv1
            Connect_bn(n_channel, tch0_layer[0].bn1, tch1_layer[0].bn1), #self.bn1
            nn.ReLU(inplace=True),
            L0GateLayer2d(n_channel), #self.gate1
            Connect_conv(tch0_layer[0].conv2, tch1_layer[0].conv2), # self.conv2
            Connect_bn(n_channel, tch0_layer[0].bn2, tch1_layer[0].bn2) #self.bn2
        )
        self.gate1 = self.block0[3]
        # second block
        self.block1 = nn.Sequential(
            Connect_conv(tch0_layer[1].conv1, tch1_layer[1].conv1), #self.conv3 =
            Connect_bn(n_channel, tch0_layer[1].bn1, tch1_layer[1].bn1), # self.bn3
            nn.ReLU(inplace=True),
            L0GateLayer2d(n_channel), # self.gate2            
            Connect_conv(tch0_layer[1].conv2, tch1_layer[1].conv2), # self.conv4
            Connect_bn(n_channel, tch0_layer[1].bn4, tch1_layer[1].bn2) # self.bn4
        )
        self.gate2 = self.block1[3]

    def forward(self, x, main_mask=None):
        if main_mask is None:
            main_mask = self.main_gate.mask(x)

        # downsample
        identity = x
        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        # first block
        x = self.block0(x)
        x += identity
        x = self.main_gate(x, main_mask)

        # second block
        identity = x
        x = self.block0(x)
        x += identity
        x = self.main_gate(x, main_mask)

        return x

    def l0_loss(self):
        return self.main_gate.l0_loss() + self.gate1.l0_loss() + self.gate2.l0_loss()

    def compress(self, in_importance_indices):
        out_importance_indices = self.main_gate.important_indices().detach()
        planes = len(out_importance_indices)

        # downsample
        if self.downsample_conv is not None:
            self.downsample_conv = compress_conv2d(self.downsample_conv, in_importance_indices, out_importance_indices)
            self.downsample_bn = nn.BatchNorm2d(planes)

        # first block
        important_indices_in_block = self.gate1.important_indices()
        self.conv1 = compress_conv2d(self.conv1, in_importance_indices, important_indices_in_block)
        self.bn1 = compress_bn(self.bn1, planes, important_indices_in_block)
        self.conv2 = compress_conv2d(self.conv2, important_indices_in_block, out_importance_indices)
        self.bn2 = compress_bn(self.bn2, planes, out_importance_indices)

        # second block
        important_indices_in_block = self.gate2.important_indices()
        self.conv3 = compress_conv2d(self.conv3, out_importance_indices, important_indices_in_block)
        self.bn3 = compress_bn(self.bn3, planes, important_indices_in_block)
        self.conv4 = compress_conv2d(self.conv4, important_indices_in_block, out_importance_indices)
        self.bn4 = compress_bn(self.bn4, planes, out_importance_indices)

        important_indices_in_block = self.gate3.important_indices()
        self.conv5 = compress_conv2d(self.conv5, out_importance_indices, important_indices_in_block)
        self.bn5 = compress_bn(self.bn5, planes, important_indices_in_block)
        self.conv6 = compress_conv2d(self.conv6, important_indices_in_block, out_importance_indices)
        self.bn6 = compress_bn(self.bn6, planes, out_importance_indices)

        delattr(self, 'main_gate')
        delattr(self, 'gate1')
        delattr(self, 'gate2')
        delattr(self, 'gate3')
        self.forward = types.MethodType(new_forward, self)
        return out_importance_indices

    def gate_parameters(self):
        return chain(self.main_gate.parameters(), self.gate1.parameters(), self.gate2.parameters(), self.gate3.parameters())

    def non_gate_parameters(self):
        parameters = [self.conv1.parameters(),
                      self.bn1.parameters(),
                      self.conv2.parameters(),
                      self.bn2.parameters(),
                      self.conv3.parameters(),
                      self.bn3.parameters(),
                      self.conv4.parameters(),
                      self.bn4.parameters(),
                      self.conv5.parameters(),
                      self.bn5.parameters(),
                      self.conv6.parameters(),
                      self.bn6.parameters()]
        if self.downsample_conv is not None:
            parameters += [self.downsample_conv.parameters(),
                           self.downsample_bn.parameters()]

        return chain.from_iterable(parameters)

    def gate_values(self):
        """used only for plots"""
        return [self.main_gate.importance_of_features(), self.gate1.importance_of_features(),
                self.gate2.importance_of_features()]

class Student(nn.Module):
    def __init__(self, teacher0, teacher1):
        super().__init__()
        self.conv1 = connect_first_conv2d(teacher0.conv1, teacher1.conv1)
        self.bn1 = Connect_bn(64 * 2, teacher0.bn1, teacher1.bn1)
        self.relu = nn.ReLU()

        self.layer1 = Layer_l0(teacher0.layer1, teacher1.layer1)
        self.gate = self.layer1.main_gate
        self.layer2 = Layer_l0(teacher0.layer2, teacher1.layer2)
        self.layer3 = Layer_l0(teacher0.layer3, teacher1.layer3)

        self.layers = [self.layer1, self.layer2, self.layer3]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = connect_final_linear(teacher0.fc, teacher1.fc)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        mask = self.gate.mask(x)
        x = self.gate(x, mask)

        x = self.layers[0](x, mask)
        for layer in self.layers[1:]:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    def l0_loss(self):
        l0_loss = 0#self.gate.l0_loss()
        for layer in self.layers:
            l0_loss += layer.l0_loss()
        return l0_loss

    def compress(self):
        importance_indices = self.gate.important_indices()
        self.conv1 = compress_conv2d(self.conv1, torch.ones(3, dtype=torch.bool), importance_indices)
        self.bn1 = compress_bn(self.bn1, 16, importance_indices)
        for layer in self.layers:
            importance_indices = layer.compress(importance_indices)
        self.fc = compress_final_linear(self.fc, importance_indices)

        delattr(self, 'gate')
        self.forward = types.MethodType(new_forward, self)

    def gate_parameters(self):
        parameters = []# [self.gate.parameters()]
        for layer in self.layers:
            parameters.append(layer.gate_parameters())
        return chain.from_iterable(parameters)

    def non_gate_parameters(self):
        parameters = [self.conv1.parameters(), self.bn1.parameters(), self.fc.parameters()]
        for layer in self.layers:
            parameters.append(layer.non_gate_parameters())
        return chain.from_iterable(parameters)

    def gate_values(self):
        values = []#[self.gate.importance_of_features()]
        for layer in self.layers:
            values += layer.gate_values()

        return values

model = Student(teacher0, teacher1)

#%% prepare data
num_train = 2600+2200
num_val = 2200//2
num_test = 2200//2
split1 = num_val+num_train
split2 = num_val+num_train + num_test
neg_all = torch.load('./data/neg_all.pt') # check prep_data.py for more info
pos_all = torch.load('./data/pos_all.pt')

train_dataset = Data.TensorDataset(torch.cat((pos_all[:num_train], neg_all[:num_train]), dim=0), \
                torch.cat((torch.ones(num_train, dtype=int), torch.zeros(num_train,  dtype=int)), dim=0))
val_dataset = Data.TensorDataset(torch.cat((pos_all[num_train:split1],neg_all[num_train:split1]), dim=0), \
                torch.cat((torch.ones(num_val, dtype=int), torch.zeros(num_val, dtype=int)), dim=0))
test_dataset = Data.TensorDataset(torch.cat((pos_all[split1:split2], neg_all[split1:split2]), dim=0), \
                torch.cat((torch.ones(num_test, dtype=int), torch.zeros(num_test, dtype=int)), dim=0))
train_batchsize = 64
eval_batchsize = 32
train_loader = DataLoader(train_dataset, train_batchsize, shuffle=True)                                      
validation_loader = DataLoader(val_dataset, eval_batchsize)
test_loader = DataLoader(test_dataset, eval_batchsize)


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
        # for batch_index, (inputs, gt_label) in tqdm(enumerate(validation_loader), total=len(validation_dataset)//eval_batchsize):
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