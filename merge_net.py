#%%
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
class Connect_1stconv(nn.Module):
    def __init__(self, conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
        super().__init__()
        """first conv is the conv at the beginning of resnet"""
        ker_size = conv1.kernel_size
        n_stride = conv1.stride
        out_channels = conv1.out_channels
        n_padding = conv1.padding
        self.conv = nn.Conv2d(1, out_channels*2, ker_size, n_stride, n_padding, bias=False)
        self.conv.weight.data[:out_channels] = conv1.weight.data.detach().clone()
        self.conv.weight.data[out_channels:] = conv2.weight.data.detach().clone()
        
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

class Connect_bn(nn.Module):
    def __init__(self, n_channel, bn1, bn2):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_channel)
        self.bn.weight.data[:n_channel//2] = bn1.weight.data.clone()
        self.bn.weight.data[n_channel//2:] = bn2.weight.data.clone()
        self.bn.bias.data[:n_channel//2] = bn1.bias.data.clone()
        self.bn.bias.data[n_channel//2:] = bn2.bias.data.clone()

class Connect_fc(nn.Module):
    def __init__(self, lin1: nn.Linear, lin2: nn.Linear) -> nn.Linear:
        super().__init__()
        in_features = lin1.in_features
        self.fc = nn.Linear(in_features * 2, lin1.out_features)
        self.fc.weight.data *= 0
        self.fc.weight.data[:, :in_features] = lin1.weight.data.detach().clone() / 2
        self.fc.weight.data[:, in_features:] = lin2.weight.data.detach().clone() / 2
        self.fc.bias.data = lin1.bias.data.detach().clone() / 2 + lin2.bias.data.detach().clone() / 2

class Prune_conv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, in_index: Tensor, out_index: Tensor) -> nn.Conv2d:
        super().__init__()
        self.conv = nn.Conv2d(len(in_index), len(out_index), conv.kernel_size,
                            stride=conv.stride, padding=conv.padding, bias=False)
        self.conv.weight.data = conv.weight.data[out_index][:, in_index].detach().clone()

class Prune_bn(nn.Module):
    def __init__(self, bn, planes, indices):
        super().__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.bn.weight.data = bn.weight.data[indices]
        self.bn.bias.data = bn.bias.data[indices]

class Layer_l0(nn.Module):
    def __init__(self, tch0_layer, tch1_layer):
        super().__init__()
        self.relu = nn.ReLU()

        n_channel = tch0_layer[0].conv1.out_channels * 2
        self.main_gate = L0GateLayer2d(n_channel)

        # downsample
        if tch0_layer[0].downsample is not None:
            self.downsample_conv = Connect_conv(tch0_layer[0].downsample[0], \
                                    tch1_layer[0].downsample[0]).conv
            self.downsample_bn = nn.BatchNorm2d(n_channel)
        else:
            self.downsample_conv = None

        # first block
        self.block0 = nn.Sequential(
            Connect_conv(tch0_layer[0].conv1, tch1_layer[0].conv1).conv, #self.conv1
            Connect_bn(n_channel, tch0_layer[0].bn1, tch1_layer[0].bn1).bn, #self.bn1
            nn.ReLU(inplace=True),
            L0GateLayer2d(n_channel), #self.gate1
            Connect_conv(tch0_layer[0].conv2, tch1_layer[0].conv2).conv, # self.conv2
            Connect_bn(n_channel, tch0_layer[0].bn2, tch1_layer[0].bn2).bn #self.bn2
        )
        self.gate1 = self.block0[3]
        # second block
        self.block1 = nn.Sequential(
            Connect_conv(tch0_layer[1].conv1, tch1_layer[1].conv1).conv, #self.conv3 =
            Connect_bn(n_channel, tch0_layer[1].bn1, tch1_layer[1].bn1).bn, # self.bn3
            nn.ReLU(inplace=True),
            L0GateLayer2d(n_channel), # self.gate2            
            Connect_conv(tch0_layer[1].conv2, tch1_layer[1].conv2).conv, # self.conv4
            Connect_bn(n_channel, tch0_layer[1].bn2, tch1_layer[1].bn2).bn # self.bn4
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
        x = self.block1(x)
        x += identity
        x = self.main_gate(x, main_mask)

        return x

    def l0_loss(self):
        return self.main_gate.l0_loss() + self.gate1.l0_loss() + self.gate2.l0_loss()

    def compress(self, in_index):
        out_index = self.main_gate.important_indices().detach()
        n_outchannels = len(out_index)

        # downsample
        if self.downsample_conv is not None:
            self.downsample_conv = compress_conv2d(self.downsample_conv, in_index, out_index)
            self.downsample_bn = nn.BatchNorm2d(n_outchannels)

        # first block
        block_index = self.gate1.important_indices()
        self.conv1 = Prune_conv2d(self.conv1, in_index, block_index).conv
        self.bn1 = Prune_bn(self.bn1, n_outchannels, block_index).bn
        self.conv2 = Prune_conv2d(self.conv2, block_index, out_index).conv
        self.bn2 = Prune_bn(self.bn2, n_outchannels, out_index).bn

        # second block
        block_index = self.gate2.important_indices()
        self.conv3 = Prune_conv2d(self.conv3, out_index, block_index).conv
        self.bn3 = Prune_bn(self.bn3, n_outchannels, block_index).bn
        self.conv4 = Prune_conv2d(self.conv4, block_index, out_index).conv
        self.bn4 = Prune_bn(self.bn4, n_outchannels, out_index).bn

        delattr(self, 'main_gate')
        delattr(self, 'gate1')
        delattr(self, 'gate2')
        self.forward = types.MethodType(new_forward, self)
        return out_index

    def gate_parameters(self):
        return chain(self.main_gate.parameters(), self.gate1.parameters(), self.gate2.parameters())

    def non_gate_parameters(self):
        parameters = [self.conv1.parameters(),
                      self.bn1.parameters(),
                      self.conv2.parameters(),
                      self.bn2.parameters(),
                      self.conv3.parameters(),
                      self.bn3.parameters(),
                      self.conv4.parameters(),
                      self.bn4.parameters()]
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
        self.conv1 = Connect_1stconv(teacher0.conv1, teacher1.conv1).conv
        self.bn1 = Connect_bn(64 * 2, teacher0.bn1, teacher1.bn1).bn
        self.relu = nn.ReLU()

        self.layer1 = Layer_l0(teacher0.layer1, teacher1.layer1)
        self.gate = self.layer1.main_gate
        self.layer2 = Layer_l0(teacher0.layer2, teacher1.layer2)
        self.layer3 = Layer_l0(teacher0.layer3, teacher1.layer3)
        self.layer4 = Layer_l0(teacher0.layer4, teacher1.layer4)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Connect_fc(teacher0.fc, teacher1.fc).fc

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

model = Student(teacher0, teacher1).cuda()

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

optimizer1 = torch.optim.RAdam(model.parameters(),
                lr= 0.001,
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[50, 70], gamma=0.1)
optimizer2 = torch.optim.SGD(model.gate_parameters(), lr=0.2, momentum=0.9)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[], gamma=1)
opt = {'epochs':100}
loss_func = nn.CrossEntropyLoss()
lamb = 0.05
for epoch in range(opt['epochs']):
    model.train()

    # print training info
    print("### Epoch {}:".format(epoch))
    total_train_examples = 0
    num_correct_train = 0

    lamb += 0.05 * math.sqrt(lamb)
    for batch_index, (inputs, gt_label) in enumerate(train_loader):
        inputs = inputs.cuda()
        gt_label = gt_label.cuda()
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        predictions = model(inputs)
        main_loss = loss_func(predictions, gt_label)
        l0_loss = lamb * model.l0_loss()
        loss = main_loss + l0_loss
        loss.backward()
        optimizer1.step()
        optimizer2.step()

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
    with torch.no_grad(): 
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
    scheduler1.step()
    scheduler2.step()

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