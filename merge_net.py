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