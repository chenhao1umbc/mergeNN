#%% load dependency
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
import torch.nn.functional as Func

#%% prepare data
class DBCDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.data_dir = data_dir
        self.img_size = img_size
        
        # assign labels to data within this Dataset
        self.labels = None
        self.create_labels()

    def create_labels(self):
        # create and store a label (positive/1 or negative/0 for each image)
        # each label is the tuple: (img filename, label number (0 or 1))
        labels = []
        for folder in os.listdir(data_dir):
            d = os.path.join(data_dir, folder)
            for i, v in enumerate(['neg', 'pos']):
                case_dir = os.path.join(d, v)
                for fname in os.listdir(case_dir):
                    if '.png' in fname:
                        fpath = os.path.join(case_dir, fname)
                        labels.append((fpath, i))
        self.labels = labels

                 
    def normalize(self, img):
        # convert uint16 -> float
        img = img.astype(np.float32) * 255. / img.max()       
        return img
    
    def __getitem__(self, idx):
        # required method for accessing data samples
        # returns data with its label
        fpath, target  = self.labels[idx]
        
        # load img from file (png or jpg)
        img_arr = imread(fpath, as_gray=True)
        
        # normalize image
        img_arr = self.normalize(img_arr)
        
        # convert to tensor (PyTorch matrix)
        data = torch.from_numpy(img_arr)
        data = data.type(torch.FloatTensor) 
       
        # add image channel dimension (to work with neural network)
        data = torch.unsqueeze(data, 0)
        
        # resize image
        data = transforms.Resize((self.img_size, self.img_size))(data)
        
        return data, target

    def __len__(self):
        # required method for getting size of dataset
        return len(self.labels)
data_dir = './data' #data directory
img_size = 128  # image size
dataset = DBCDataset(data_dir, img_size)
#split data
train_fraction = 0.8
validation_fraction = 0.1
test_fraction = 0.1
dataset_size = len(dataset)
num_train = int(train_fraction * dataset_size)
num_validation = int(validation_fraction * dataset_size)
num_test = int(test_fraction * dataset_size)
print(num_train, num_validation, num_test)

train_dataset, validation_dataset, test_dataset = \
    torch.utils.data.random_split(dataset, [num_train, num_validation, num_test])
train_batchsize = 64
eval_batchsize = 32
train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True)                                      
validation_loader = DataLoader(validation_dataset, batch_size=eval_batchsize)
test_loader = DataLoader(test_dataset, batch_size=eval_batchsize)

#%% load model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# make first layer channel==1
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = model.cuda()

#%% train_net
id = 0 # for diff. runs
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

	for batch_index, (inputs, gt_label) in tqdm(enumerate(train_loader), total=len(train_dataset)//train_batchsize):
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



	# predict on validation set (similar to training set):
	total_val_examples = 0
	num_correct_val = 0

	# switch network from training mode (parameters can be trained) to evaluation mode (parameters can't be trained)
	model.eval()

	with torch.no_grad(): # don't save parameter gradients/changes since this is not for model training
		for batch_index, (inputs, gt_label) in tqdm(enumerate(validation_loader), total=len(validation_dataset)//eval_batchsize):
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