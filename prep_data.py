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
"""pos/neg_all contains 300 objects(patients). first 100 objects give 2600 pos images, 
    then 101-200 objects gives 2200 pos images, and 201-300 gives 2200 pos images.
    Total 7000 positive images and 7000 negative images
    If we randomly split as train/val/test, it is very easy to get 0.999 acc.
    Here we use first 200 objects, 4600 pos and 4600 neg as training,
    the 100 objects with 2200 pos and 2200 neg as validation and test
    """
data_dir = './data/DBC' #data directory
img_size = 128  # image size

labels = []
for folder in os.listdir(data_dir):
    d = os.path.join(data_dir, folder)
    for i, v in enumerate(['neg', 'pos']):
        case_dir = os.path.join(d, v)
        for fname in os.listdir(case_dir):
            if '.png' in fname:
                fpath = os.path.join(case_dir, fname)
                labels.append((fpath, i))
dataset_pos, dataset_neg = [], []
for p, l in labels:
    img_arr = imread(p, as_gray=True)
    # normalize image
    img_arr = img_arr.astype(np.float32) * 255. / img_arr.max() 
    # convert to tensor (PyTorch matrix)
    data = torch.from_numpy(img_arr)
    data = data.type(torch.FloatTensor) 
    # add image channel dimension (to work with neural network)
    data = torch.unsqueeze(data, 0)
    # resize image
    data = transforms.Resize((img_size, img_size))(data)
    if l == 0 :
        dataset_neg.append(data)
    else:
        dataset_pos.append(data)
pos_all = torch.stack(dataset_pos, dim=0) # shape of [7k,1,128,128]
neg_all = torch.stack(dataset_neg, dim=0) # shape of [7k,1,128,128]
if False:
    torch.save(pos_all, './data/pos_all.pt')
    torch.save(pos_all, './data/pos_all.pt')



#%% prepare data  -- typical way, which is too easy
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
data_dir = './data/DBC' #data directory
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
train_loader = DataLoader(train_dataset, train_batchsize, shuffle=True)                                      
validation_loader = DataLoader(validation_dataset, eval_batchsize)
test_loader = DataLoader(test_dataset, eval_batchsize)
