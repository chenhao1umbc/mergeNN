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

    
        


#%%
#split data
train_fraction = 0.8
validation_fraction = 0.1
test_fraction = 0.1
dataset_size = len(dataset)
num_train = int(train_fraction * dataset_size)
num_validation = int(validation_fraction * dataset_size)
num_test = int(test_fraction * dataset_size)
print(num_train, num_validation, num_test)