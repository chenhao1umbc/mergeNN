#%% load dependency
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
from datetime import datetime
print('starting date time ', datetime.now())


#%%


