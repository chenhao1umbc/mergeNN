#%% load dependency
import os
import numpy as np
import pandas as pd
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

#%% prepare data  -- DBC typical way, not save as .pt
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

#%% DBC split by objects
neg_all = torch.load('./data/neg_all.pt') # check prep_data.py for more info
pos_all = torch.load('./data/pos_all.pt')
if True:  # if False means split by objects
    idx = torch.randperm(pos_all.shape[0])
    neg_all = neg_all[idx]
    pos_all = pos_all[idx]
num_train = 2600+2200  # first 200 obejects
num_val = 2200//2
num_test = 2200//2
split1 = num_val+num_train
split2 = num_val+num_train + num_test

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

#%% Lung cancer data
data = np.load('./data/LCD/LIDC_Latest.npy', allow_pickle = True)
df = pd.DataFrame({'subtlety':data[:,0],
                           'internalStructure':data[:,1],
                           'calcification':data[:,2],
                           'sphericity':data[:,3],
                           'margin':data[:,4],
                           'lobulation':data[:,5],
                           'spiculation':data[:,6],
                           'texture':data[:,7],
                           'Malignancy':data[:,8],
                           'Patient_id':data[:,9],
                           '3D Volume':data[:,10],
                           '3D Mask':data[:,11]})
print(df)
exclude_list = ['LIDC-IDRI-0098', 'LIDC-IDRI-0292', 'LIDC-IDRI-0294', 'LIDC-IDRI-0296', 'LIDC-IDRI-0297', 'LIDC-IDRI-0301', 'LIDC-IDRI-0302', 'LIDC-IDRI-0303', 'LIDC-IDRI-0304', 'LIDC-IDRI-0306', 'LIDC-IDRI-0309', 'LIDC-IDRI-0310', 'LIDC-IDRI-0314', 'LIDC-IDRI-0316', 'LIDC-IDRI-0317', 'LIDC-IDRI-0318', 'LIDC-IDRI-0319', 'LIDC-IDRI-0320', 'LIDC-IDRI-0322', 'LIDC-IDRI-0323', 'LIDC-IDRI-0324', 'LIDC-IDRI-0325', 'LIDC-IDRI-0329', 'LIDC-IDRI-0330', 'LIDC-IDRI-0332', 'LIDC-IDRI-0334', 'LIDC-IDRI-0335', 'LIDC-IDRI-0336', 'LIDC-IDRI-0337', 'LIDC-IDRI-0338', 'LIDC-IDRI-0342', 'LIDC-IDRI-0343', 'LIDC-IDRI-0344', 'LIDC-IDRI-0348', 'LIDC-IDRI-0349', 'LIDC-IDRI-0351', 'LIDC-IDRI-0352', 'LIDC-IDRI-0354', 'LIDC-IDRI-0355', 'LIDC-IDRI-0356', 'LIDC-IDRI-0357', 'LIDC-IDRI-0359', 'LIDC-IDRI-0361', 'LIDC-IDRI-0362', 'LIDC-IDRI-0364', 'LIDC-IDRI-0365', 'LIDC-IDRI-0369', 'LIDC-IDRI-0371', 'LIDC-IDRI-0373', 'LIDC-IDRI-0375', 'LIDC-IDRI-0376', 'LIDC-IDRI-0377', 'LIDC-IDRI-0379', 'LIDC-IDRI-0380', 'LIDC-IDRI-0381', 'LIDC-IDRI-0384', 'LIDC-IDRI-0385', 'LIDC-IDRI-0386', 'LIDC-IDRI-0387', 'LIDC-IDRI-0390', 'LIDC-IDRI-0391', 'LIDC-IDRI-0396', 'LIDC-IDRI-0398', 'LIDC-IDRI-0400', 'LIDC-IDRI-0401', 'LIDC-IDRI-0404', 'LIDC-IDRI-0405', 'LIDC-IDRI-0407', 'LIDC-IDRI-0409', 'LIDC-IDRI-0410', 'LIDC-IDRI-0411', 'LIDC-IDRI-0412', 'LIDC-IDRI-0413', 'LIDC-IDRI-0414', 'LIDC-IDRI-0416', 'LIDC-IDRI-0417', 'LIDC-IDRI-0419', 'LIDC-IDRI-0420', 'LIDC-IDRI-0425', 'LIDC-IDRI-0431', 'LIDC-IDRI-0433', 'LIDC-IDRI-0435', 'LIDC-IDRI-0436', 'LIDC-IDRI-0437', 'LIDC-IDRI-0438', 'LIDC-IDRI-0439', 'LIDC-IDRI-0441', 'LIDC-IDRI-0446', 'LIDC-IDRI-0448', 'LIDC-IDRI-0449', 'LIDC-IDRI-0451', 'LIDC-IDRI-0453', 'LIDC-IDRI-0454', 'LIDC-IDRI-0456', 'LIDC-IDRI-0457', 'LIDC-IDRI-0462', 'LIDC-IDRI-0463', 'LIDC-IDRI-0465', 'LIDC-IDRI-0467', 'LIDC-IDRI-0469', 'LIDC-IDRI-0471', 'LIDC-IDRI-0472', 'LIDC-IDRI-0474', 'LIDC-IDRI-0478', 'LIDC-IDRI-0479', 'LIDC-IDRI-0480', 'LIDC-IDRI-0482', 'LIDC-IDRI-0484', 'LIDC-IDRI-0486', 'LIDC-IDRI-0487', 'LIDC-IDRI-0488', 'LIDC-IDRI-0490', 'LIDC-IDRI-0492', 'LIDC-IDRI-0493', 'LIDC-IDRI-0494', 'LIDC-IDRI-0498', 'LIDC-IDRI-0499', 'LIDC-IDRI-0500', 'LIDC-IDRI-0502', 'LIDC-IDRI-0503', 'LIDC-IDRI-0505', 'LIDC-IDRI-0508', 'LIDC-IDRI-0511', 'LIDC-IDRI-0512', 'LIDC-IDRI-0513', 'LIDC-IDRI-0514', 'LIDC-IDRI-0516', 'LIDC-IDRI-0517', 'LIDC-IDRI-0519', 'LIDC-IDRI-0520', 'LIDC-IDRI-0525', 'LIDC-IDRI-0526', 'LIDC-IDRI-0527', 'LIDC-IDRI-0528', 'LIDC-IDRI-0530', 'LIDC-IDRI-0531', 'LIDC-IDRI-0536', 'LIDC-IDRI-0537', 'LIDC-IDRI-0538', 'LIDC-IDRI-0543', 'LIDC-IDRI-0544', 'LIDC-IDRI-0547', 'LIDC-IDRI-0548', 'LIDC-IDRI-0550', 'LIDC-IDRI-0551', 'LIDC-IDRI-0552', 'LIDC-IDRI-0556', 'LIDC-IDRI-0557', 'LIDC-IDRI-0558', 'LIDC-IDRI-0560', 'LIDC-IDRI-0561', 'LIDC-IDRI-0562', 'LIDC-IDRI-0563', 'LIDC-IDRI-0564', 'LIDC-IDRI-0565', 'LIDC-IDRI-0566', 'LIDC-IDRI-0567', 'LIDC-IDRI-0569', 'LIDC-IDRI-0570', 'LIDC-IDRI-0571', 'LIDC-IDRI-0572', 'LIDC-IDRI-0573', 'LIDC-IDRI-0575', 'LIDC-IDRI-0576', 'LIDC-IDRI-0579', 'LIDC-IDRI-0580', 'LIDC-IDRI-0581', 'LIDC-IDRI-0582', 'LIDC-IDRI-0583', 'LIDC-IDRI-0584', 'LIDC-IDRI-0586', 'LIDC-IDRI-0587', 'LIDC-IDRI-0589', 'LIDC-IDRI-0590', 'LIDC-IDRI-0595', 'LIDC-IDRI-0596', 'LIDC-IDRI-0597', 'LIDC-IDRI-0598', 'LIDC-IDRI-0599', 'LIDC-IDRI-0601', 'LIDC-IDRI-0602', 'LIDC-IDRI-0603', 'LIDC-IDRI-0604', 'LIDC-IDRI-0605', 'LIDC-IDRI-0606', 'LIDC-IDRI-0607', 'LIDC-IDRI-0608', 'LIDC-IDRI-0609', 'LIDC-IDRI-0610', 'LIDC-IDRI-0612', 'LIDC-IDRI-0613', 'LIDC-IDRI-0614', 'LIDC-IDRI-0615', 'LIDC-IDRI-0616', 'LIDC-IDRI-0617', 'LIDC-IDRI-0620', 'LIDC-IDRI-0621', 'LIDC-IDRI-0622', 'LIDC-IDRI-0624', 'LIDC-IDRI-0626', 'LIDC-IDRI-0627', 'LIDC-IDRI-0630', 'LIDC-IDRI-0631', 'LIDC-IDRI-0632', 'LIDC-IDRI-0634', 'LIDC-IDRI-0637', 'LIDC-IDRI-0639', 'LIDC-IDRI-0643', 'LIDC-IDRI-0646', 'LIDC-IDRI-0651', 'LIDC-IDRI-0653', 'LIDC-IDRI-0654', 'LIDC-IDRI-0655', 'LIDC-IDRI-0657', 'LIDC-IDRI-0659', 'LIDC-IDRI-0663', 'LIDC-IDRI-0664', 'LIDC-IDRI-0668', 'LIDC-IDRI-0669', 'LIDC-IDRI-0670', 'LIDC-IDRI-0675', 'LIDC-IDRI-0678', 'LIDC-IDRI-0679', 'LIDC-IDRI-0680', 'LIDC-IDRI-0683', 'LIDC-IDRI-0684', 'LIDC-IDRI-0685', 'LIDC-IDRI-0687', 'LIDC-IDRI-0690', 'LIDC-IDRI-0692', 'LIDC-IDRI-0693', 'LIDC-IDRI-0694', 'LIDC-IDRI-0696', 'LIDC-IDRI-0701', 'LIDC-IDRI-0703', 'LIDC-IDRI-0711', 'LIDC-IDRI-0712', 'LIDC-IDRI-0713', 'LIDC-IDRI-0715', 'LIDC-IDRI-0716', 'LIDC-IDRI-0717', 'LIDC-IDRI-0719', 'LIDC-IDRI-0720', 'LIDC-IDRI-0721', 'LIDC-IDRI-0722', 'LIDC-IDRI-0724', 'LIDC-IDRI-0726', 'LIDC-IDRI-0727', 'LIDC-IDRI-0731', 'LIDC-IDRI-0732', 'LIDC-IDRI-0733', 'LIDC-IDRI-0735', 'LIDC-IDRI-0737', 'LIDC-IDRI-0738', 'LIDC-IDRI-0740', 'LIDC-IDRI-0741', 'LIDC-IDRI-0742', 'LIDC-IDRI-0743', 'LIDC-IDRI-0744', 'LIDC-IDRI-0745', 'LIDC-IDRI-0747', 'LIDC-IDRI-0749', 'LIDC-IDRI-0750', 'LIDC-IDRI-0753', 'LIDC-IDRI-0755', 'LIDC-IDRI-0759', 'LIDC-IDRI-0763', 'LIDC-IDRI-0764', 'LIDC-IDRI-0765', 'LIDC-IDRI-0766', 'LIDC-IDRI-0767', 'LIDC-IDRI-0768', 'LIDC-IDRI-0769', 'LIDC-IDRI-0771', 'LIDC-IDRI-0773', 'LIDC-IDRI-0779', 'LIDC-IDRI-0780', 'LIDC-IDRI-0786', 'LIDC-IDRI-0787', 'LIDC-IDRI-0789', 'LIDC-IDRI-0792', 'LIDC-IDRI-0794', 'LIDC-IDRI-0795', 'LIDC-IDRI-0797', 'LIDC-IDRI-0798', 'LIDC-IDRI-0800', 'LIDC-IDRI-0801', 'LIDC-IDRI-0804', 'LIDC-IDRI-0806', 'LIDC-IDRI-0809', 'LIDC-IDRI-0811', 'LIDC-IDRI-0813', 'LIDC-IDRI-0814', 'LIDC-IDRI-0816', 'LIDC-IDRI-0817', 'LIDC-IDRI-0819', 'LIDC-IDRI-0821', 'LIDC-IDRI-0822', 'LIDC-IDRI-0823', 'LIDC-IDRI-0824', 'LIDC-IDRI-0830', 'LIDC-IDRI-0831', 'LIDC-IDRI-0832', 'LIDC-IDRI-0834', 'LIDC-IDRI-0835', 'LIDC-IDRI-0836', 'LIDC-IDRI-0841', 'LIDC-IDRI-0843', 'LIDC-IDRI-0844', 'LIDC-IDRI-0845', 'LIDC-IDRI-0846', 'LIDC-IDRI-0847', 'LIDC-IDRI-0849', 'LIDC-IDRI-0851', 'LIDC-IDRI-0852', 'LIDC-IDRI-0853', 'LIDC-IDRI-0857', 'LIDC-IDRI-0860', 'LIDC-IDRI-0862', 'LIDC-IDRI-0863', 'LIDC-IDRI-0866', 'LIDC-IDRI-0868', 'LIDC-IDRI-0869', 'LIDC-IDRI-0870', 'LIDC-IDRI-0871', 'LIDC-IDRI-0873', 'LIDC-IDRI-0875', 'LIDC-IDRI-0876', 'LIDC-IDRI-0878', 'LIDC-IDRI-0880', 'LIDC-IDRI-0881', 'LIDC-IDRI-0885', 'LIDC-IDRI-0886', 'LIDC-IDRI-0887', 'LIDC-IDRI-0889', 'LIDC-IDRI-0891', 'LIDC-IDRI-0893', 'LIDC-IDRI-0895', 'LIDC-IDRI-0896', 'LIDC-IDRI-0898', 'LIDC-IDRI-0899', 'LIDC-IDRI-0901', 'LIDC-IDRI-0903', 'LIDC-IDRI-0904', 'LIDC-IDRI-0905', 'LIDC-IDRI-0906', 'LIDC-IDRI-0909', 'LIDC-IDRI-0911', 'LIDC-IDRI-0912', 'LIDC-IDRI-0913', 'LIDC-IDRI-0915', 'LIDC-IDRI-0916', 'LIDC-IDRI-0917', 'LIDC-IDRI-0918', 'LIDC-IDRI-0921', 'LIDC-IDRI-0922', 'LIDC-IDRI-0923', 'LIDC-IDRI-0926', 'LIDC-IDRI-0927', 'LIDC-IDRI-0931', 'LIDC-IDRI-0934', 'LIDC-IDRI-0935', 'LIDC-IDRI-0937', 'LIDC-IDRI-0939', 'LIDC-IDRI-0942', 'LIDC-IDRI-0945', 'LIDC-IDRI-0946', 'LIDC-IDRI-0948', 'LIDC-IDRI-0951', 'LIDC-IDRI-0952', 'LIDC-IDRI-0953', 'LIDC-IDRI-0958', 'LIDC-IDRI-0962', 'LIDC-IDRI-0963', 'LIDC-IDRI-0964', 'LIDC-IDRI-0966', 'LIDC-IDRI-0967', 'LIDC-IDRI-0973', 'LIDC-IDRI-0974', 'LIDC-IDRI-0976', 'LIDC-IDRI-0977', 'LIDC-IDRI-0978', 'LIDC-IDRI-0980', 'LIDC-IDRI-0981', 'LIDC-IDRI-0982', 'LIDC-IDRI-0985', 'LIDC-IDRI-0989', 'LIDC-IDRI-0990', 'LIDC-IDRI-0993', 'LIDC-IDRI-0995', 'LIDC-IDRI-0996', 'LIDC-IDRI-0997', 'LIDC-IDRI-0999', 'LIDC-IDRI-1000', 'LIDC-IDRI-1001', 'LIDC-IDRI-1002', 'LIDC-IDRI-1003', 'LIDC-IDRI-1004', 'LIDC-IDRI-1006']
df_curated = df.copy()
df_curated = df_curated[~df_curated['Patient_id'].isin(exclude_list)]
print(len(df_curated))
# Create the dataset
df1 = df_curated.copy()
df1 = df1[df1['Malignancy']!=3]
df1.loc[df1['Malignancy']<=2, ['Malignancy']] = 0
df1.loc[df1['Malignancy']>=4, ['Malignancy']] = 1
neg = np.stack(df1[df1['Malignancy']==0]['3D Volume'].to_numpy())
pos = np.stack(df1[df1['Malignancy']==1]['3D Volume'].to_numpy())
n = torch.tensor(neg)
p = torch.tensor(pos)
min_n = min(n.shape[0], p.shape[0])
# torch.save([n[:min_n], p[:min_n]], 'neg_pos.pt')
