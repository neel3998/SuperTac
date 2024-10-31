
from torch.utils.data.dataset import Dataset, T
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
import pandas as pd 

from utils.get3Dmap import CoordinateDesc
import cv2
import numpy as np
# warnings.filterwarnings("error")

def findMinMax(array):

    logvar = 0
    if np.var(array)>0:
        logvar = np.log(np.var(array))

    std = np.exp(logvar*0.5)
    eps = np.random.normal(0,0.01)
    max_v = np.max(array)+eps*std*0.01
    min_v = np.min(array)+eps*std*0.01

    return min_v, max_v

def normalise(min_v, max_v, array):
    if min_v == max_v:
        return np.zeros_like(array)
    else:
        array = 1 - (array - min_v)/(max_v - min_v)
        return array

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

# 12 cm dimension of entire sensor. 
class TrainDataset(Dataset):

    def __init__(self, startIndex, endIndex):
        self.sequenceLength = 6
        # self.prefix = prefix
        self.startIndex = startIndex
        self.endIndex = endIndex

        self.Ztrain = pd.read_csv('../Datasets/vel300/cleanZ.csv', delimiter= ',', index_col=False)
        self.Vtrain = pd.read_csv('../Datasets/vel300/cleanV.csv', delimiter= ',', index_col=False)
        
        self.length = self.endIndex - self.startIndex

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.counter = self.startIndex
        self.pass_no = 0
        self.class_name = 'plain'
        self.iter_no = 1

        self.sensordim_Len = 12 # in cms
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.seq_len = 512

        self.prevdatadict = None

        self.zDescriptor = CoordinateDesc(pat = '3_Bumps', num = 0) #num --> iter no, pat ---> pattern name
        self.imgdim = 4 #dimension of high resolution image


    # Make Image from given z and v data.
    def mk_img(self,data_dict, dataType = 'z'):
        a = []
        b= [] 
        for i in range(4):
            z_horiz = np.concatenate([data_dict[dataType+str(4*i + j)].reshape(1,self.seq_len,1) \
                                    for j in range(4)], axis = 0)
            z_vert = np.concatenate([data_dict[dataType+str(i + 4*j)].reshape(1,self.seq_len,1) \
                                    for j in range(4)], axis = 0)
            z_horiz = np.repeat(z_horiz, 32, axis = -1)
            z_vert = np.repeat(z_vert, 32, axis = -1)

            a.append(z_horiz)
            b.append(z_vert)

        return np.array(a), np.array(b)
      
    def _genHRImg(self,xloc, yloc, scaling):
        xstart = xloc - self.sensordim_Len/2 # xloc and yloc are the central position of the overall sensor
        ystart = yloc - self.sensordim_Len/2 

        HRimgdim = self.imgdim*scaling
        img = np.zeros((HRimgdim,HRimgdim))
        # smallimg = np.zeros((self.imgdim//8,self.imgdim//8))

        for i in range(HRimgdim):
            for j in range(HRimgdim):
            # pass    
                z = self.zDescriptor.get_Z_coord([np.round(xstart + j*3*4/HRimgdim,3),np.round(ystart + i*3*4/HRimgdim,3)])
                # print((z+2.5)/5)
                img[i][j] = (z + 2.5)/5

        return img

                

    def __getitem__(self, index):
        index += self.startIndex
        data_dict = {}

        check = 0
        t_dash = index
        classname = self.Ztrain[index:index+1]['Class'].item()
        iter_no = self.Ztrain[index:index+1]['iter_No'].item()
        pass_no = self.Ztrain[index:index+1]['Pass_No'].item()

        xcent = self.Ztrain[index:index+1]['Xcent'].item()
        ycent = self.Ztrain[index:index+1]['Ycent'].item()


        for i in range(self.seq_len):
            m = index - i
            if (self.Ztrain[m:m+1]['Class'].item() != classname or \
                self.Ztrain[m:m+1]['iter_No'].item() != iter_no or \
                self.Ztrain[m:m+1]['Pass_No'].item() != pass_no):
                t_dash = m
                check = 1
                break

            if m==0:
                t_dash = m
                check = 1
                break
        a = []
        for key in self.Ztrain.keys():
            if key.isnumeric():
                if check:
                    
                    temp_v1 = np.array([2.44]*(self.seq_len - index + t_dash))
                    min_v, max_v = findMinMax(temp_v1)
                    temp_v1 = normalise(min_v, max_v, temp_v1)+0.5
                    
                    temp_v2 = np.array(self.Vtrain[key].to_list()[t_dash+1:index+1])
                    
                    if len(temp_v2) != 0:
                        min_v, max_v = findMinMax(temp_v2)
                        temp_v2 = normalise(min_v, max_v, temp_v2)

                        temp_v = np.hstack((temp_v1,temp_v2))
                        temp_v = moving_average(temp_v, 30)

                        temp_z = np.array([0]*(self.seq_len - index + t_dash))
                        temp_z = np.hstack((temp_z, np.array(self.Ztrain[key].to_list()[t_dash+1:index+1])))
                    else:
                        data_dict =self.prevdatadict
                        break
                        # temp_v = self.prevV
                        # temp_z = self.prevZ

                else:
                    
                    temp_v = np.array(self.Vtrain[key].to_numpy()[index-self.seq_len+1:index+1])
                    min_v, max_v = findMinMax(temp_v)
                    temp_v = normalise(min_v, max_v, temp_v)
                    temp_v = moving_average(temp_v, 30)
                    temp_z = np.array(self.Ztrain[key].to_numpy()[index-self.seq_len+1:index+1])
                
                data_dict['z'+key] = (temp_z+2.5)/5
                data_dict['v'+key] = temp_v

        self.prevdatadict = data_dict

        z_horiz, z_vert = self.mk_img(data_dict, 'z')
        v_horiz, v_vert = self.mk_img(data_dict, 'v')

        self.zDescriptor.pat = classname.split('_')[1]
        self.zDescriptor.num = int(classname.split('_')[0])
        # hrimg = cv2.imread('../Datasets/vel300/img/HR_'+str(index)+'.jpg')[:,:,0]/255
        hrimg = self._genHRImg(xcent, ycent, 2)
        hrimg = self.transform(hrimg)

        class_type = self.class_name.split('_')[-1]
        classes = [0,0,0]
        if class_type == 'Bumps':
            classes[0] = 1
        elif class_type == 'Waves':
            classes[1] = 1
        else:
            classes[2] = 1

        return {'z_vert':z_vert, 'v_vert':v_vert, 'z_horiz':z_horiz, 'v_horiz':v_horiz, 'hr':hrimg, 'class':classes}

    def __len__(self):
        return self.length
    
class custom_data(pl.LightningDataModule):
    def setup(self, stage = None):
        self.cpu = 20
        self.pin = True
        self.batchsize = 100
        print('Loading Dataset ...')
    
    def train_dataloader(self):
        dataset = TrainDataset(1,13401)
        return DataLoader(dataset, batch_size=self.batchsize,
                          num_workers=self.cpu, shuffle=True, pin_memory=self.pin)

    def val_dataloader(self):
        dataset = TrainDataset(20000,22000)
        return DataLoader(dataset, batch_size=self.batchsize,
                          num_workers=self.cpu, shuffle=False, pin_memory=self.pin)
    
    def test_dataloader(self):
        dataset = TrainDataset(1500,31590)
        return DataLoader(dataset, batch_size=1,
                          num_workers=self.cpu, shuffle=False, pin_memory=self.pin)


if __name__ == "__main__":
    data = TrainDataset(5000, 36080)
    m = data.__getitem__(36079)
    print(m['z_horiz'].shape)
    print(m['z_vert'].shape)

    # horizontal visualisation
    a = np.concatenate(tuple([ m['z_horiz'][:,i] for i in range(4)]), axis = -1)
    a = np.concatenate(tuple([ a[i] for i in range(4)]), axis = -1)
    cv2.imwrite('a.jpg', (a*255).astype('uint8'))

    # Vertical visualisation
    a = np.concatenate(tuple([ m['z_vert'][:,i] for i in range(4)]), axis = -1)
    a = np.concatenate(tuple([ a[i] for i in range(4)]), axis = -1)
    cv2.imwrite('b.jpg', (a*255).astype('uint8'))
