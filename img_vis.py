import numpy as np
import ast
import cv2
import pandas as pd
# from models.neuronCluster import neuronCluster
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("error")

def findMinMax(array):
    try:
        logvar  = np.log(np.var(array))
    except RuntimeWarning:
        logvar = 0
    std = np.exp(logvar*0.5)
    # print(logvar, std)
    eps = np.random.normal(0,0.1)
    # print(eps)
    # print(len(temp_v))
    max_v = np.max(array)+eps*std*0.01
    min_v = np.min(array)+eps*std*0.01

    return min_v, max_v

def normalise(min_v, max_v, array):
    if min_v == max_v:
        return np.zeros_like(array)
    else:
        array = (1 - (array - min_v)/(max_v - min_v))*0.5 + 0.5
        return array

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

class visualise():
    def __init__(self):
        self.seq_len = 128
        print('reading dataset!')
        self.Vtrain = pd.read_csv('../Datasets/vel300/cleanV.csv',index_col=False)
        self.Ztrain = pd.read_csv('../Datasets/vel300/cleanZ.csv',index_col=False)
        self.sensStat = pd.read_csv("sensorStats.csv")
        print('dataset reading done!')
        # self.imgDir = '../Datasets/upd_vel600_DS/clean_imgs/'
        self.seq_len = 500
        # self.clusters = []
        # for _ in range(16):
        #     self.clusters.append(neuronCluster())

    def get_dict(self, index):
        data_dict = {}
        # inp_dict = {}
        # self.seq_len = 500
        check = 0
        t_dash = index
        classname = self.Vtrain[index:index+1]['Class'].item()
        iter_no = self.Vtrain[index:index+1]['iter_No'].item()
        pass_no = self.Vtrain[index:index+1]['Pass_No'].item()

        for i in range(self.seq_len):
            m = index - i
            if (self.Vtrain[m:m+1]['Class'].item() != classname or \
                self.Vtrain[m:m+1]['iter_No'].item() != iter_no or \
                self.Vtrain[m:m+1]['Pass_No'].item() != pass_no):
                t_dash = m
                check = 1
                break

            if m==0:
                t_dash = m
                check = 1
                break

        for key in self.Vtrain.keys():
            if key.isnumeric():
                if check:
                    temp_v1 = np.array([2.44]*(self.seq_len - index + t_dash))
                    # min_v, max_v = findMinMax(temp_v1)
                    # temp_v1 = normalise(min_v, max_v, temp_v1)+0.5
                    
                    temp_v2 = np.array(self.Vtrain[key].to_list()[t_dash+1:index+1])
                    # min_v, max_v = findMinMax(temp_v2)
                    # temp_v2 = normalise(min_v, max_v, temp_v2)

                    temp_v = np.hstack((temp_v1,temp_v2))
                    temp_v = (temp_v*2 - 2.44*1 - 2.5)+1
                    # temp_v = moving_average(temp_v, 30)


                    temp_z = np.array([0]*(self.seq_len - index + t_dash))
                    temp_z = np.hstack((temp_z, np.array(self.Ztrain[key].to_list()[t_dash+1:index+1])))

                else:
                    temp_v = np.array(self.Vtrain[key].to_numpy()[index-self.seq_len+1:index+1])
                    temp_v = (temp_v*20 - 2.44*19 - 2.5)+1

                    # min_v, max_v = findMinMax(temp_v)
                    # temp_v = normalise(min_v, max_v, temp_v)
                    # temp_v = moving_average(temp_v, 30)
                    temp_z = np.array(self.Ztrain[key].to_numpy()[index-self.seq_len+1:index+1])

                
                
                # NORMALISING
                # temp_v = 1 - (temp_v - self.sensStat[key][1])/(self.sensStat[key][0] - self.sensStat[key][1])
                # temp_v = 1 - (temp_v - min_v)/(max_v - min_v)
                
                temp_z = (temp_z + 2.5)/5

                # cluster_v = (self.clusters[int(key)].runCluster(temp_v) + 100)/131
                data_dict['v'+key] = temp_v
                data_dict['z'+key] = temp_z

        return data_dict

v = visualise()
v.seq_len = 512
# for j in range(1,len(v.Vtrain)):
def xyz(index):
    a = v.get_dict(index)
    st = a['v0'].reshape(v.seq_len,1)

# print(st)
    z_st = a['z0'].reshape(v.seq_len,1)

    for i in range(5):
        st = np.hstack((st,st))
        # print(st.shape)
        z_st = np.hstack((z_st,z_st))

# print(st.shape)
    for i in range(1,16):
        # temp = a['v'+str(i)].transpose(1,0).reshape(v.seq_len,1,3)
        temp = a['v'+str(i)].reshape(v.seq_len,1)

        z_temp = a['z'+str(i)].reshape(v.seq_len,1)

        for _ in range(5):
            temp = np.hstack((temp,temp))
            z_temp = np.hstack((z_temp,z_temp))
        st = np.hstack((st,temp))
        z_st = np.hstack((z_st,z_temp))

    # if j%100000==0:
    #     print(j)
    cv2.imwrite('images/v_'+str(index)+'.jpg',(st*255).astype('uint8'))
    cv2.imwrite('images/z_'+str(index)+'.jpg',(z_st*255).astype('uint8'))

    
    # cv2.imwrite('/home/KaustubRane_grp/18110146/JHU/simpleVAE/hR/'+str(j)+'.jpg',(z_st*255).astype('uint8'))

xyz(2000)
