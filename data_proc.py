import os
import cv2
import random
import tensorflow as tf
import numpy as npy
from matplotlib import pyplot as plt
from evaluate_metric import sr_metric
  
class DataIterSR(object):
    def __init__(self, datadir,img_list, crop_num, crop_size, scale_factor, is_shuffle):
        self._datadir=datadir
        self._img_list=img_list
        self._crop_num=crop_num
        self._crop_size=crop_size
        self._scale_fator=scale_factor
        self._is_shuffle=is_shuffle
        self._provide_input=zip(["img_in"],[(crop_num,3, crop_size, crop_size)])
        self._provide_output=zip(["img_out"],[(crop_num,3, crop_size, crop_size)])
        self._num_img=len(img_list)
        self._cur_idx=0
        self._iter_cnt=0
        
    def reset(self):
        self._cur_idx=0
        self._iter_cnt=0
        
    def fetch_next(self):
        if self._is_shuffle and npy.mod(self._cur_idx,self._num_img)==0:
            self._cur_idx=0
            random.shuffle(self._img_list)  
        crop_size=self._crop_size
        img_path=os.path.join(self._datadir,self._img_list[self._cur_idx])
        img=cv2.imread(img_path, cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        [nrow, ncol, nchl]=img.shape
        self._iter_cnt += 1
        self._cur_idx += 1
        if nrow < crop_size or ncol < crop_size:
            raise ValueError("Crop size is larger than image size")
        img_blur=cv2.GaussianBlur(img,(3,3),1.2)
        img_struct=cv2.GaussianBlur(img_blur,(3,3),1.5)
        img_ds=cv2.resize(img_blur, (ncol//self._scale_fator, nrow//self._scale_fator),
                          interpolation=cv2.INTER_CUBIC)
        img_lr=cv2.resize(img_ds, (ncol, nrow), interpolation=cv2.INTER_CUBIC)
        img=img.astype(npy.float32)
        img_lr=img_lr.astype(npy.float32)
        img_struct=img_struct.astype(npy.float32)
        img_detail=img-img_struct
        sub_img_hr=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img_lr=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img_struct=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img_detail=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        for i in range(self._crop_num):
            nrow_start=npy.random.randint(0,nrow-crop_size)
            ncol_start=npy.random.randint(0,ncol-crop_size)
            img_crop=img_lr[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]              
            img_crop=img_crop/255.0       
            sub_img_lr[i,:,:,:]=img_crop
            
            img_crop=img[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0              
            sub_img_hr[i,:,:,:]=img_crop

            img_crop=img_struct[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0             
            sub_img_struct[i,:,:,:]=img_crop

            img_crop=img_detail[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0                   
            sub_img_detail[i,:,:,:]=img_crop

        return (sub_img_hr.astype(npy.float32),sub_img_lr.astype(npy.float32),
                sub_img_struct.astype(npy.float32),sub_img_detail.astype(npy.float32))

class DataIterEPF(object):
    def __init__(self, datadir,img_list, crop_num, crop_size, is_shuffle):
        self._datadir=datadir
        self._img_list=img_list
        self._crop_num=crop_num
        self._crop_size=crop_size
        self._is_shuffle=is_shuffle
        self._provide_input=zip(["img_in"],[(crop_num,3, crop_size, crop_size)])
        self._provide_output=zip(["img_out"],[(crop_num,3, crop_size, crop_size)])
        self._num_img=len(img_list)
        self._cur_idx=0
        self._iter_cnt=0
        
    def reset(self):
        self._cur_idx=0
        self._iter_cnt=0
        
    def fetch_next(self):
        if self._is_shuffle and npy.mod(self._cur_idx,self._num_img)==0:
            self._cur_idx=0
            random.shuffle(self._img_list) 
        crop_size=self._crop_size
        img_path1=os.path.join(self._datadir,self._img_list[self._cur_idx][0])
        img1=cv2.imread(img_path1, cv2.IMREAD_COLOR)
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        [nrow1, ncol1, nchl1]=img1.shape
        img_path2=os.path.join(self._datadir,self._img_list[self._cur_idx][1])
        img2=cv2.imread(img_path2, cv2.IMREAD_COLOR)
        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        [nrow, ncol, nchl]=img2.shape
        
        if (nrow1!=nrow) or ncol1!=ncol or nchl1 !=nchl:
            raise ValueError("Two images have different size")
     
        self._iter_cnt += 1
        self._cur_idx += 1
        if nrow < crop_size or ncol < crop_size:
            raise ValueError("Crop size is larger than image size")
        img1=img1.astype(npy.float32)
        img2=img2.astype(npy.float32)     
        sub_img1=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img2=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        for i in range(self._crop_num):
            nrow_start=npy.random.randint(0,nrow-crop_size)
            ncol_start=npy.random.randint(0,ncol-crop_size)
            img_crop=img1[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]              
            img_crop=img_crop/255.0       
            sub_img1[i,:,:,:]=img_crop
            
            img_crop=img2[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0              
            sub_img2[i,:,:,:]=img_crop

        return (sub_img1.astype(npy.float32),sub_img2.astype(npy.float32))    
 
def test_SRDataIter():
    datadir=r"_Datasets\SuperResolution\SR_training_datasets\T91"
    img_list=[f for f in os.listdir(datadir) if f.find(".png")!=-1]
    crop_num=5
    crop_size=64
    scale_factor=3
    data_iter=DataIterSR(datadir, img_list, crop_num, crop_size, scale_factor, True)
    try:
        img_hr, img_lr, img_struct, img_detail=data_iter.fetch_next()
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(img_hr[0, :,:,:])
        plt.subplot(2,2,2)
        plt.imshow(img_lr[0, :,:,:])
        plt.subplot(2,2,3)
        plt.imshow(img_struct[0, :,:,:])
        plt.subplot(2,2,4)
        plt.imshow(img_detail[0, :,:,:])
        mse, psnr=sr_metric(img_hr, img_lr)
        print("mse={}, psnr={}".format(mse, psnr))
    except ValueError:
        print("data_iter get no data")
        
def test_DataIterEPF():
    datadir=r"_Datasets\DeRaining\train\RainTrainL"
    img_list1=[f for f in os.listdir(datadir) if f.find(".png")!=-1 and f.find("norain")!=-1]
    img_list2=[f for f in os.listdir(datadir) if f.find(".png")!=-1 
               and f.find("rain")!=-1 and f.find("norain")==-1]
    img_list=[[f1,f2] for f1, f2 in zip(img_list1,img_list2)]
    crop_num=5
    crop_size=64
    data_iter=DataIterEPF(datadir, img_list, crop_num, crop_size, True)
    try:
        img_norain, img_rain=data_iter.fetch_next()
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img_norain[0, :,:,:])
        plt.subplot(1,2,2)
        plt.imshow(img_rain[0, :,:,:])
        mse, psnr=sr_metric(img_norain, img_rain)
        print("mse={}, psnr={}".format(mse, psnr))
    except ValueError:
        print("data_iter get no data")

if __name__=="__main__":
#    test_SRDataIter()
    test_DataIterEPF()
              