import numpy as npy
#import cv2

def sr_metric(hr_imgs, sr_imgs):
    sr_shape=sr_imgs.shape
    hr_shape=hr_imgs.shape
    rs=(hr_shape[1]-sr_shape[1])//2
    cs=(hr_shape[2]-sr_shape[2])//2
    hr_imgs=hr_imgs*255
    sr_imgs=sr_imgs*255
    
    hr_center_imgs=hr_imgs[:,rs:rs+sr_shape[1],cs:cs+sr_shape[2],:]

    psnr=npy.zeros(hr_shape[0],dtype=npy.float32)
    mse=npy.zeros(hr_shape[0],dtype=npy.float32)
    for i in range(hr_shape[0]):
        diff=hr_center_imgs[i,:,:,:]-sr_imgs[i,:,:,:]       
        mse[i]=npy.mean(diff*diff)
        psnr[i]=10*npy.log10(255*255/mse[i])
#    print psnr
    return (npy.mean(npy.sqrt(mse)),npy.mean(psnr))
    
         
    

        

