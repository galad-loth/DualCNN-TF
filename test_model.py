import os
import time
import numpy as npy
import cv2
import tensorflow as tf
import tensorlayer as tl
from model import DualCNN

from evaluate_metric import sr_metric

def test_SR():
    datadir=r"data\test\sr"
    img_list=[f for f in os.listdir(datadir) if f.find(".png")!=-1]
    scale_factor=4

    check_point_dir=r"checkpoint\sr"
    res_dir="test_result\sr"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    x=tf.placeholder(tf.float32,shape=[None,None,None,3],name="x")
    net,endpoints=DualCNN(x)
    y_out=net.outputs

    saver=tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess,os.path.join(check_point_dir,"model_{}.ckpt".format(50000)))
        for f in img_list:
            img=cv2.imread(os.path.join(datadir, f), cv2.IMREAD_COLOR)
            [nrow, ncol, nchl]=img.shape
            img_blur=cv2.GaussianBlur(img,(3,3),1.2)
            img_ds=cv2.resize(img_blur, (ncol//scale_factor, nrow//scale_factor),
                              interpolation=cv2.INTER_CUBIC)
            img_lr=cv2.resize(img_ds, (ncol, nrow), interpolation=cv2.INTER_CUBIC)
            img_in=(img_lr.astype(npy.float32))/255.0
            img_in=img_in[npy.newaxis,:,:,:].astype(npy.float32)
            y_pred=sess.run(y_out, feed_dict={x:img_in})
            img_out=npy.maximum(0, npy.minimum(1,y_pred[0,:,:,:]))*255
            img_out=img_out.astype(npy.uint8)
            cv2.imwrite(os.path.join(res_dir, f+"_imglr.png"),img_lr)
            cv2.imwrite(os.path.join(res_dir, f+"_imgsr.png"),img_out)

def test_EPF():
        
#    datadir=r"data\test\derain"
#    img_list=[f for f in os.listdir(datadir) if f.find(".png")!=-1]
#    check_point_dir=r"checkpoint\derain"
#    res_dir="test_result\derain"

    datadir=r"data\test\epf"
    img_list=[f for f in os.listdir(datadir) if f.find(".jpg")!=-1]

    check_point_dir=r"checkpoint\rtv"
    res_dir="test_result\rtv"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    x=tf.placeholder(tf.float32,shape=[None,None,None,3],name="x")
    net,endpoints=DualCNN(x)
    y_out=net.outputs

    saver=tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess,os.path.join(check_point_dir,"model_{}.ckpt".format(20000)))
        for f in img_list:
            img=cv2.imread(os.path.join(datadir, f), cv2.IMREAD_COLOR)
            [nrow, ncol, nchl]=img.shape
            img_in=(img.astype(npy.float32))/255.0
            img_in=img_in[npy.newaxis,:,:,:].astype(npy.float32)
            y_pred=sess.run(y_out, feed_dict={x:img_in})
            img_out=npy.maximum(0, npy.minimum(1,y_pred[0,:,:,:]))*255
            img_out=img_out.astype(npy.uint8)
            cv2.imwrite(os.path.join(res_dir, f+"_derain.png"),img_out)

        
if __name__=="__main__":
    tl.layers.clear_layers_name()
    tf.reset_default_graph() 
#    test_SR()
    test_EPF()        
    