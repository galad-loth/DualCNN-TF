import os
import numpy as npy
import tensorflow as tf
import tensorlayer as tl
from model import DualCNN
from evaluate_metric import sr_metric
from data_proc import DataIterEPF

tl.layers.clear_layers_name()
tf.reset_default_graph()

#datadir=r"_Datasets\DeRaining\train\RainTrainL"
#img_list1=[f for f in os.listdir(datadir) if f.find(".png")!=-1 and f.find("norain")!=-1]
#img_list2=[f for f in os.listdir(datadir) if f.find(".png")!=-1 
#           and f.find("rain")!=-1 and f.find("norain")==-1]
#img_list=[[f1,f2] for f1, f2 in zip(img_list1,img_list2)]
datadir=r"data\epf"
img_list1=[os.path.join("BSDS200_RTV",f) for f in os.listdir(os.path.join(datadir,"BSDS200_RTV")) if f.find(".png")!=-1] 
img_list2=[os.path.join("BSDS200",f) for f in os.listdir(os.path.join(datadir,"BSDS200")) if f.find(".png")!=-1]
img_list=[[f1,f2] for f1, f2 in zip(img_list1,img_list2)] 


num_epoch=100000
batch_size=10
train_img_size=41
lr0=0.0001
print_freq=200
save_freq=5000
check_point_dir=r"checkpoint\epf"
if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)

x=tf.placeholder(tf.float32,shape=[None,train_img_size,train_img_size,3],name="x")
y=tf.placeholder(tf.float32,shape=[None,train_img_size,train_img_size,3],name="y")
yd=tf.placeholder(tf.float32,shape=[None,train_img_size,train_img_size,3],name="yd")
net,endpoints=DualCNN(x)

y_out=net.outputs
y_detail=endpoints["compD"].outputs
cost=tl.cost.mean_squared_error(y,y_out, name="cost_all")
cost=cost+0.001*tl.cost.mean_squared_error(yd,y_detail, name="cost_d")

l2_regular_loss = 0
for w in tl.layers.get_variables_with_name('W_conv2d', train_only=True, printable=False):#[-3:]:
    l2_regular_loss += tf.contrib.layers.l2_regularizer(1e-3)(w)
#cost=cost+tf.contrib.layers.l2_regularizer(0.001)(net.all_params)
cost=cost+l2_regular_loss

global_step = tf.Variable(0)
lr = tf.train.exponential_decay(lr0, global_step, 100, 0.96, staircase=True) 
train_op=tf.train.AdamOptimizer(lr).minimize(cost)

data_iter=DataIterEPF(datadir, img_list, batch_size, train_img_size, True)
saver=tf.train.Saver()

with tf.Session() as sess:
    tl.layers.initialize_global_variables(sess)
    net.print_params()
    net.print_layers()
    mean_loss=0
    mean_mse=0
    mean_psnr=0
    epoch_cnt=0
    for epoch in range(num_epoch):
        img_norain, img_rain=data_iter.fetch_next()
        train_loss,y_pred,_=sess.run([cost,y_out,train_op],
                                     feed_dict={x:img_rain, y:img_norain,yd:img_norain})
        mse, psnr=sr_metric(img_norain,y_pred)
        mean_loss+=train_loss
        mean_mse+=mse
        mean_psnr+=psnr
        epoch_cnt+=1
        if npy.mod(epoch,print_freq)==0:
            print("Epoch:{},train_loss:{}, mse:{}, psnr:{}".format(
                  epoch, mean_loss/epoch_cnt,mean_mse/epoch_cnt, mean_psnr/epoch_cnt))
            mean_loss=0
            mean_mse=0
            mean_psnr=0
            epoch_cnt=0
        if epoch>0 and npy.mod(epoch,save_freq)==0:
            print("Saving model at epoch {}".format(epoch))
            saver.save(sess,os.path.join(check_point_dir,"model_{}.ckpt".format(epoch)))
        
        
        
        



