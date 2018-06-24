import os
import numpy as npy
import tensorflow as tf
import tensorlayer as tl
from model import DualCNN
from evaluate_metric import sr_metric
from data_proc import DataIterSR

tl.layers.clear_layers_name()
tf.reset_default_graph()

datadir=r"_Datasets\SuperResolution\SR_training_datasets\T91"
img_list=[f for f in os.listdir(datadir) if f.find(".png")!=-1]

scale_factor=4
num_epoch=100000
batch_size=10
train_img_size=41
lr0=0.0001
print_freq=200
save_freq=5000
check_point_dir="checkpoint"
if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)

x=tf.placeholder(tf.float32,shape=[None,train_img_size,train_img_size,3],name="x")
y=tf.placeholder(tf.float32,shape=[None,train_img_size,train_img_size,3],name="y")
ys=tf.placeholder(tf.float32,shape=[None,train_img_size,train_img_size,3],name="ys")
yd=tf.placeholder(tf.float32,shape=[None,train_img_size,train_img_size,3],name="yd")
net,endpoints=DualCNN(x)

y_out=net.outputs
y_struct=endpoints["compS"].outputs
y_detail=endpoints["compD"].outputs
cost=tl.cost.mean_squared_error(y,y_out, name="cost_all")
cost=cost+0.001*tl.cost.mean_squared_error(ys,y_struct, name="cost_s")
cost=cost+0.01*tl.cost.mean_squared_error(yd,y_detail, name="cost_d")

l2_regular_loss = 0
for w in tl.layers.get_variables_with_name('W_conv2d', train_only=True, printable=False):#[-3:]:
    l2_regular_loss += tf.contrib.layers.l2_regularizer(1e-3)(w)
#cost=cost+tf.contrib.layers.l2_regularizer(0.001)(net.all_params)
cost=cost+l2_regular_loss

global_step = tf.Variable(0)
lr = tf.train.exponential_decay(lr0, global_step, 100, 0.96, staircase=True) 
train_op=tf.train.AdamOptimizer(lr).minimize(cost)

data_iter=DataIterSR(datadir, img_list, batch_size, train_img_size, scale_factor, True)
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
        img_hr, img_lr, img_struct, img_detail=data_iter.fetch_next()
        train_loss,y_pred,_=sess.run([cost,y_out,train_op],
                                     feed_dict={x:img_lr, y:img_hr,
                                                ys:img_struct, yd:img_detail})
        mse, psnr=sr_metric(img_hr,y_pred)
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
        
        
        
        



