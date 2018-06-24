import tensorflow as tf
import tensorlayer as tl


def StructNet(input_data):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    net=tl.layers.conv2d(input_data,n_filter=64,filter_size=(9,9), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netS_conv0")
    net=tl.layers.conv2d(net,n_filter=32,filter_size=(1,1),strides=(1,1),
                         act=tf.nn.relu,W_init=W_init,name="netS_conv1")
    net=tl.layers.conv2d(net,n_filter=3,filter_size=(5,5),strides=(1,1),
                         act=tf.nn.relu,W_init=W_init,name="netS_conv2")
    return net
    
def DetailNet(input_data):
     W_init = tf.truncated_normal_initializer(stddev=5e-2)
     net=tl.layers.conv2d(input_data,n_filter=64,filter_size=(5,5), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_conv0")
     for i in range(16):
         net=tl.layers.conv2d(net,n_filter=64,filter_size=(3,3), strides=(1,1),
                              act=tf.nn.relu,W_init=W_init,name="netD_conv{}".format(i+1))
     net=tl.layers.conv2d(net,n_filter=32,filter_size=(1,1), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_convout1")
     net=tl.layers.conv2d(net,n_filter=3,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_convout")
     return net
 
def DualCNN(x):
    endpoints={}
    net=tl.layers.InputLayer(x,name="img_in")
    netD=DetailNet(net)
    netS=StructNet(net)

    net=tl.layers.ElementwiseLayer([netD, netS], tf.add,name="sumDS")
    endpoints["compS"]=netS
    endpoints["compD"]=netD
    return net, endpoints
    
    
if __name__=="__main__":
    tl.layers.clear_layers_name()
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    x=tf.placeholder(tf.float32,shape=[None,64,64,3],name="x")
    y=tf.placeholder(tf.float32,shape=[None,64,64,3],name="y")
    net,endpoints=DualCNN(x)
    tl.layers.initialize_global_variables(sess)
#    print(tf.shape(net))
    net.print_params()
    net.print_layers()
    
  