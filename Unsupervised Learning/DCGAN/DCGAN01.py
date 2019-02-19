
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../../mnist/data/',one_hot=True)


# In[2]:


training_epoch=100
batch_size=100
n_noise=100
n_class=10

D_global_step = tf.Variable(0, trainable=False, name='D_global_step')
G_global_step = tf.Variable(0, trainable=False, name='G_global_step')
 


# In[3]:


X=tf.placeholder(tf.float32, [None,28,28,1])
Z=tf.placeholder(tf.float32, [None,n_noise])
is_training = tf.placeholder(tf.bool)
 
def leaky_relu(x, leak=0.2):
    return tf.maximum(x, x * leak)

def generator(noise):
    with tf.variable_scope('generator'):
        output = tf.layers.dense(noise, 128*7*7)
        output = tf.reshape(output, [-1, 7, 7, 128])
        output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        output = tf.layers.conv2d_transpose(output, 64, [5, 5], strides=(2, 2), padding='SAME')
        output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        output = tf.layers.conv2d_transpose(output, 32, [5, 5], strides=(2, 2), padding='SAME')
        output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        output = tf.layers.conv2d_transpose(output, 1, [5, 5], strides=(1, 1), padding='SAME')
        output = tf.tanh(output)
    return output

def discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        output = tf.layers.conv2d(inputs, 32, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)
        output = tf.layers.conv2d(output, 64, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(tf.layers.batch_normalization(output, training=is_training))
        output = tf.layers.conv2d(output, 128, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(tf.layers.batch_normalization(output, training=is_training))
        output = tf.layers.flatten(output)
        output = tf.layers.dense(output, 1, activation=None)
    return output
    
def get_noise(batch_size, n_noise):
    return np.random.uniform(-1.,1.,size=(batch_size,n_noise))

def get_moving_noise(batch_size, n_noise):
    assert batch_size > 0
 
    noise_list = []
    base_noise = np.random.uniform(-1.0, 1.0, size=[n_noise])
    end_noise = np.random.uniform(-1.0, 1.0, size=[n_noise])
 
    step = (end_noise - base_noise) / batch_size
    noise = np.copy(base_noise)
    for _ in range(batch_size - 1):
        noise_list.append(noise)
        noise = noise + step
    noise_list.append(end_noise)
    
    return noise_list


# In[4]:


G=generator(Z)
D_real = discriminator(X)
D_gene = discriminator(G, True)

loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_gene, labels=tf.zeros_like(D_gene)))
loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_real, labels=tf.ones_like(D_real)))
                                                      
loss_D = loss_D_gene+loss_D_real                                                      
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_gene, labels=tf.ones_like(D_gene)))

vars_D=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
vars_G=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_D = tf.train.AdamOptimizer().minimize(loss_D,
        var_list=vars_D, global_step=D_global_step)
    train_G = tf.train.AdamOptimizer().minimize(loss_G,
        var_list=vars_G, global_step=G_global_step)


# In[6]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_batch=int(mnist.train.num_examples/batch_size)
    loss_val_D, loss_val_G =0, 0

    print('Learning Start')

    for epoch in range(training_epoch):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape(-1,28,28,1)
            noise = get_noise(batch_size,n_noise)

            _,loss_val_D = sess.run([train_D,loss_D],
                                    feed_dict={X:batch_xs, Z:noise, is_training:True})
            _,loss_val_G = sess.run([train_G,loss_G],
                                    feed_dict={X:batch_xs, Z:noise, is_training:True})

        print('Epoch','%04d'%(epoch+1), 
              'loss_D:{:.4}'.format(loss_val_D),
              'loss_G:{:.4}'.format(loss_val_G))
        if epoch != -1 :
            sample_size = 10
            noise = get_noise(sample_size, n_noise)
            samples = sess.run(G, 
                               feed_dict = {Z:noise, is_training:False})
            test_noise = get_moving_noise(sample_size, n_noise)
            test_samples = sess.run(G, feed_dict={Z: test_noise, is_training: False})

            fig, ax = plt.subplots(2, sample_size, figsize = (sample_size,2))

            for i in range(sample_size):
                ax[0][i].set_axis_off()
                ax[1][i].set_axis_off()
                ax[0][i].imshow(np.reshape(samples[i],(28,28)))
                ax[1][i].imshow(np.reshape(test_samples[i],(28,28)))

            plt.savefig('./result/{}.png'.format(str(epoch).zfill(3)),
                            bbox_inches='tight')
            plt.close(fig)
    print('Learning Finished')


# In[ ]:


def one_hot(i):
    return sess.run(tf.one_hot([i],10))

def sample_show(i):
    a=sess.run(tf.one_hot([i],10))
    for k in range(9):
        a=np.append(a,one_hot(i),axis=0)
        
    sample = sess.run(G,feed_dict={Y:a,Z:get_noise(10,128)})
    fig, ax = plt.subplots(1, 10, figsize=(10, 1))
    for k in range(10):
            ax[k].set_axis_off()
            ax[k].imshow(np.reshape(sample[k],(28,28)))
    
    plt.show()

sample_show(3)

