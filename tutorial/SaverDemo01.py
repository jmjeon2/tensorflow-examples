import tensorflow as tf
import numpy as np

data = np.loadtxt('test.csv',delimiter=',',unpack=True,dtype='float32')

x_data=np.transpose(data[0:2])
y_data=np.transpose(data[2:])

global_step = tf.Variable(0, trainable=True, name='global_step')#초기값0 trainable-카운트

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W1=tf.Variable(tf.random_uniform([2,10],-1.,1.))
b1=tf.Variable(tf.random_normal([10]))
L1=tf.nn.relu(tf.add(tf.matmul(X,W1),b1))

W2=tf.Variable(tf.random_uniform([10,20],-1.,1.))
b2=tf.Variable(tf.random_normal([20]))
L2=tf.nn.relu(tf.add(tf.matmul(L1,W2),b2))

W3=tf.Variable(tf.random_uniform([20,3],-1.,1.))
b3=tf.Variable(tf.random_normal([3]))
model = tf.add(tf.matmul(L2,W3),b3)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=model))

optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost,global_step=global_step)

sess=tf.Session()
saver = tf.train.Saver(tf.global_variables())#앞서 정의한 변수를 가져오는 함수

ckpt = tf.train.get_checkpoint_state('./SaverDemo01')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

for step in range(60):
    sess.run(train, feed_dict={X:x_data, Y:y_data})

    print('Step: %d, ' %sess.run(global_step),
          'Cost: %.3f, ' %sess.run(cost, feed_dict={X:x_data,Y:y_data}))

saver.save(sess, './SaverDemo01/test_model.ckpt',global_step = global_step)

prediction = tf.argmax(model,axis=1)
target = tf.argmax(Y,axis=1)
print('예측값:',sess.run(prediction,feed_dict={X:x_data}))
print('실제값:',sess.run(target,feed_dict={Y:y_data}))

is_correct=tf.equal(prediction,target)
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도:',sess.run(accuracy*100,feed_dict={X:x_data,
                                              Y:y_data}))

