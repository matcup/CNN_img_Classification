import tensorflow as tf
import numpy as np
from skimage import io, transform, color, data, img_as_ubyte
import glob
import os
from inference import inference
path = './101_Objectcategories'
model_path = './new_model/model.ckpt'
w = 100
h = 100
c = 3


def read_img(path):
    imgs = []
    labels = []
    cate = [path + '/' + x for x in os.listdir(path)]
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + './*.jpg'):
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)
num_example = data.shape[0]
arr = np.arange(num_example)  # arr.shape = (9144,)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[: s]
x_val = data[s:]
y_train = label[: s]
y_val = label[s:]

x = tf.placeholder(tf.float32, [None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, [None, ], name='y_')


regularizer = tf.contrib.layers.l2_regularizer(0.001)
logits = inference(x, False, regularizer)


# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些

n_epoch = 100
batch_size = 64
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss) / n_batch))
    print("   train acc: %f" % (np.sum(train_acc) / n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err
        val_acc += ac
        n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
    print("   validation acc: %f" % (np.sum(val_acc) / n_batch))
saver.save(sess, model_path)
sess.close()


