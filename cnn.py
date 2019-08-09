import numpy as np
import pandas as pd
from CNN.layers import conv, pooling, relu, fc

#feature_map = [length,width,map_num]
#filter = [filter_num,length,width,signle_filter_num]
#train_x = [image_num,image_pixel]
#train_y = [data_num,]

#读取训练数据train_x, train_y
mnist_train = pd.read_csv("D:\\文档文件\\机器学习\\机器学习数据集\\mnist_dataset\\mnist_train_100.csv",header=None)
mnist_test = pd.read_csv("D:\\文档文件\\机器学习\\机器学习数据集\\mnist_dataset\\mnist_test_10.csv",header=None)
train_x = np.array(mnist_train.values[:,1:785])
test_x = np.array(mnist_test.values[:,1:785])
y_train = np.array(mnist_train.values[:,0])
y_test = np.array(mnist_test.values[:,0])
def translate(y):
  train_y = np.zeros([y.shape[0],10])
  for i in range(y.shape[0]):
    for j in range(10):
      if y[i]==j:
        train_y[i][j] = 1
  return train_y
train_y = translate(y_train)
test_y=  translate(y_test)


#初始化l1_filter,l1_conv,l1_relu,l1_pooling
l1_filter = np.random.rand(2,3,3)
l1_bias = np.zeros(2)
l1_conv = conv.Conv(l1_filter, l1_bias)
l1_relu = relu.Relu()
l1_pooling = pooling.MaxPooling(2, 2)


#第二层：卷积、Relu激活、池化
#初始化l2_filter,l2_conv,l2_relu,l2_pooling
l2_filter = np.random.rand(3,2,2,l1_filter.shape[0])
l2_bias = np.zeros(3)
l2_conv = conv.Conv(l2_filter, l2_bias)
l2_relu = relu.Relu()
l2_pooling = pooling.MaxPooling(2, 2)


#全连接层：输入层、隐藏层、输出层（sigmoid激活）
#初始化全连接层fc
fc = fc.FullConnect(108, 1000, 10, 0.005)


#训练数据
for around in range(100):
  correct_rate = 0.0
  for i in range(train_x.shape[0]):

    #前向传播
    #输入数据归一化
    image = (train_x[i]/255).reshape(28,28)
    l1_feature_map = l1_conv.forward(image)
    l1_feature_map_relu = l1_relu.forward(l1_feature_map)
    l1_feature_map_relu_pool = l1_pooling.forward(l1_feature_map_relu)

    l2_feature_map = l2_conv.forward(l1_feature_map_relu_pool)
    l2_feature_map_relu = l2_relu.forward(l2_feature_map)
    l2_feature_map_relu_pool = l2_pooling.forward(l2_feature_map_relu)

    fc_input = l2_feature_map_relu_pool.reshape(1, -1)
    fc_output = fc.forward(fc_input)
    y = np.argmax(fc_output, axis=1)
    if y==y_train[i]:
      correct_rate += 1

    #反向传播
    #两层卷积层
    delta_flatten = fc.backward(train_y[i].reshape(1,-1))
    l2_delta_pooling = delta_flatten.reshape(l2_feature_map_relu_pool.shape)
    l2_delta_conv = l2_pooling.backward(l2_delta_pooling)
    l1_delta_pooling = l2_conv.backward(l1_feature_map_relu_pool,l2_delta_conv)
    l1_delta_conv = l1_pooling.backward(l1_delta_pooling)
    #一层卷积层
    # l1_delta_pooling = delta_flatten.reshape(l1_feature_map_relu_pool.shape)
    # l1_delta_conv = l1_pooling.backward(l1_delta_pooling)
    l1_conv.backward(train_x[i].reshape(28,28),l1_delta_conv)
  print('预测正确率：',correct_rate/100)

#测试数据
# for i in range(test_x.shape[0]):

  # l1_feature_map = l1_conv.forward(train_x[i].reshape(28, 28))
  # l1_feature_map_relu = l1_relu.forward(l1_feature_map)
  # l1_feature_map_relu_pool = l1_pooling.forward(l1_feature_map_relu)
  #
  # l2_feature_map = l2_conv.forward(l1_feature_map_relu_pool)
  # l2_feature_map_relu = l2_relu.forward(l2_feature_map)
  # l2_feature_map_relu_pool = l2_pooling.forward(l2_feature_map_relu)
  #
  # fc_input = l2_feature_map_relu_pool.reshape(1, -1)
  # fc_input = (test_x[i] / 255).reshape(1, -1)
  # fc_output = fc.forward(fc_input)
  # y = np.argmax(fc_output, axis=1)
  # print(y)