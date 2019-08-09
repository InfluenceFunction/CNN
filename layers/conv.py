import numpy as np

class Conv(object):
    def __init__(self, filter, bias):
        self.filter = filter
        self.bias = bias

    #卷积操作：filter*feature_map（对应位置相乘）
    def conv_operation(self, image, fil):
        shape = image.shape
        conv_out = np.zeros((shape[0] - fil.shape[0] + 1, shape[1] - fil.shape[1] + 1))
        for i in range(conv_out.shape[0]):
            for j in range(conv_out.shape[1]):
                conv_out[i,j] = np.sum(fil*image[i:i+fil.shape[0],j:j+fil.shape[1]])
        return conv_out

    #输出卷积结果
    def forward(self,image):
        self.image = image
        shape = self.filter.shape
        feature_map = np.zeros((self.image.shape[0]-shape[1]+1,
                                self.image.shape[1]-shape[2] + 1,shape[0]))
        for img_num in range(shape[0]):
            feature_map[:,:,img_num] = \
                self.conv_operation(self.image,self.filter[img_num,:])+self.bias[img_num]
        return feature_map

    def backward(self,pool_out,delta_conv):
        delta_filter = np.zeros(self.filter.shape)
        delta_bias = np.sum(np.sum(delta_conv, axis=0), axis=0)
        if len(self.filter.shape)==4:
            for layer_num in range(delta_filter.shape[-1]):
                for filter_num in range(delta_filter.shape[0]):
                    delta_filter[filter_num,:,:,layer_num] \
                        = self.conv_operation(pool_out[:,:,layer_num],
                                              delta_conv[:,:,filter_num])
            #将delta_conv进行填充
            delta_conv_padding = np.pad(delta_conv,((self.filter.shape[1]-1,self.filter.shape[1]-1),
                                                    (self.filter.shape[2]-1,self.filter.shape[2]-1),(0,0)),
                                        'constant',constant_values=0)
            delta_pooling = np.zeros(self.image.shape)
            for map_num in range(delta_pooling.shape[-1]):
                for filter_num in range(self.filter.shape[0]):
                    delta_pooling[:,:,map_num] += \
                        self.conv_operation(delta_conv_padding[:,:,filter_num],
                                            np.rot90(self.filter[filter_num,:,:,map_num],2))
            self.bias -= 0.006 * delta_bias
            self.filter -= 0.006 * delta_filter
            return delta_pooling

        elif len(self.filter.shape)==3:
            for filter_num in range(delta_filter.shape[0]):
                delta_filter[filter_num,:,:] \
                    = self.conv_operation(self.image,delta_conv[:,:,filter_num])
            # return delta_filter, delta_bias
            self.bias -= 0.0013 * delta_bias
            self.filter -= 0.0013 * delta_filter


def mean_squared_loss(y_predict, y_true):
    """
    均方误差损失函数
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值
    :return:
    """
    loss = np.mean(np.sum(np.square(y_predict - y_true), axis=-1))  # 损失函数值
    dy = y_predict - y_true  # 损失函数关于网络输出的梯度
    return loss, dy


def main():
    img = np.random.randn(13, 13, 2)
    w = np.random.randn(3, 2, 2, 2)
    y_true = np.ones((12, 12, 3))
    b = np.zeros(3)
    conv = Conv(w,b)
    for i in range(1000):
        # 前向
        next_z = conv.forward(img)

        # 反向
        loss, dy = mean_squared_loss(next_z, y_true)
        _, dw, db = conv.backward(img, dy)
        w -= 0.006 * dw
        b -= 0.005* db
        print(b)
        # 打印损失
        print("step:{},loss:{}".format(i, loss))

        if np.allclose(y_true, next_z):
            print("yes")
            break


if __name__ == '__main__':
    main()

