import numpy as np

class MaxPooling(object):
    def __init__(self, size=2.0, stride=2.0):
        self.size = size
        self.stride = stride

    def forward(self,feature_map):
        self.feature_map = feature_map
        shape = self.feature_map.shape
        #池化层输出
        self.pool_out = np.zeros((np.uint16((shape[0]-self.size)/self.stride+1),
                             np.uint16((shape[1]-self.size)/self.stride+1),shape[-1]))
        #记录每次池化滤波器的最大值的位置
        self.pool_max_loc = np.zeros((np.uint16((shape[0]-self.size)/self.stride+1),
                             np.uint16(2*((shape[1]-self.size)/self.stride+1)),shape[-1]))

        for map_num in range(shape[-1]):
            for i in range(self.pool_out.shape[0]):
                for j in range(self.pool_out.shape[1]):
                    #池化滤波器
                    pool_filter = self.feature_map[2*i:2*(i+1),2*j:2*(j+1),map_num]
                    #池化层输出
                    self.pool_out[i,j,map_num] = np.max(pool_filter)
                    #池化滤波器最大值位置(加上相对位置)
                    location = np.array(np.where(pool_filter == np.max(pool_filter)))
                    self.pool_max_loc[i,2*j:2*(j+1),map_num] = location[:,0]+[2*i,2*j]
        return self.pool_out

    def backward(self,delta_pooling):
        shape = self.feature_map.shape
        #卷积层损失
        delta_conv = np.zeros((shape))
        for map_num in range(shape[-1]):
            for i in range(delta_pooling.shape[0]):
                for j in range(delta_pooling.shape[1]):
                    #将delta返回到feature_map对应位置
                    delta_conv[int(self.pool_max_loc[i,2*j,map_num]),int(self.pool_max_loc[i,2*j+1,map_num]),map_num] \
                        = delta_pooling[i,j,map_num]
        return delta_conv


def main():
    image = np.random.rand(12,12,3)
    delta_pooling = np.random.rand(6,6,3)
    pool = MaxPooling()
    pool_out, pool_max_loc = pool.forward(image)
    delta_conv = pool.backward(delta_pooling)
    print(delta_conv)


if __name__ == '__main__':
    main()