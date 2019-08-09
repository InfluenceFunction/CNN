import numpy as np

class Relu(object):

    def forward(self,feature_map):
        self.feature_map = feature_map
        relu_out = np.zeros(self.feature_map.shape)
        for map_num in range(self.feature_map.shape[-1]):
            for i in range(self.feature_map.shape[0]):
                for j in range(self.feature_map.shape[1]):
                    relu_out[i, j,map_num] = np.max(self.feature_map[i, j,map_num], 0)
        return relu_out

    def backward(self):
        pass