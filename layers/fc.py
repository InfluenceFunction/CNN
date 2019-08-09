import numpy as np

class FullConnect():

    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):

        #初始化输入数据：输入层神经元个数、隐藏层神经元个数、输出层神经元个数、学习率
        self.input = input_nodes
        self.hidden = hidden_nodes
        self.output = output_nodes
        self.lr = learning_rate

        # 输入层和隐藏层之间的权重            #高斯分布的均值           #标准差                #大小
        self.weight_i_h = np.random.normal(0.0,             pow(self.hidden,- 0.5),(self.input,self.hidden))
        # 隐藏层和输出层之间的权重
        self.weight_h_o = np.random.normal(0.0,             pow(self.output,- 0.5),(self.hidden,self.output))
        # sigmoid激活函数
        self.sigmoid = lambda x: 1.0/(1 + np.exp(-x*1.0))

    def forward(self, input_data):
        self.input_data = input_data
        #计算隐藏层输入
        hidden_input = np.dot(self.input_data, self.weight_i_h)
         #计算隐藏层输出
        self.hidden_output = self.sigmoid(hidden_input)
        #计算输出层输入
        final_input = np.dot(self.hidden_output, self.weight_h_o)
        #计算输出层输出
        self.final_output = self.sigmoid(final_input)

        return self.final_output

    def backward(self,target):

        #计算在输出层的损失
        delta_h_o = (target -self.final_output) * self.final_output * (1-self.final_output)
        #计算在隐藏层的损失
        delta_i_h = delta_h_o.dot(self.weight_h_o.T) * self.hidden_output * (1-self.hidden_output)
        #计算输入层(展平层flatten)的损失
        delta_flatten = delta_i_h.dot(self.weight_i_h.T)
        #隐藏层_输出层权重更新
        delta_weight_h_o = self.lr * self.hidden_output.T.dot(delta_h_o)
        self.weight_h_o += delta_weight_h_o
        #输入层_隐藏层权重更新
        delta_weight_i_h = self.lr * self.input_data.T.dot(delta_i_h)
        self.weight_i_h += delta_weight_i_h

        return delta_flatten


