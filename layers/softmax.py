import numpy as np

class Softmax(object):
    def __init__(self):
        pass

    def forward(self, y_input):
        y_input -= np.max(y_input)
        y_input_exp = np.exp(y_input)
        y_output = y_input_exp / np.sum(y_input_exp)
        return y_output

    def backward(self):
        pass

