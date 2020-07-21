import numpy as np
from collections import OrderedDict

# Template class of layers
class Layer(object):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass
    
    def backward(self):
        pass

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Layer,self).__init__()
        self.w = np.random.randn(in_features,out_features)
        self.has_bias = bias
        self.b = np.random.randn(out_features)
            
        # Initialize gradients to zeros --Seems like negligible--
        # self.grad = np.zeros(self.w.size)
        
        # Record input & output vector
        self.x = None
        self.out = None
        
    def forward(self, x):
        '''
            Input:
                x: np.array
        '''        
        self.x = x
        
        # Calculate output results
        if self.has_bias is True:
            # Output = XW + b
            self.out = np.dot(self.x,self.w) + self.b
        else:
            # Output = XW
            self.out = np.dot(self.x,self.w)
        
        return self.out
    
    def backward(self, prev_grad, lr = 0.1):
        '''
            Input:
                prev_grad: np.array that comes from "next" layers
                lr : Learning rate
        '''
        # Calculate gradient of w, and update it with (learning rate*gradient)
        dw = np.dot(self.x.T,prev_grad)
        self.w -= lr * dw
        
        # Update bias if needed. Gradient of bias is the element-sum of prev_grad 
        if self.has_bias is True:
            db = np.sum(prev_grad,axis=0)
            self.b -= lr * db   
        
        # Return gradient that prpogate to next layer
        return np.dot(prev_grad,self.w.T)
    
class Sigmoid(Layer):
    def __init__(self):
        self.y = None
        
    def forward(self, x):
        self.y = 1.0 /(1.0 + np.exp(-x))
        return self.y
    
    def backward(self, prev_grad,lr=0.1):
        # Return prev_grad * derivative of sigmoid func.
        return prev_grad * np.multiply((1.0 - self.y) , self.y)
    
class ReLU(Layer):
    def __init__(self):
        self.mask = None
        self.y = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        self.y = x.copy()
        self.y[self.mask] = 0
        return self.y
    
    def backward(self, prev_grad,lr=0.1):
        # Return prev_grad * derivative of sigmoid func.
        grad = prev_grad
        grad[self.mask] = 0
        return grad
    
class MSE(Layer):
    def __init__(self):
        self.y = None
        self.ground_truth = None
    
    def forward(self, y, ground_truth):
        self.y = y
        self.ground_truth = ground_truth
        return np.mean((self.y-self.ground_truth)**2)
    
    def backward(self,prev_grad=1,lr=0.1):
        '''
            prev_grad, lr: pseudo parameters
        '''
        return (2/self.y.shape[0])*(self.y-self.ground_truth)
    
class Binary_Cross_Entropy(Layer):    
    def __init__(self):
        self.y = None
        self.ground_truth = None
    
    def forward(self, y, ground_truth):
        # 使此 function 可以處理 batch ,也可以處理 single data
        if y.ndim == 1:
            ground_truth = ground_truth.reshape(1, ground_truth.size)
            y = y.reshape(1, y.size)
            
        batch_size = y.shape[0]
        self.y = y
        self.ground_truth = ground_truth
        # To avoid -inf; 重要！！
        delta = 1e-7 
        
        return -np.sum(self.ground_truth * np.log2(self.y + delta) + (1-self.ground_truth)* np.log2(1 - self.y + delta) ) / batch_size
    
    def backward(self,prev_grad=1,lr=0.1):
        '''
            prev_grad, lr: pseudo parameters
        '''
        batch_size = self.y.shape[0]
        dx = (self.y - self.ground_truth) / (self.y * (1 - self.y) * batch_size)
        return dx
    
    
class TLNN(object):
    def __init__(self, layer_1_units = 4, layer_2_units = 4, bias= True):
        self.layers = OrderedDict()
        self.layers['linear_1'] = Linear(2,layer_1_units,bias=bias)
        self.layers['ReLU_1'] = ReLU()
        #self.layers['sigmoid_1'] = Sigmoid()
        self.layers['linear_2'] = Linear(layer_1_units,layer_2_units,bias=bias)
        self.layers['ReLU_2'] = ReLU()
        #self.layers['sigmoid_2'] = Sigmoid()
        self.layers['output'] = Linear(layer_2_units,1,bias = False)
        #self.layers['ReLU_3'] = ReLU()
        self.layers['sigmoid_3'] = Sigmoid()
        
        #self.loss_func = Binary_Cross_Entropy()
        self.loss_func = MSE()
        
    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def cal_loss(self, y, ground_truth):
        return self.loss_func.forward(y,ground_truth)
    
    def backward(self,lr=0.05):
        dy = self.loss_func.backward()
        
        # Reverse the layers list for easily conducting backward
        back_layers = list(self.layers.values())
        back_layers.reverse()
        for layer in back_layers:
            dy = layer.backward(dy,lr=lr)