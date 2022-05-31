# use packages from standard python library and some functions from torch
from torch import empty
from torch.nn.functional import unfold, fold
import math
import random
from pathlib import Path
import pickle

class Module(object) :
    """
    Base Module 
    """
    def forward (self, *input):
        """
        forward pass
        Args:
            *input: take the ouput from the previous layer as input
        Returns:
            output will be send to next layer as its input
        """
        raise NotImplementedError
        
    def backward (self, *gradwrtoutput):
        """
        backward pass
        Args:
            *gradwrtoutput:  gradient wrt the output variables of this layer in the forward pass
        Returns:
            gradient wrt the input variables of this layer in the forward pass
        """
        raise NotImplementedError
    
    def param(self):
        return []
    
    def __call__(self, *input):
        """
        Magic method so that `Module(input)` runs a forward pass
        """
        return self.forward(*input)
    
class Conv2d(Module):  
    """
    A Convolution layer
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size:None, stride=1, padding=0, dilation=1):
        """
        Parameters:
            in_channels (int): input tensor channel size
            out_features (int): output tensor channel size
            kernel_size (int, int): (height, width) of kernel
            stride (int): controls the stride for the move of kernel (default: 1)
            padding (int): controls the amount of padding applied to the input (default: 0)
            dilation (int): spacing between kernel elements (default: 1)
        """
        # parameter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # kernel_size
        if type(kernel_size)==int:
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
        
        # forward
        self.input = None
        self.unfolded = None 
        self.w_unfolded = None  
        
        # backward
        self.col = None    
        self.col_W = None
        
        # initialize weight and bias with xaiver uniform distribution
        lim = 1 / (self.in_channels)**0.5
        self.weight = empty((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])).uniform_(-lim,lim)
        self.bias = empty((self.out_channels)).uniform_(-lim,lim)
        
        # initialize gradients to 0
        self.dweight = empty((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])).fill_(0)
        self.dbias = empty(self.out_channels).fill_(0)
        
        
    def forward(self, input):
        """
        use unfold to forward the convolution
        """
        # output shape
        self.input = input
        N, C, H, W = input.shape  
        out_h = 1 + int((H + 2*self.padding - self.dilation*(self.kernel_size[0]-1) - 1 ) / self.stride)
        out_w = 1 + int((W + 2*self.padding - self.dilation*(self.kernel_size[1]-1) - 1 ) / self.stride)
        
        # use unfold to forward the convolution
        self.unfolded = unfold(input, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        self.w_unfolded = self.weight.view(self.out_channels, -1)
        output = (self.w_unfolded @ self.unfolded + self.bias.view(1, -1, 1)).view(N, self.out_channels , out_h, out_w)
        
        return output
    
    def backward(self, dout):
        """
        use fold to backward the convolution
        """
        # input and output shape
        N, out_c, out_h, out_w = dout.shape
        N, C, H, W = self.input.shape
        
        # gradient of bias
        db = dout.sum(axis=(0,2,3))
        self.dbias.copy_(db)
        
        # gradient of weight
        self.col = self.unfolded.transpose(1,2).reshape(-1, self.in_channels*self.kernel_size[0]*self.kernel_size[1])
        self.col_W = self.w_unfolded.T
        dout = dout.transpose(1,2).transpose(2,3).reshape(-1, self.out_channels)
        dW = self.col.T @ dout
        dW = dW.transpose(1, 0).reshape(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.dweight.copy_(dW)
        
        # gradient of input
        dcol = dout @ self.col_W.T
        col_input = dcol.reshape(N, out_h*out_w, -1).transpose(1,2)
        gradwrtinput = fold(col_input, output_size=(H,W), kernel_size=self.kernel_size, padding=self.padding ,stride=self.stride, dilation=self.dilation)
        
        return gradwrtinput
        
    def param(self):
        # return the parameters
        return [[self.weight, self.dweight], [self.bias, self.dbias]]
    
    
    
class NearestUpsampling(Module):
    """
    A NearestUpsampling layer
    """
    def __init__(self,scale_factor:None):
        """
        Parameters:
            scale_factor (int, int): scale factor along the (height, width) of input tensor
        """
        self.input = None
        self.scale_factor = scale_factor
        
        if type(scale_factor)==int:
            self.sh = scale_factor
            self.sw = scale_factor
        else:
            self.sh = scale_factor[0]
            self.sw = scale_factor[1]
        
    def forward(self, input):
        """
        upsamling the input by 'nearest neighbor' mode
        """
        self.input = input
        return input.repeat_interleave(self.sh, dim=2).repeat_interleave(self.sw, dim=3)
        
    def backward(self, gradwrtoutput):
        """
        aggregate the loss for each input element
        """
        N,C,H,W = self.input.shape
        gradwrtinput = gradwrtoutput.reshape(N, C, H, self.sh, W, self.sw).sum(axis=(3, 5), keepdims=True).reshape(self.input.shape)
        return gradwrtinput

    
class Upsampling(Module):
    """
    A upsampling layer to integrate NearestUpsampling and Conv2d
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size:None, dilation=1, padding=0, scale_factor=None):
        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.scale_factor = scale_factor
        self.stride = 1
        
        # layers
        self.conv = Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation)
        self.upsampling  = NearestUpsampling(self.scale_factor)
        
        # weight and bias
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        
    def forward(self, x):
        output_upsample = self.upsampling(x)
        output_conv = self.conv(output_upsample)
        
        return output_conv
    
    def backward(self, dout):
        grad_conv = self.conv.backward(dout)
        grad_upsample = self.upsampling.backward(grad_conv)
        
        return grad_upsample
    
    def param(self):
        return self.conv.param()
        
    
    
class Sigmoid(Module):
    """
    A Sigmoid layer to apply the sigmoid function element-wise 
    """
    def __init__(self):
        super().__init__()
        self.s = 0
        
    def sigmoid(self, x):
        return 1 / (1 + (-x).exp())
    
    def forward(self, input):
        self.s = input
        return self.sigmoid(input)

    def backward(self, grdwrtoutput):
        dsigmoid = self.sigmoid(self.s) * (1 - self.sigmoid(self.s))
        return grdwrtoutput * dsigmoid
    
    
    
class ReLU(Module):
    """
    A ReLU layer to apply the rectified linear unit function element-wise
    """
    def __init__(self):
        super().__init__()
        self.s = 0

    def forward(self, input):
        self.s = input
        return input.clamp(min=0.0)

    def backward(self, grdwrtoutput):
        drelu = self.s.sign().clamp(min=0.0)
        return grdwrtoutput * drelu
    
    
class MSE(Module):
    """
    Module to measure the mean square error (MSE) between each element in the input and output
    """
    def __init__(self):
        super().__init__()
        self.s = None
        
    def mse(self, pred, target):
        return ((pred - target)**2).mean()

    def forward(self, pred, target):
        """
        Args:
            pred: model output
            target: target output
        Returns: 
            1/N||pred - target||_2^2
        """
        self.s = pred - target
        return self.mse(pred, target)

    def backward(self):
        return (2 * self.s) / self.s.numel()
    
    
class Optimizer(object):
    """
    Base class for optimizers.
    """
    def step(self):
        """
         Perform the single optimization step
        """
        raise NotImplementedError
        
        
class SGD(Optimizer):
    """
    Module to perform Stochastic Gradient Descent
    """
    def __init__(self, params: list, lr=0.001, momentum=0, dampening=0):
        """
        Parameters:
            params (list): List of the parameters of the network
            lr (float): The learning rate of the network (default: 0.001)
            momentum (float): momentum factor to accelerate gradients vectors in the right directions (default: 0)
            dampening (float): dampening factor for momentum (default: 0)
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.velocity = []

        if self.lr <= 0.0:
            raise ValueError("Learning rate {} should be greater than zero".format(self.lr))
            
        for param in self.params:
            self.velocity.append(param[1])

    def step(self):
        """
        update the weight and bias
        """
        for i, param in enumerate(self.params):
            # momentum and dampening to update the velocity
            self.velocity[i] = self.momentum * self.velocity[i] + (1 - self.dampening) * param[1]
            param[0].add_(- self.lr * self.velocity[i])
                
    def zero_grad(self):
        """
        set the gradients of all optimized tensors to zero.
        """
        for param in self.params:
            param[1].zero_()
          
        
class Sequential(Module):
    """
    A sequential container to pass the modules in the order
    """
    def __init__(self, *modules):
        """
        Parameters:
            modules: Modules to implement in order
        """
        super().__init__()
        self.modules = modules

    def add_module(self, module):
        """
        add the module to the list of modules
        """
        self.modules.append(module)
        return module

    def forward(self, input):
        output = input
        for module in self.modules:
            # apply forward of each module to the input
            output = module.forward(output)
        return output

    def backward(self, grdwrtoutput):
        output = grdwrtoutput
        for module in self.modules[::-1]:
            # apply backward of each module in reverse order
            output = module.backward(output)

    def param(self):
        params = []
        for module in self.modules:
            # append all the parameters of all the modules
            params += module.param()
        return params      
    
    
class Model(Module):
    """
    Model implemented on denoising images
    """
    def __init__(self):
        super().__init__()
        """
        instantiate model
        """
        # the model structure
        self.net =  Sequential(
             Conv2d(3,24,3, stride = 2,padding=1), 
             ReLU(),
             Conv2d(24,24,3, stride = 2,padding=1), 
             ReLU(),
             Upsampling(24,24,3,1,1,2),
             ReLU(),
             Upsampling(24,3,3,1,1,2),
             Sigmoid()
        )
             
    def forward(self, x):
        return self.net(x)
    
    def backward(self, gradwrtoutput):
        return self.net.backward(gradwrtoutput)
        
    def param(self):
        return self.net.param()
    
    def load_pretrained_model(self):
        """
        This loads the parameters saved in bestmodel.pth into the model
        """
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, 'rb') as f:
            params = pickle.load(f)

        for idx in range(4):
            i = 2*idx
            self.net.modules[i].weight.copy_(params[idx][0])
            self.net.modules[i].bias.copy_(params[idx][1])

    def save_parameter(self):
        """
        This returns the parameters to a list
        """
        params = []
        for module in self.net.modules:
            if module.param()!=[]: 
                lst = [module.param()[0][0], module.param()[1][0]]
                params.append(lst)
                      
        return params
                      
    
    def train(self, train_input, train_target, num_epochs):
        """
        train the model
        Args:
            train_input: tensor of size (N, C, H, W) containing a noisy version of the images
            train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, 
                          which only differs from the input by their noise
            num_epochs (int): number of epoches to train the model
        """
        # parameters
        # bacth size is 16 for fast testing, but 4 is used when training
        BATCH_SIZE = 16
        LEARNING_RATE = 1.5
        MOMENTUM = 0
        sum_loss = 0
        
        # optimizer
        optimizer = SGD(params = self.param(), lr = LEARNING_RATE, momentum = MOMENTUM)
        
        # normalization
        train_input_normalized = train_input/255
        train_target_normalized = train_target/255
        
        len_train = train_input.size(0) 
        
        for epoch in range(num_epochs):
            # shuffle the train set
            order = list(range(len_train))
            random.shuffle(order)
            train_input_shuffle = train_input_normalized[order]
            train_target_shuffle = train_target_normalized[order]
            
            # batch
            for b in range(0, len_train, BATCH_SIZE): 
                if b+BATCH_SIZE <= len_train:
                    data = train_input_shuffle.narrow(0, b, BATCH_SIZE)
                    target = train_target_shuffle.narrow(0, b, BATCH_SIZE)
                else:
                    data = train_input_shuffle.narrow(0, b, len_train-b)
                    target = train_target_shuffle.narrow(0, b, len_train-b)
                    
                # forward
                output = self.forward(data) 
                criterion = MSE()
                loss = criterion.forward(output, target)
                sum_loss += loss.item()
                
                # backward
                optimizer.zero_grad()
                dl_dloss = criterion.backward()
                self.backward(dl_dloss)
                
                # optimize
                optimizer.step()
           
                
    def predict(self, test_input):
        """
        predict on the test data
        Args:
            test_input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        Retur
            output: the predicted clean images for input.
        """
        # normalization
        test_input_normalized = test_input/255
        
        # prediction
        output = self(test_input_normalized)
        output = output*255
        
        return output
            