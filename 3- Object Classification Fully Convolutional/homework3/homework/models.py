import torch
import torch.nn.functional as F

import torch.nn as nn
import numpy as np

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        self.conv1 = torch.nn.Conv2d(3, 20, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(20, 16, 3)
        self.fc1 = torch.nn.Linear(3136, 200)
        self.d1 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(200, 150)
        self.d2 = torch.nn.Dropout(p=0.5)
        self.fc3 = torch.nn.Linear(150, 30)
        self.d3 = torch.nn.Dropout(p=0.5)
        #raise NotImplementedError('CNNClassifier.__init__')


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.d1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.d2(x)
        x = self.fc3(x)
        x = self.d3(x)
        return x
        #raise NotImplementedError('CNNClassifier.forward')



class FCN(torch.nn.Module):
    def __init__(self, in_channels =3 , out_channels = 5):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)


    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Dropout2d(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Dropout2d(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        return expand

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        #print("in",x.size())
        conv1 = self.conv1(x)
        #print("l1",conv1.size())
        conv2 = self.conv2(conv1)
        #print("l2",conv2.size())
        conv3 = self.conv3(conv2)
        #print("l3",conv3.size())


        upconv3 = self.upconv3(conv3)
        #print("l4", upconv3.size())

        #print("cat 1", "l4", upconv3.size(), "l2", conv2.size())
        upconv2 = self.upconv2(torch.cat([upconv3[:,:,0:conv2.size()[2],0:conv2.size()[3]], conv2], 1))
        #print("l5", upconv2.size())
        #print("cat 2", "l5", upconv2.size(), "l1",conv1.size())
        upconv1 = self.upconv1(torch.cat([upconv2[:,:,0:conv1.size()[2],0:conv1.size()[3]], conv1], 1))
        upconv1 = upconv1[:,:,0:x.size()[2],0:x.size()[3]]
        #print("l6", upconv1.size())

        return upconv1



model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
