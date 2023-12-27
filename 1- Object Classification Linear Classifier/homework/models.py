import torch
import torch.nn.functional as F
import torch.nn as nn


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        #print("INPUT-----------",input.shape)
        #print("target-----------",target.shape)

        #loss =  nn.log_softmax(input,dim=0)
        #soft = nn.Softmax(input)
        #log_softmax = soft
        #print("LS", log_softmax)
        #m = nn.LogSoftmax(input)


        output = F.nll_loss(F.log_softmax(input), target)
        #print("m", output)


        #soft = nn.Softmax(input).sum()
        #log_softmax = nn.LogSoftmax()
        #log_probabilities = self.log_softmax(input)
        #loss = -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()

        #loss= -torch.mean(torch.sum(target.view(batch_size, -1) * torch.log(input.view(batch_size, -1)), dim=1))
        #x = torch.flatten(input,start_dim=1)
        #pred_y = torch.matmul(x, self.w) + self.b
        #p_y = 1/(1+(-pred_y).exp())
        #logit = (input * self.w[None,:]).sum(dim=1) + self.b
        #return 1/(1+(-logit).exp())
        
  

        #criterion=nn.CrossEntropyLoss()
        #loss=criterion(input, target)
        #print("loss",loss)
        return output
        #raise NotImplementedError('ClassificationLoss.forward')


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        #print("INIT-----------------------------------------------")
        #self.linear1=nn.Linear(3* 64* 64, 6)

        #print(self.linear1)
        #self.w = torch.zeros([3*64*64,6], requires_grad=True)
        #self.b = torch.zeros([6], requires_grad=True)
        self.w = torch.nn.Parameter(torch.zeros([3*64*64,6], requires_grad=True))
        self.b = torch.nn.Parameter(torch.zeros([6], requires_grad=True))
        



        #raise NotImplementedError('LinearClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        
        #x0 = torch.tensor(x[0], dtype=torch.float)
        #print(x.shape)
        x = torch.flatten(x,start_dim=1)
        pred_y = torch.matmul(x, self.w) + self.b
        #label = (x * self.w[None,:]).sum(dim=1) + self.bias
        #print(logits.shape)

        #x = torch.flatten(x,start_dim=1)
        #x1 = self.linear1(x)
        #x=torch.sigmoid(x1) 

        return pred_y
        #raise NotImplementedError('LinearClassifier.forward')


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """

        self.hidden_size = 6
        self.w1 = torch.nn.Parameter(torch.rand([3*64*64,self.hidden_size], requires_grad=True))
        self.b1 = torch.nn.Parameter(torch.rand([self.hidden_size], requires_grad=True))

        self.w2 = torch.nn.Parameter(torch.rand([self.hidden_size,6], requires_grad=True))
        self.b2 = torch.nn.Parameter(torch.rand([1,6], requires_grad=True))
        #raise NotImplementedError('MLPClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x = torch.flatten(x,start_dim=1)


        #print(x.shape)
        #int(torch.matmul(x, self.w1).shape)
        y1 = torch.matmul(x, self.w1) + self.b1
        #print(y1.shape)
        m = nn.ReLU()
        y2 = m(y1)
        #print(y2.shape)
        #print("Y1",y1)
        #print("Y2",y2)
        #         #print(y2.shape)
        #
        pred_y = torch.matmul(y2, self.w2) + self.b2

        #print(pred_y.shape)
        #print("=============")
        #label = (x * self.w[None,:]).sum(dim=1) + self.bias
        #print(logits.shape)

        #x = torch.flatten(x,start_dim=1)
        #x1 = self.linear1(x)
        #x=torch.sigmoid(x1) 

        return y1
        #raise NotImplementedError('MLPClassifier.forward')


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
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
