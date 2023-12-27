from .models import ClassificationLoss, LinearClassifier, model_factory, save_model
from .utils import accuracy, load_data

import torch

def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """

    
    n_epochs = 20
    train_data = load_data("data/train")
    validation_data = load_data("data/valid")
    #lc=LinearClassifier()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(n_epochs):
        acc = 0
        counter1 =0
        acc_val = 0
        counter2 =0
        running_loss = 0
        running_loss_val = 0
        for x, y in train_data:
            #print(x.shape)
            #print(y.shape)
            
            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss = ClassificationLoss.forward(model,y_pred,y)
            #criterion= torch.nn.CrossEntropyLoss()
            #loss=criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss
            model.train() 
            
            acc += accuracy(y_pred,y)
            counter1 += 1
        for x, y in validation_data:
            y_pred_val = model.forward(x)
            loss_val = ClassificationLoss.forward(model,y_pred_val,y)
            acc_val += accuracy(y_pred_val,y)
            running_loss_val += loss_val

            counter2 += 1
            model.eval() 

        print('epoch {}, loss {}'.format(epoch, running_loss/counter1))
        print('epoch {}, acc {}'.format(epoch, acc/counter1))
        print('epoch {}, loss {}'.format(epoch, running_loss_val/counter2))
        print('epoch {}, acc {}'.format(epoch, acc_val/counter2))
        if (acc_val/counter2 > 0.8):
            n_epochs = epoch
        #na = lc.w.detach().numpy()
        #print(na)
        #na = lc.b.detach().numpy()
        #print(na)
        #print(lc.w.shape)
        #print(lc.b.shape)
        print("=============================================================")



    

    #raise NotImplementedError('train')

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
