from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb

from .utils import accuracy, load_data
import torchvision.transforms as T
from torch.autograd import Variable

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    n_epochs = 20
    train_data = load_data("data/train")
    validation_data = load_data("data/valid")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #logger = tb.SummaryWriter('cnn')
    criterion = torch.nn.CrossEntropyLoss()
    #data_augmentation = 1

    transforms = torch.nn.Sequential(
        T.ColorJitter(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    scripted_transforms = torch.jit.script(transforms)

    for epoch in range(n_epochs):
        acc = 0
        counter1 = 0
        acc_val = 0
        counter2 = 0
        running_loss = 0
        running_loss_val = 0
        print(epoch)
        for x, y in train_data:
            # print(x.shape)
            # print(y.shape)
            #x, y = x.cuda(), y.cuda()
            #x, y = Variable(x), Variable(y)
            optimizer.zero_grad()
            #x = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(T.ToTensor()(x))
            x1 = transforms(x)
            #x  = transforms(x)
            y_pred = model.forward(x)
            #loss = ClassificationLoss.forward(model, y_pred, y)

            loss=criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss
            model.train()
            train_logger.add_scalar('loss', loss, counter1 + 2*len(train_data)*epoch)
            acc += accuracy(y_pred, y)
            counter1 += 1

            ###########################################
            optimizer.zero_grad()
            # x = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(T.ToTensor()(x))
            #x1 = transforms(x)

            y_pred = model.forward(x1)
            # loss = ClassificationLoss.forward(model, y_pred, y)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss
            model.train()
            train_logger.add_scalar('loss', loss, counter1 + 2*len(train_data) * epoch)
            acc += accuracy(y_pred, y)
            counter1 += 1



        for x, y in validation_data:
            #x, y = x.cuda(), y.cuda()
            #x, y = Variable(x), Variable(y)
            y_pred_val = model.forward(x)
            criterion= torch.nn.CrossEntropyLoss()
            loss_val=criterion(y_pred_val, y)

            valid_logger.add_scalar('loss', loss_val, counter2 + len(validation_data)*epoch)
            acc_val += accuracy(y_pred_val, y)
            running_loss_val += loss_val

            counter2 += 1
            model.eval()

        print('epoch {}, loss {}'.format(epoch, running_loss / counter1))
        print('epoch {}, acc {}'.format(epoch, acc / counter1))
        print('epoch {}, loss {}'.format(epoch, running_loss_val / counter2))
        print('epoch {}, acc {}'.format(epoch, acc_val / counter2))

        train_logger.add_scalar('accuracy', acc / counter1, epoch)
        valid_logger.add_scalar('accuracy', acc_val / counter2, epoch)
        #if (acc_val / counter2 > 0.8):
        #    n_epochs = epoch
        # na = lc.w.detach().numpy()
        # print(na)
        # na = lc.b.detach().numpy()
        # print(na)
        # print(lc.w.shape)
        # print(lc.b.shape)
        print("=============================================================")

    print("DONE!")
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
