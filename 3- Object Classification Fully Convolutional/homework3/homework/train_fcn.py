import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb

import torchvision.transforms as T
import torch.nn.functional as F
from .utils import accuracy, load_data, _one_hot
#torch.cuda.empty_cache()

from torch.autograd import Variable
from .dense_transforms import RandomResizedCrop


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss



def train(args):
    from os import path

    model = FCN(3,5).cuda()
    #model = model.cuda()


    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    n_epochs = 10
    train_data = load_dense_data("dense_data/train")
    validation_data = load_dense_data("dense_data/valid")
    n_class = 5


    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # logger = tb.SummaryWriter('cnn')
    #criterion = torch.nn.CrossEntropyLoss()
    # data_augmentation = 1

    transforms = torch.nn.Sequential(
        T.ColorJitter(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    scripted_transforms = torch.jit.script(transforms)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    w = []
    for i in DENSE_CLASS_DISTRIBUTION:
        w.append(1/(i**0.75))
    weights = torch.tensor(w).cuda()
    CM_train = ConfusionMatrix()
    CM = ConfusionMatrix()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        acc = 0
        counter1 = 0
        acc_val = 0
        counter2 = 0
        running_loss = 0
        running_loss_val = 0
        print(epoch)
        #torch.cuda.empty_cache()
        for x, y in train_data:
            x, y = x.cuda(), y.cuda()
            x1 = scripted_transforms(x)
            #model.cuda()
            x, y = Variable(x), Variable(y)

            optimizer.zero_grad()
            # x = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(T.ToTensor()(x))

            #x1 = transforms(x)

            y_pred = model.forward(x)
            #loss = ClassificationLoss.forward(model, y_pred, y)

            #y2 = y_pred.max(1)[1].type_as(y)
            #loss = criterion(y2, y)

            size_average = True
            loss = cross_entropy2d(y_pred, y, weight=weights,  size_average=size_average)

            loss.backward()
            optimizer.step()

            running_loss += loss
            model.train()
            train_logger.add_scalar('loss_fcn', loss, counter1 + 2 * len(train_data) * epoch)


            #train_logger.add_scalar('mem0', torch.cuda.memory_allocated(0)/1024/1024/1024, counter1 + 2 * len(train_data) * epoch)
            #train_logger.add_scalar('mem1', torch.cuda.memory_allocated(1), counter1 + 2 * len(train_data) * epoch)
            acc += accuracy(y_pred, y)
            counter1 += 1
            #y
            #10 96 128
            #z = (y.view(10,96,128, 1) == torch.arange(5, dtype=y.dtype, device=x.device)).int()

            #y2 = torch.sum(y_pred, dim=1)

            y2 = y_pred.max(1)[1].type_as(y)
            CM_train.add(y2, y)
            train_logger.add_scalar('fcn avg accuracy', CM_train.global_accuracy, counter1 +  2 * len(train_data) * epoch)
            #print(CM_train.average_accuracy)
            #print(CM_train.iou)

            ###########################################

            x1 = Variable(x1)

            optimizer.zero_grad()

            y_pred = model.forward(x1)
            #loss = ClassificationLoss.forward(model, y_pred, y)

            #loss = criterion(y_pred, y, weights)
            size_average = True
            loss = cross_entropy2d(y_pred, y,weight=weights, size_average=size_average)

            loss.backward()
            optimizer.step()

            running_loss += loss
            model.train()
            train_logger.add_scalar('loss_fcn', loss, counter1 + 2 * len(train_data) * epoch)
            #train_logger.add_scalar('mem0', torch.cuda.memory_allocated(0)/1024/1024/1024, counter1 + 2 * len(train_data) * epoch)
            #train_logger.add_scalar('mem1', torch.cuda.memory_allocated(1), counter1 + 2 * len(train_data) * epoch)
            acc += accuracy(y_pred, y)
            counter1 += 1



        #torch.cuda.empty_cache()
        for x, y in validation_data:
            x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            y_pred_val = model.forward(x)
            #criterion = torch.nn.CrossEntropyLoss()

            size_average = True
            loss_val = cross_entropy2d(y_pred, y,weight=weights, size_average=size_average)
            #loss_val = criterion(y_pred_val, y, weights)

            valid_logger.add_scalar('loss_fcn', loss_val, counter2 + len(validation_data) * epoch)
            acc_val += accuracy(y_pred_val, y)
            running_loss_val += loss_val

            #log(valid_logger, x, y, y_pred_val, (counter2 + len(validation_data) * epoch))
            counter2 += 1
            model.eval()

            y2 = y_pred_val.max(1)[1].type_as(y)
            #y2 = torch.sum(y_pred, dim = 1)
            #y2 = F.one_hot(y.to(torch.int64), 5)
            #y2 = torch.reshape(y2, (10,5,96,128))
            CM.add(y2, y)

        print('epoch {}, loss {}'.format(epoch, running_loss / counter1))
        print('epoch {}, acc {}'.format(epoch, acc / counter1))
        print('epoch {}, loss {}'.format(epoch, running_loss_val / counter2))
        print('epoch {}, acc {}'.format(epoch, acc_val / counter2))
        print('epoch {}, iou {}'.format(epoch, CM.iou))
        print('epoch {}, global_accuracy {}'.format(epoch, CM.global_accuracy))
        print('epoch {}, average_accuracy {}'.format(epoch, CM.average_accuracy))
        print('epoch {}, class_accuracy {}'.format(epoch, CM.class_accuracy))
        print('epoch {}, matric {}'.format(epoch, CM.matrix))

        train_logger.add_scalar('accuracy_fcn', acc / counter1, epoch)
        valid_logger.add_scalar('accuracy_fcn', acc_val / counter2, epoch)

        if(acc_val / counter2 > 0.8):
            save_model(model)
        # if (acc_val / counter2 > 0.8):
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


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
