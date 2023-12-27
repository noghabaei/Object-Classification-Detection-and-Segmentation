from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))


    n_epochs = 10
    train_data = load_data("data/train")
    validation_data = load_data("data/valid")
    # lc=LinearClassifier()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #logger = tb.SummaryWriter('cnn')

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

            optimizer.zero_grad()
            y_pred = model.forward(x)
            #loss = ClassificationLoss.forward(model, y_pred, y)
            criterion= torch.nn.CrossEntropyLoss()
            loss=criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss
            model.train()
            train_logger.add_scalar('loss', loss, counter1 + len(train_data)*epoch)
            acc += accuracy(y_pred, y)
            counter1 += 1
        for x, y in validation_data:
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
