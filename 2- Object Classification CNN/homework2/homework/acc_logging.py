from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """
    logger = tb.SummaryWriter('cnn')

    # This is a strongly simplified training loop
    for epoch in range(10):
        torch.manual_seed(epoch)
        train_acc = torch.zeros(10)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)

            #raise NotImplementedError('Log the training loss')

            train_logger.add_scalar('loss', dummy_train_loss, epoch*20+iteration)

            #a = torch.mean(torch.cat([dummy_train_accuracy]))
            train_acc += dummy_train_accuracy
            #print(dummy_train_accuracy)


        train_logger.add_scalar('accuracy', torch.mean(train_acc/20), epoch*20+iteration)
        #raise NotImplementedError('Log the training accuracy')
        #print(dummy_train_accuracy)

        v_acc = torch.zeros(10)
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            v_acc += dummy_validation_accuracy/10
            #valid_logger.add_scalar('accuracy', torch.mean(v_acc), epoch * 10 +iteration)
        a = epoch*20 + iteration*2 + 1
        print(a)
        print(torch.mean(v_acc))
        print("===============")

        valid_logger.add_scalar('accuracy', torch.mean(v_acc), a)
        #raise NotImplementedError('Log the validation accuracy')


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)