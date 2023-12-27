import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data, DetectionSuperTuxDataset
from . import dense_transforms
from .dense_transforms import detections_to_heatmap
import torch.utils.tensorboard as tb
from torch.autograd import Variable

def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    import torch

    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    device = torch.device('cuda')

    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'fcn.th')))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    #DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]
    DENSE_CLASS_DISTRIBUTION = [0.5, 0.45, 0.05]
    #w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION) ** (-args.gamma)
    pos_weight = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)

    #loss = torch.nn.CrossEntropyLoss().to(device)

    #loss = torch.nn.BCEWithLogitsLoss().to(device)
    #3, 96, 128
    pos_weight = torch.ones([16, 96, 128])
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    #loss1 = torch.nn.BCEWithLogitsLoss(reduction="none").to(device)

    #loss2 = torch.nn.BCEWithLogitsLoss(reduction="sum").to(device)
    #w = w[None, :, None, None]

    #pos_weight = torch.as_tensor([0.25,0.5, 0.25])  # All weights are equal to 1
    #loss = torch.nn.BCEWithLogitsLoss(pos_weight=w[None, :, None, None]).to(device)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    #train_data = load_detection_data('dense_data/train')
    valid_data = load_detection_data('dense_data/valid', num_workers=4, transform=transform)
    #train_data = load_detection_data('dense_data/train', num_workers=0)
    #train_data = DetectionSuperTuxDataset('dense_data/train')
    #valid_data = DetectionSuperTuxDataset('dense_data/valid')

    global_step = 0


    for epoch in range(args.num_epoch):
        model.train()

        #conf = ConfusionMatrix()
        running_loss = 0
        for img, det, size in train_data:
            #img, label = img.to(device), label.to(device).long()
            #img, kart, bomb, pickup = img.to(device), kart bomb.to(device).long(), pickup.to(device).long()
            img = img.to(device)

            #img = torch.sigmoid(img)
            logit = model.forward(img)
            #logit = torch.sigmoid(logit)

            #dets = [kart, bomb, pickup]


            ###detections_to_heatmap
            #peak, size = detections_to_heatmap(dets, img.shape[1:], radius=4)
            #peak = peak.to(device)

            #label = model.detect(logit)
            #peak = peak[None, :]
            det = det.to(device)
            loss_val = loss(logit, det)
            #l1 = loss1(logit, det)
            #l2 = loss2(logit, det)
            #loss_val = loss(logit, label)
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, det, logit, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            #conf.add(logit.argmax(1), label)
            running_loss += loss_val
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        # if train_logger:
        #     train_logger.add_scalar('global_accuracy', conf.global_accuracy, global_step)
        #     train_logger.add_scalar('average_accuracy', conf.average_accuracy, global_step)
        #     train_logger.add_scalar('iou', conf.iou, global_step)
        print('epoch {}, loss {}'.format(epoch, running_loss / (10000/16)))

        model.eval()
        #val_conf = ConfusionMatrix()
        for img, det, size in valid_data:
            img = img.to(device)
            det = det.to(device)
            #img, label = img.to(device), label.to(device).long()
            logit = model(img)
            #logit = torch.sigmoid(logit)

            #val_conf.add(logit.argmax(1), label)

        if valid_logger is not None:
            log(valid_logger, img, det, logit, global_step)

        #if valid_logger:
            #valid_logger.add_scalar('global_accuracy', val_conf.global_accuracy, global_step)
           # valid_logger.add_scalar('global_accuracy', val_conf.global_accuracy, global_step)
            #valid_logger.add_scalar('average_accuracy', val_conf.average_accuracy, global_step)
            #valid_logger.add_scalar('iou', val_conf.iou, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f \t iou = %0.3f \t val iou = %0.3f' %
                  (epoch, conf.global_accuracy, val_conf.global_accuracy, conf.iou, val_conf.iou))
        save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """

    #imgs = imgs[None, :]
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap()])')

    args = parser.parse_args()
    train(args)

