import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import cls_model, seg_model
from data_loader import get_data_loader
from utils import save_checkpoint, create_dir

from point_transformer import PointTransformerCls, PointTransformerSeg

def train(train_dataloader, model, opt, epoch, args, writer):
    
    model.train()
    step = epoch*len(train_dataloader)
    epoch_loss = 0

    for i, batch in enumerate(train_dataloader):
        point_clouds, labels = batch
        point_clouds = point_clouds.to(args.device)
        labels = labels.to(args.device).to(torch.long)

        # ------ TO DO: Forward Pass ------
        if args.task == "cls":
            predictions = model(point_clouds).squeeze(dim=1)
            # predictions = model(point_clouds)

        else:
            # pt_ft = point_feat(point_clouds)
            # global_ft = global_feat(pt_ft)

            # global_ft = global_ft.repeat(1,pt_ft.shape[1],1)
            # concat = torch.cat((pt_ft,global_ft),dim=2)

            # predictions = seg_net(concat)
            predictions = model(point_clouds)


        if (args.task == "seg"):
            labels = labels.reshape([-1])
            predictions = predictions.reshape([-1, args.num_seg_class])
            

        # Compute Loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)
        epoch_loss += loss

        # Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar('train_loss', loss.item(), step+i)

    return epoch_loss

def test(test_dataloader, model, epoch, args, writer):
    
    model.eval()
    # Evaluation in Classification Task
    if (args.task == "cls"):
        correct_obj = 0
        num_obj = 0
        softmax = nn.Softmax(dim=1)
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                predictions = model(point_clouds).squeeze(dim=1)
                # predictions = model(point_clouds)
                predictions = softmax(predictions)
                pred_labels = predictions.argmax(dim=1)
            correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_obj / num_obj
                
        
    # Evaluation in Segmentation Task
    else:
        correct_point = 0
        num_point = 0
        softmax = nn.Softmax(dim=2)
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                predictions = model(point_clouds)
                # predictions = predictions.reshape([-1, args.num_seg_class])
                predictions = softmax(predictions)
                pred_labels = predictions.argmax(dim=2)

                # print(predictions.shape)
                # print(pred_labels.shape)
                # print(labels.shape)

            correct_point += pred_labels.eq(labels.data).cpu().sum().item()
            num_point += labels.view([-1,1]).size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_point / num_point

    writer.add_scalar("test_acc", accuracy, epoch)
    return accuracy


def main(args):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create Directories
    create_dir(args.checkpoint_dir)
    create_dir('./logs')

    # Tensorboard Logger
    writer = SummaryWriter('./logs/{0}'.format(args.task+"_"+args.exp_name))

    # ------ TO DO: Initialize Model ------
    if args.task == "cls":
        # model = nn.Sequential(
        #         nn.Linear(3,64),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(10000),

        #         nn.Linear(64,64),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(10000),

        #         nn.Linear(64,128),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(10000),

        #         nn.Linear(128,1024),
        #         nn.ReLU(),

        #         nn.MaxPool2d((10000,1),1),
        #         # nn.BatchNorm1d(10000),

        #         nn.Linear(1024,512),
        #         nn.ReLU(),
        #         # nn.BatchNorm1d(10000),

        #         nn.Linear(512,256),
        #         nn.ReLU(),
        #         # nn.BatchNorm1d(10000),

        #         nn.Linear(256,3)
        # )

        model = PointTransformerCls()
        
    else:
        # model = seg_model()
        model = PointTransformerSeg()
        

    model = model.to(args.device)

    # Load Checkpoint 
    if args.load_checkpoint:
        model_path = "{}/{}.pt".format(args.checkpoint_dir,args.load_checkpoint)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict)
        print ("successfully loaded checkpoint from {}".format(model_path))

    # Optimizer
    opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=0.0001)
    # opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    
    # Dataloader for Training & Testing
    train_dataloader = get_data_loader(args=args, train=True)
    test_dataloader = get_data_loader(args=args, train=False)

    print ("successfully loaded data")

    best_acc = -1

    print ("======== start training for {} task ========".format(args.task))
    print ("(check tensorboard for plots of experiment logs/{})".format(args.task+"_"+args.exp_name))
    
    for epoch in range(args.num_epochs):

        # Train
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)
        
        # Test
        current_acc = test(test_dataloader, model, epoch, args, writer)

        print ("epoch: {}   train loss: {:.4f}   test accuracy: {:.4f}".format(epoch, train_epoch_loss, current_acc))
        
        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            print ("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=False)

        # Save Best Model Checkpoint
        if (current_acc >= best_acc):
            best_acc = current_acc
            print ("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=True)

    print ("======== training completes ========")


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model & Data hyper-parameters
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (default 0.001)')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--main_dir', type=str, default='./data/')
    # parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/pt_xformer')

    parser.add_argument('--checkpoint_every', type=int , default=10)

    parser.add_argument('--load_checkpoint', type=str, default='')
    

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir+"/"+args.task # checkpoint directory is task specific

    main(args)