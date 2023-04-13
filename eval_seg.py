import numpy as np
import argparse

import torch
import torch.nn as nn
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg

from point_transformer import PointTransformerSeg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model()
    # model = PointTransformerSeg()
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    # model_path = './checkpoints/pt_xformer/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device)
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind]).to(args.device)

    # ------ TO DO: Make Prediction ------
    softmax = nn.Softmax(dim=2)
    batches = torch.split(test_data,16,dim=0)
    pred_label = None
    for b in batches:
        predictions = model(b)
        predictions = softmax(predictions)
        labels = predictions.argmax(dim=2) 
        if pred_label == None:
            pred_label = labels
        else:
            pred_label = torch.cat((pred_label, labels), dim=0)

    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    #Visualization
    # for i in range(32):
    #     acc = pred_label[i].eq(test_label[i].data).cpu().sum().item() / (pred_label[i].shape[0])
    #     print("****")
    #     print(i,acc)

    #     fname = "./output/seg/gt_" + str(i) + ".gif"
    #     viz_seg(test_data[i], test_label[i], fname, args.device)
        
    #     fname = "./output/seg/pred_" + str(i) + ".gif"
    #     viz_seg(test_data[i], pred_label[i], fname, args.device)

    # np.random.seed(777)
    # ind = np.random.randint(0,test_data.shape[0], (64,))
    
    # ind = [577,47,80,85,365,423,351,276]
    # for i in ind:
    #     acc = pred_label[i].eq(test_label[i].data).cpu().sum().item() / (pred_label[i].shape[0])
    #     print("****")
    #     print(i,acc)

    #     fname = "./output/q4/seg/gt_" + str(i) + "_pt" + ".gif"
    #     viz_seg(test_data[i].cpu(), test_label[i].cpu(), fname, args.device)
        
    #     fname = "./output/q4/seg/pred_" + str(i) + "_pt" + ".gif"
    #     viz_seg(test_data[i].cpu(), pred_label[i].cpu(), fname, args.device)



    


    ind = []
    for i in range(0,test_data.shape[0]):
        acc = pred_label[i].eq(test_label[i].data).cpu().sum().item() / (pred_label[i].shape[0])
        print("****")
        print(i,acc)

        if acc<0.7:
            ind.append(i)

            fname = "./output/q4/seg/gt_" + str(i) + ".gif"
            viz_seg(test_data[i].cpu(), test_label[i].cpu(), fname, args.device)
            
            fname = "./output/q4/seg/pred_" + str(i) + ".gif"
            viz_seg(test_data[i].cpu(), pred_label[i].cpu(), fname, args.device)



    model = PointTransformerSeg()
    model_path = './checkpoints/pt_xformer/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)
    print ("successfully loaded checkpoint from {}".format(model_path))

    batches = torch.split(test_data,6,dim=0)
    pred_label = None
    for b in batches:
        predictions = model(b)
        predictions = softmax(predictions)
        labels = predictions.argmax(dim=2) 
        if pred_label == None:
            pred_label = labels
        else:
            pred_label = torch.cat((pred_label, labels), dim=0)
    
    for i in ind:
        acc = pred_label[i].eq(test_label[i].data).cpu().sum().item() / (pred_label[i].shape[0])
        print("****")
        print(i,acc)
        
        fname = "./output/q4/seg/pred_" + str(i) + "_pt" + ".gif"
        viz_seg(test_data[i].cpu(), pred_label[i].cpu(), fname, args.device)



    # Visualize Segmentation Result (Pred VS Ground Truth)
    # viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    # viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
