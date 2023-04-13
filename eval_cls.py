import numpy as np
import argparse

import torch
import torch.nn as nn
from models import cls_model
from utils import create_dir, viz_pc

from point_transformer import PointTransformerCls

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model()
    # model = PointTransformerCls()
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    # model_path = './checkpoints/pt_xformer/cls/{}.pt'.format(args.load_checkpoint)

    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.model.load_state_dict(state_dict)
        # model.load_state_dict(state_dict)

    model.eval()
    model.to(args.device)
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device)
    test_label = torch.from_numpy(np.load(args.test_label)).to(args.device)

    
    # Rotate pointcloud
    # R = np.eye(3)
    # R[1,1] = R[2,2] = 0.866
    # R[1,2] = -0.5
    # R[2,1] = 0.5
    # R = torch.from_numpy(R).float()
    # test_data = torch.matmul(test_data,R)

    th1 = 90
    th = np.radians(th1)
    R = np.eye(3)
    R[1,1] = R[2,2] = np.cos(th)
    R[1,2] = -np.sin(th)
    R[2,1] = np.sin(th)
    R = torch.from_numpy(R).float().to(args.device)
    test_data = torch.matmul(test_data,R)

    # Add Gaussian noise to points
    # var = 0.05
    # test_data_copy = test_data
    # test_data = test_data + (var**0.5)*torch.randn(test_data.shape)

    
    # ------ TO DO: Make Prediction ------
    softmax = nn.Softmax(dim=1)
    batches = torch.split(test_data,16,dim=0)
    pred_label = None
    for b in batches:
        predictions = model(b).squeeze(dim=1)
        predictions = softmax(predictions)
        labels = predictions.argmax(dim=1) 
        if pred_label == None:
            pred_label = labels
        else:
            pred_label = torch.cat((pred_label, labels))

    
    #Visualization
    # ind = torch.where(pred_label!=test_label)[0]
    # for i in ind:
    #     gt = test_label[i].item()
    #     pred = pred_label[i].item()
    #     idx = i.item()

    #     print("******")
    #     print("Idx: " + str(idx))
    #     print("GT label "+str(gt))
    #     print("Pred label "+str(pred))

    #     fname = "./output/pc_" + str(idx) + ".gif"
    #     pc = test_data[idx].unsqueeze(0).to(args.device)
    #     viz_pc(pc,fname)

    # ind1 = torch.where(test_label==0)[0]
    # ind2 = torch.where(test_label==1)[0]
    # ind3 = torch.where(test_label==2)[0]
    # ind = [ind1[0].item(), ind2[0].item(), ind3[0].item()]
    # ind = [517, 620, 730]
    ind = [510, 640, 740]
    # ind = [500, 650, 750]

    # ind = [595, 619, 623, 643, 758, 835, 922, 931, 933, 944, 946]
    for i in ind:
        gt = test_label[i].item()
        pred = pred_label[i].item()

        print("******")
        print("Idx: " + str(i))
        print("GT label "+str(gt))
        print("Pred label "+str(pred))

        fname = "./output/q3/rot/rot_" + str(th1) + "_" + str(i) + ".gif"
        pc = test_data[i].unsqueeze(0).to(args.device)
        viz_pc(pc,fname)

        # fname = "./output/q4/q4_gt_" + str(i) + ".gif"
        # pc = test_data[i].unsqueeze(0).to(args.device)
        # viz_pc(pc,fname)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))
