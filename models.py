import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        
        self.model = nn.Sequential(
                        nn.Linear(3,64),
                        nn.ReLU(),
                        nn.BatchNorm1d(10000),

                        nn.Linear(64,64),
                        nn.ReLU(),
                        nn.BatchNorm1d(10000),

                        nn.Linear(64,128),
                        nn.ReLU(),
                        nn.BatchNorm1d(10000),

                        nn.Linear(128,1024),
                        nn.ReLU(),

                        nn.MaxPool2d((10000,1),1),

                        nn.Linear(1024,512),
                        nn.ReLU(),

                        nn.Linear(512,256),
                        nn.ReLU(),

                        nn.Linear(256,3)
        )

    def forward(self, point_clouds):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        return self.model(point_clouds).squeeze(dim=1)




# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        
        self.classification_net = cls_model()

        with open('./checkpoints/cls/best_model.pt', 'rb') as f:
            state_dict = torch.load(f, map_location=torch.device("cuda"))
            self.classification_net.model.load_state_dict(state_dict)

        self.fc1 = nn.Sequential(
            nn.Linear(1088,512),
            nn.ReLU(),
        )
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
        )
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
        )
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128,num_seg_classes)

    
    def forward(self, point_clouds):

        x = point_clouds
        for i,layer in enumerate(self.classification_net.model):

            x = layer(x)

            if i == 1:
                pt_ft = x
            if i == 11:
                global_ft = x

        # pt_ft = self.point_feat(point_clouds)
        # global_ft = self.global_feat(pt_ft)

        global_ft = global_ft.repeat(1,pt_ft.shape[1],1)
        concat = torch.cat((pt_ft,global_ft),dim=2)

        # predictions = self.seg_net(concat)

        x = self.fc1(concat)
        x = self.bn1(x.transpose(1,2))

        x = self.fc2(x.transpose(1,2))
        x = self.bn2(x.transpose(1,2))

        x = self.fc3(x.transpose(1,2))
        x = self.bn3(x.transpose(1,2))

        predictions = self.fc4(x.transpose(1,2))

        return predictions
