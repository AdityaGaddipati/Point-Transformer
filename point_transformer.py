import torch
import torch.nn as nn
from pytorch3d.ops import knn_points, knn_gather, sample_farthest_points

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, nsample=16):
        super().__init__()
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm2d(3), 
            nn.ReLU(inplace=True), 
            nn.Linear(3, out_planes)
        )

        self.linear_w = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes, out_planes)
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, p):
        '''
        x: B,N,C
        p: B,N,3
        '''

        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)

        dist, ind, _ = knn_points(p1=p, p2=p, K=self.nsample+1, return_sorted=True)  #  (B,N,nsample+1), (B,N,nsample+1s)
        ind = ind[:,:,1:]

        x_k = knn_gather(x_k, ind) # (B,N,nsample,C)
        x_v = knn_gather(x_v, ind) # (B,N,nsample,C)
        
        # position encoding 
        pj = knn_gather(p, ind)  # (B,N,nsample,3)
        pr = pj - p.unsqueeze(2) # (B,N,nsample,3)
        for i,layer in enumerate(self.linear_p): 
            pr = layer(pr.transpose(1,3)).transpose(1,3) if i==1 else layer(pr)

        w = x_q.unsqueeze(2) - x_k + pr  # (B,N,nsample,C)
        for i,layer in enumerate(self.linear_w):
            w = layer(w.transpose(1,3)).transpose(1,3) if i%3==0 else layer(w)

        w = self.softmax(w) # (B,N,nsample,C)
        x = ((x_v + pr)*w).sum(2) # (B,N,C)

        return x, ind


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()

        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool2d((self.nsample,1))
            self.bn = nn.BatchNorm2d(out_planes)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
            self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, xpi):
        '''
        x: B,N,C
        p: B,N,3
        '''
        x, p, knn_ind = xpi
        if  self.stride != 1:
            M = p.shape[1]// self.stride
            new_p, new_p_ind = sample_farthest_points(p, K=M)  # (B,M,3) , (B,M) 
            new_p_nn_ind = knn_gather(knn_ind, new_p_ind.unsqueeze(1)).squeeze(1) # (B,M,nsample)
            new_p_feat = knn_gather(x, new_p_nn_ind)  # (B,M,nsample,C)

            gathered_xyz = knn_gather(p, new_p_nn_ind) # (B,M,nsample,3)
            gathered_xyz -= new_p.unsqueeze(2) # (B,M,nsample,3)

            new_p_feat = torch.cat((gathered_xyz,new_p_feat), -1)  #(B,M,nsample,3+C)

            x = self.relu(self.bn(self.linear(new_p_feat).transpose(1, 3)))  # (B, C, nsample, M)
            x = self.pool(x.transpose(1,3)).squeeze(2)  # (B, M, C)
            p = new_p

        else:
            x = self.relu(self.bn(self.linear(x).transpose(1,2)).transpose(1,2))  # (B, N, C)

        return [x,p,knn_ind]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()

        if out_planes==None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))


    def forward(self, xp1, xp2):
        '''
        x1,p1: (B, 4N, C/2), (B, 4N, 3)
        x2,p2: (B, N, C), (B, N, 3)
        
        xout: (B, 4N, C/2)
        new_p: p1
        '''
        if xp2==None:
            x, p = xp1
            x = torch.cat((x, self.linear2(x.sum(1, True)/x.shape[1]).repeat(1,x.shape[1],1)), dim=2)
            for i,layer in enumerate(self.linear1): 
                x = layer(x.transpose(1,2)).transpose(1,2) if i==1 else layer(x)
            xout = x
        else:
            x1,p1 = xp1
            x2,p2 = xp2

            for i,layer in enumerate(self.linear1): 
                x1 = layer(x1.transpose(1,2)).transpose(1,2) if i==1 else layer(x1)
            for i,layer in enumerate(self.linear2): 
                x2 = layer(x2.transpose(1,2)).transpose(1,2) if i==1 else layer(x2)
            
            xout = x1 + self.interpolate(p2, p1, x2)
        return xout


    def interpolate(self, p, new_p, feat, k=3):
        '''
        p: (B,N,3)
        new_p: (B,4N,3) 
        feat: (B,N,C/2)
        '''
        B,N,C = feat.shape

        dist, ind, _ = knn_points(new_p, p, K=k)   # (B, 4N, k), (B, 4N, k)
        
        dist_ = 1/(dist+1e-8)
        norm = dist_.sum(2, True)
        w = dist_/norm

        new_feat = torch.cuda.FloatTensor(B, new_p.shape[1], C).zero_()
        for i in range(k):
            m = ind[:,:,i].long()
            n = (torch.ones_like(m)*torch.arange(B).to(device='cuda').reshape(-1,1))
            new_feat += feat[n,m, :]*w[:,:,i].unsqueeze(-1)

        return new_feat


class PointTransformerBlock(nn.Module):
    def __init__(self, in_planes, planes, nsample):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xpi):
        '''
        x: B,N,C
        p: B,N,3
        knn_ind: None
        '''
        x, p, knn_ind = xpi
        identity = x
        x = self.relu(self.bn1(self.linear1(x).transpose(1,2)).transpose(1,2))
        x, knn_ind = self.transformer2(x,p)
        x = self.relu(self.bn2(x.transpose(1,2)).transpose(1,2))
        x = self.bn3(self.linear3(x).transpose(1,2)).transpose(1,2)
        x += identity
        x = self.relu(x)  # (B,N,C)
        
        return [x, p, knn_ind]



class PointTransformerSeg(nn.Module):

    def __init__(self, c=3, num_classes=6):
        super().__init__()

        self.in_planes, self.out_planes = c, [32, 64, 128, 256, 512]
        stride, nsample = [1, 4, 4, 4, 4], [16, 16, 16, 16, 16]

        self.enc1 = self.make_enc(self.out_planes[0], stride[0], nsample[0])
        self.enc2 = self.make_enc(self.out_planes[1], stride[1], nsample[1])
        self.enc3 = self.make_enc(self.out_planes[2], stride[2], nsample[2])
        self.enc4 = self.make_enc(self.out_planes[3], stride[3], nsample[3])
        self.enc5 = self.make_enc(self.out_planes[4], stride[4], nsample[4])

        self.dec5 = self.make_dec(self.out_planes[4], nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self.make_dec(self.out_planes[3], nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self.make_dec(self.out_planes[2], nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self.make_dec(self.out_planes[1], nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self.make_dec(self.out_planes[0], nsample=nsample[0])  # fusion p2 and p1

        self.seg = nn.Sequential(nn.Linear(self.out_planes[0], self.out_planes[0]), 
                                nn.BatchNorm1d(self.out_planes[0]), 
                                nn.ReLU(inplace=True), 
                                nn.Linear(self.out_planes[0], num_classes))


        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        for net in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.dec5, self.dec4, self.dec3, self.dec2, self.dec1, self.seg]:
            net.apply(init_weights)


    def make_enc(self, out_planes, stride=1, nsample=16):
        layers=[]
        layers.append(TransitionDown(self.in_planes, out_planes, stride, nsample))
        self.in_planes = out_planes
        layers.append(PointTransformerBlock(self.in_planes, self.in_planes, nsample))
        return nn.Sequential(*layers)

    def make_dec(self, out_planes, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else out_planes))
        self.in_planes = out_planes
        layers.append(PointTransformerBlock(self.in_planes, self.in_planes, nsample=nsample))
        return nn.Sequential(*layers)


    def forward(self, p):
        x = p
        xpi = self.enc1([x,p,None])
        xpi = self.enc2(xpi)
        xpi = self.enc3(xpi)
        xpi = self.enc4(xpi)
        xpi = self.enc5(xpi)

        x1, p1, i1 = self.enc1([x,p,None])
        x2, p2, i2 = self.enc2([x1, p1, i1])
        x3, p3, i3 = self.enc3([x2, p2, i2])
        x4, p4, i4 = self.enc4([x3, p3, i3])
        x5, p5, i5 = self.enc5([x4, p4, i4])

        x5 = self.dec5[1:]([self.dec5[0]([x5, p5],     None), p5, None])[0]
        x4 = self.dec4[1:]([self.dec4[0]([x4, p4], [x5, p5]), p4, None])[0]
        x3 = self.dec3[1:]([self.dec3[0]([x3, p3], [x4, p4]), p3, None])[0]
        x2 = self.dec2[1:]([self.dec2[0]([x2, p2], [x3, p3]), p2, None])[0]
        x1 = self.dec1[1:]([self.dec1[0]([x1, p1], [x2, p2]), p1, None])[0]
        

        for i,layer in enumerate(self.seg): 
                x1 = layer(x1.transpose(1,2)).transpose(1,2) if i==1 else layer(x1)

        return x1


class PointTransformerCls(nn.Module):

    def __init__(self, c=3, num_classes=3):
        super().__init__()

        self.in_planes, self.out_planes = c, [32, 64, 128, 256, 512]
        stride, nsample = [1, 4, 4, 4, 4], [16, 16, 16, 16, 16]

        self.enc1 = self.make_enc(self.out_planes[0], stride[0], nsample[0])
        self.enc2 = self.make_enc(self.out_planes[1], stride[1], nsample[1])
        self.enc3 = self.make_enc(self.out_planes[2], stride[2], nsample[2])
        self.enc4 = self.make_enc(self.out_planes[3], stride[3], nsample[3])
        self.enc5 = self.make_enc(self.out_planes[4], stride[4], nsample[4])

        self.cls = nn.Sequential(nn.Linear(self.out_planes[4], self.out_planes[4]), 
                                nn.BatchNorm1d(self.out_planes[4]), 
                                nn.ReLU(inplace=True), 
                                nn.Linear(self.out_planes[4], num_classes))


    def make_enc(self, out_planes, stride=1, nsample=16):
        layers=[]
        layers.append(TransitionDown(self.in_planes, out_planes, stride, nsample))
        self.in_planes = out_planes
        layers.append(PointTransformerBlock(self.in_planes, self.in_planes, nsample))
        return nn.Sequential(*layers)

    def forward(self, p):
        x = p
        xpi = self.enc1([x,p,None])
        xpi = self.enc2(xpi)
        xpi = self.enc3(xpi)
        xpi = self.enc4(xpi)
        xpi = self.enc5(xpi)

        pool = torch.nn.AvgPool2d((xpi[0].shape[1],1))
        x = pool(xpi[0]).squeeze(1)

        for i,layer in enumerate(self.cls): x = layer(x)

        return x


if __name__ == "__main__":

    # Unit Tests

    pt_xformer = PointTransformerLayer(3,32)
    block = PointTransformerBlock(3,3, nsample=16)

    x = torch.randint(0,9,(4,10000,3)).float()
    p = torch.randint(0,9,(4,10000,3)).float()

    # xout = pt_xformer(x,p)
    # xout = block(x,p)
    # print(x.shape)
    # print(xout.shape)

    # cls_net = PointTransformerCls()
    # p = torch.randint(0,9,(4,10000,3)).float()
    # cls_net(p)
    
    # tup = TransitionUp(64, None)

    # x1 = torch.randint(0,9,(4,10000,32)).float()
    # p1 = torch.randint(0,9,(4,10000,3)).float()

    # x2 = torch.randint(0,9,(4,2500,64)).float()
    # p2 = torch.randint(0,9,(4,2500,3)).float()

    # xout = tup([x2,p2], None)
    # print(xout.shape)

    seg_net = PointTransformerSeg()
    p = torch.randint(0,9,(4,10000,3)).float()
    x = seg_net(p)

    print(x.shape)