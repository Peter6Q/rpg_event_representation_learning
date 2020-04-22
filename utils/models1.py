import math
import numpy as np
from os.path import join, dirname, isfile
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from .miscellaneous import outlier1d
from .median_pool import MedianPool2d

DEBUG = 0

if DEBUG>0:
    import matplotlib.pyplot as plt
    import sys
    import time

class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values

class QuantizationLayer(nn.Module):
    def __init__(self, dim, device):
        nn.Module.__init__(self)
        self.dim = dim
        self.mode = 0 # 0:Training 1:Validation 
        self.device = device
        self.segments = 48
        self.startIdx = 3
        self.endBias = 10
        self.blurFilterKernel = torch.tensor([[[
                            .04, .04, .04, .04, .04,
                            .04, .04, .04, .04, .04,
                            .04, .04, .04, .04, .04,
                            .04, .04, .04, .04, .04,
                            .04, .04, .04, .04, .04
                        ]]])
        # self.blurFilterKernel = torch.tensor([[[
        #                             .03125, .03125, .01562, .03125, .03125,
        #                             .03125, .04406, .03125, .04406, .03125,
        #                             .04250, .06250, .14502, .06250, .04250,
        #                             .03125, .04406, .03125, .04406, .03125,
        #                             .03125, .03125, .01562, .03125, .03125
        #                         ]]])
        self.dilationKernel = torch.tensor([[[
                            .0000, .0000, .0700, .0000, .0000,
                            .0000, .0700, .0800, .0700, .0000,
                            .0700, .0800, .1200, .0800, .0700,
                            .0000, .0700, .0800, .0700, .0000,
                            .0000, .0000, .0700, .0000, .0000
                        ]]])
        assert math.isclose( self.blurFilterKernel.sum(), 1., rel_tol=1e-05), 'blurFilterKernel value error'
        assert math.isclose( self.dilationKernel.sum(), 1., rel_tol=1e-05), 'dilationKernel value error'
        self.blurFilterKernelSize = int( math.sqrt( len(self.blurFilterKernel[0,0])))
        self.dilationKernelSize = int( math.sqrt( len(self.dilationKernel[0,0])))

        self.value_layer = ValueLayer([1, 30, 30, 1],
                                      activation=nn.LeakyReLU(negative_slope=0.1),
                                      num_channels=self.segments-self.endBias)
    
    def setMode(self, mode):
        self.mode = mode

    def forward(self, events):
        # points is a list, since events can have any size
        B = len(events)
        # separate events into S segments with evently-divided number of events
        S = self.segments
        # obtain the height & width
        H, W = self.dim

        # obtain class instance variables
        device = self.device
        blurFiltSize = self.blurFilterKernelSize
        blurFiltPad = blurFiltSize//2
        blurFilt = self.blurFilterKernel.expand(1, B, -1)
        blurFilt = blurFilt.view(B, 1, blurFiltSize, blurFiltSize).to(device)
        sIdx = self.startIdx
        eIdx = S - self.endBias

        num_alongX = int( S * W * B)
        num_alongY = int( S * H * B)
        alongX = torch.zeros(num_alongX, dtype=torch.int32, device=device)
        alongY = torch.zeros(num_alongY, dtype=torch.int32, device=device)
        segmentLen_batch = []
        for bi in range(B):
            events[bi] = torch.from_numpy(events[bi]).to(device).squeeze(0)
            x, y, t, p, b = events[bi].t()
            segmentLen = len(x)//S
            segmentLen_batch.append(segmentLen)
            chunks = torch.arange(S, dtype=torch.int32, device=device)
            chunks = chunks.unsqueeze(-1).expand(-1, segmentLen).reshape(-1)
            resultLen = len(chunks)
            ix = x[:resultLen] + W*chunks + W*S*bi
            iy = y[:resultLen] + H*chunks + H*S*bi
            ones = torch.ones(resultLen, dtype=torch.int32, device=device)
            alongX.put_(ix.long(), ones, accumulate=True)
            alongY.put_(iy.long(), ones, accumulate=True)

        segmentLen_batch = torch.FloatTensor(segmentLen_batch).to(device)

        alongX = alongX.view(-1, S*W).float()
        mean = alongX.mean(dim=-1)
        std = alongX.std(dim=-1)
        clampVal = mean + 3*std
        for bi in range(B):
            alongX[bi] = torch.clamp(alongX[bi], 0, clampVal[bi])
        alongX = alongX.view(1, -1, S, W)
        alongX = F.conv2d(alongX, blurFilt, padding=blurFiltPad, groups=B)
        alongX = alongX.squeeze(0)
        alongDim = torch.arange(W, dtype=torch.float32, device=device)
        alongDim = alongDim.expand(B, -1).unsqueeze(-1)
        meanX = torch.bmm(alongX, alongDim).squeeze(-1) / segmentLen_batch.view(-1,1)
        start_seg_x = meanX[:, sIdx].unsqueeze(-1)
        start_seg_x_dis_to_center = W//2 - start_seg_x
        # align and centralize the image along x-axis
        alignedX = meanX - start_seg_x - start_seg_x_dis_to_center
        alignedX = alignedX.round().int()

        alongY = alongY.view(-1, S*H).float()
        mean = alongY.mean(dim=-1)
        std = alongY.std(dim=-1)
        clampVal = mean + 3*std
        for bi in range(B):
            alongY[bi] = torch.clamp(alongY[bi], 0, clampVal[bi])
        alongY = alongY.view(1, -1, S, H)
        alongY = F.conv2d(alongY, blurFilt, padding=blurFiltPad, groups=B)
        alongY = alongY.squeeze(0)
        alongDim = torch.arange(H, dtype=torch.float32, device=device)
        alongDim = alongDim.expand(B, -1).unsqueeze(-1)
        meanY = torch.bmm(alongY, alongDim).squeeze(-1) / segmentLen_batch.view(-1,1)
        start_seg_y = meanY[:, sIdx].unsqueeze(-1)
        start_seg_y_dis_to_center = H//2 - start_seg_y
        # align and centralize the image along y-axis
        alignedY = meanY - start_seg_y - start_seg_y_dis_to_center
        alignedY = alignedY.round().int()

        meanX = meanX.cpu().numpy()
        meanY = meanY.cpu().numpy()
        container_batch = []
        for bi in range(B):
            segmentLen = int(segmentLen_batch[bi].item())
            usableEventsLen = segmentLen*S
            
            x, y, t, p, b = events[bi].t()
            x = x.int()
            y = y.int()
            shiftedX = alignedX[bi].unsqueeze(-1).expand(-1, segmentLen).reshape(-1)
            x = x[:usableEventsLen]
            x -= shiftedX
            x = torch.clamp(x, 0, W-1)
            shiftedY = alignedY[bi].unsqueeze(-1).expand(-1, segmentLen).reshape(-1)
            y = y[:usableEventsLen]
            y -= shiftedY
            y = torch.clamp(y, 0, H-1)

            idx_in_container = x + W*y
            idx_in_container = idx_in_container.long()
            idx_in_container = torch.chunk(idx_in_container, S)

            # C = 3
            # num_voxels = int(2 * C * W * H)
            # vox = torch.zeros(num_voxels, dtype=torch.float32, device=device)
            # p = (p+1)/2  # maps polarity to 0, 1
            # idx_before_bins = x \
            #               + W * y \
            #               + 0 \
            #               + W * H * C * p \
            # idx_before_bins = idx_before_bins.long()
            # idx_before_bins = torch.chunk(idx_before_bins, S)
            t = t[:usableEventsLen]
            t /= t.max()
            t = torch.chunk(t, S)
            
            ones = torch.ones(segmentLen, dtype=torch.float32, device=device)
            container = torch.zeros(W*H, dtype=torch.float32, device=device)
            for si in range(sIdx, eIdx):
                isXoutlier = outlier1d(meanX[bi, si:si+10], thresh=2)[0]
                isYoutlier = outlier1d(meanY[bi, si:si+10], thresh=2)[0]
                if isXoutlier or isYoutlier:
                    continue
                values = t[si] * self.value_layer.forward(t[si]-si/(eIdx-1))
                container.put_(idx_in_container[si], values, accumulate=True)
            container_batch.append(container)
        
        containers = torch.stack(container_batch).view(-1, 1, H, W)

        if DEBUG==9:
            container_img = containers.cpu().numpy()
            for bi in range(B):
                fig = plt.figure(figsize=(15,15))
                ax = []
                rows = 3
                columns = 1
                for i in range(rows * columns):
                    ax.append( fig.add_subplot(rows, columns, i+1))
                    if i==0:
                        ax[-1].set_title('x-t graph')
                        plt.imshow(alongX[bi], cmap='gray')
                    elif i==1:
                        ax[-1].set_title('y-t graph')
                        plt.imshow(alongY[bi], cmap='gray')
                    elif i==2:
                        ax[-1].set_title('x-y graph')
                        plt.imshow(container_img[bi][0], cmap='gray', vmin=0, vmax=1)
                plt.tight_layout()
                plt.show()
            sys.exit(0)

        return containers

class Classifier(nn.Module):
    def __init__(self,
                 dimension=(180,240),
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 pretrained=True,
                 device='cpu'):

        nn.Module.__init__(self)
        self.mode = 0   # 0:Training 1:Validation 
        self.quantization_layer = QuantizationLayer(dimension, device)
        self.crop_dimension = crop_dimension
        self.num_classes = num_classes
        self.in_channels = 1
        self.USE_MEDIAN_FILTER = False

        classifierSelect = 'resnet34'
        if classifierSelect == 'resnet34':
            from torchvision.models.resnet import resnet34
            self.classifier = resnet34(pretrained=pretrained)
            # replace fc layer and first convolutional layer
            self.classifier.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.classifier.fc.in_features, num_classes)
            )
        elif classifierSelect == 'wide_resnet50':
            self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'wide_resnet50_2', pretrained=pretrained)
            # replace fc layer and first convolutional layer
            self.classifier.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.classifier.fc.in_features, num_classes)
            )
        elif classifierSelect == 'densenet201':
            self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'densenet201', pretrained=pretrained)
            # replace fc layer and first convolutional layer
            self.classifier.features.conv0 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.classifier.classifier.in_features, num_classes)
            )
        elif classifierSelect == 'hardnet68':
            # ONly usable with CUDA
            self.classifier = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68', pretrained=pretrained)
            # replace fc layer and first convolutional layer
            self.classifier.base[0].conv = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.classifier.base[-1]._modules['3'] = nn.Linear(1024, num_classes)
        elif classifierSelect == 'mobilenet_v2':
            from torchvision.models.mobilenet import mobilenet_v2, _make_divisible
            self.classifier = mobilenet_v2(pretrained=pretrained)
            classifier_input_channel = _make_divisible(32.0, 8)
            self.classifier.features._modules['0'] = nn.Conv2d(self.in_channels, classifier_input_channel, kernel_size=3, stride=1, padding=1, bias=False)
            self.classifier.classifier._modules['1'] = nn.Linear(self.classifier.last_channel, num_classes)

        self.medianFilter = MedianPool2d(kernel_size=3, stride=1, same=True)
            
    def setMode(self, mode):
        self.mode = mode
        self.quantization_layer.setMode(mode)

    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]
        x = F.interpolate(x, size=output_resolution)
        return x

    def forward(self, x):
        frame = self.quantization_layer.forward(x)
        frame_cropped = self.crop_and_resize_to_resolution(frame, self.crop_dimension)
        if self.USE_MEDIAN_FILTER:
            frame_cropped = self.medianFilter(frame_cropped)
        pred = self.classifier.forward(frame_cropped)
        return pred, frame_cropped
