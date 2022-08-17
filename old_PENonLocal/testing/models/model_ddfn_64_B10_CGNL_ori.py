import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

# feature extraction part
class MsFeat(nn.Module):
    def __init__(self, in_channels):
        outchannel_MS = 2
        super(MsFeat, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1,1,1), padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

        self.conv2 = nn.Sequential(nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1,1,1), padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv2[0].bias, 0.0)

        self.conv3 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv3[0].bias, 0.0)

        self.conv4 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv4[0].bias, 0.0)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(inputs)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv1)
        return torch.cat((conv1, conv2, conv3, conv4), 1)


class NonLocal(nn.Module):
    def __init__(self, inplanes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups

        super(NonLocal, self).__init__()
        # conv theta
        self.t = nn.Conv3d(inplanes, inplanes//1, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.t.weight, 0, 'fan_in', 'relu')
        # conv phi
        self.p = nn.Conv3d(inplanes, inplanes//1, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.p.weight, 0, 'fan_in', 'relu')
        # conv g
        self.g = nn.Conv3d(inplanes, inplanes//1, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.g.weight, 0, 'fan_in', 'relu')
        # conv z
        self.z = nn.Conv3d(inplanes//1, inplanes, kernel_size=1, stride=1,
                           groups=self.groups, bias=False)
        init.kaiming_normal_(self.z.weight, 0, 'fan_in', 'relu')
        # concat groups
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        init.constant_(self.gn.weight, 0); nn.init.constant_(self.gn.bias, 0)

        if self.use_scale:
            print("=> WARN: Non-local block uses 'SCALE'")
        if self.groups:
            print("=> WARN: Non-local block uses '{}' groups".format(self.groups))

    def kernel(self, t, p, g, b, c, d, h, w):
        """The linear kernel (dot production).

        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            d: depth of featuremaps
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * d * h * w)
        p = p.view(b, 1, c * d * h * w)
        g = g.view(b, c * d * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c * d * h * w) ** 0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, d, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x) #b,ch,d,h,w
        p = self.p(x) #b,ch,d,h,w
        g = self.g(x) #b,ch,d,h,w

        b, c, d, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, d, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, d, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


# feature integration
class Block(nn.Module):
    def __init__(self, in_channels):
        outchannel_block = 16
        super(Block, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, outchannel_block, 1, padding=0, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

        self.feat1 = nn.Sequential(nn.Conv3d(outchannel_block, 8, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat1[0].bias, 0.0)

        self.feat15 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat15[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat15[0].bias, 0.0)

        self.feat2 = nn.Sequential(nn.Conv3d(outchannel_block, 8, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat2[0].bias, 0.0)

        self.feat25 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat25[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat25[0].bias, 0.0)

        self.feat = nn.Sequential(nn.Conv3d(24, 8, 1, padding=0, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat[0].bias, 0.0)
    # note the channel for each layer
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        feat1 = self.feat1(conv1)
        feat15 = self.feat15(feat1)
        feat2 = self.feat2(conv1)
        feat25 = self.feat25(feat2)
        feat = self.feat(torch.cat((feat1, feat15, feat2, feat25), 1))
        return torch.cat((inputs, feat), 1)


# build the model
class DeepBoosting(nn.Module):
    def __init__(self, in_channels=1):
        super(DeepBoosting, self).__init__()
        self.msfeat = MsFeat(in_channels)
        self.C1 = nn.Sequential(nn.Conv3d(8,2,kernel_size=1, stride=(1,1,1),bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.C1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.C1[0].bias, 0.0)
        self.nl = NonLocal(2, use_scale=False, groups=1)

        self.ds1 = nn.Sequential(nn.Conv3d(2,4,kernel_size=3,stride=(2,1,1),padding=(1,1,1),bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.ds1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds1[0].bias, 0.0)
        self.ds2 = nn.Sequential(nn.Conv3d(4, 8, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.ds2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds2[0].bias, 0.0)
        self.ds3 = nn.Sequential(nn.Conv3d(8,16,kernel_size=3,stride=(2,1,1),padding=(1,1,1),bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.ds3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds3[0].bias, 0.0)
        self.ds4 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.ds4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds4[0].bias, 0.0)

        self.dfus_block0 = Block(32)
        self.dfus_block1 = Block(40)
        self.dfus_block2 = Block(48)
        self.dfus_block3 = Block(56)
        self.dfus_block4 = Block(64)
        self.dfus_block5 = Block(72)
        self.dfus_block6 = Block(80)
        self.dfus_block7 = Block(88)
        self.dfus_block8 = Block(96)
        self.dfus_block9 = Block(104)
        self.convr = nn.Sequential(
            nn.ConvTranspose3d(112, 56, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),nn.ReLU(inplace=True),
            nn.ConvTranspose3d(56, 28, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),nn.ReLU(inplace=True),
            nn.ConvTranspose3d(28, 14, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),nn.ReLU(inplace=True),
            nn.ConvTranspose3d(14, 7, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False))
        init.kaiming_normal_(self.convr[0].weight, 0, 'fan_in', 'relu')
        init.kaiming_normal_(self.convr[2].weight, 0, 'fan_in', 'relu')
        init.kaiming_normal_(self.convr[4].weight, 0, 'fan_in', 'relu')
        init.normal_(self.convr[6].weight, mean=0.0, std=0.001)

        self.C2 = nn.Sequential(nn.Conv3d(7,1,kernel_size=1, stride=(1,1,1),bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.C2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.C2[0].bias, 0.0)
        
    def forward(self, inputs):
        smax = torch.nn.Softmax2d()

        msfeat = self.msfeat(inputs) 
        c1 = self.C1(msfeat)
        nlout = self.nl(c1)
        dsfeat1 = self.ds1(nlout)
        dsfeat2 = self.ds2(dsfeat1) 
        dsfeat3 = self.ds3(dsfeat2) 
        dsfeat4 = self.ds4(dsfeat3) 
        b0 = self.dfus_block0(dsfeat4)
        b1 = self.dfus_block1(b0)
        b2 = self.dfus_block2(b1)
        b3 = self.dfus_block3(b2)
        b4 = self.dfus_block4(b3)
        b5 = self.dfus_block5(b4)
        b6 = self.dfus_block6(b5)
        b7 = self.dfus_block7(b6)
        b8 = self.dfus_block8(b7)
        b9 = self.dfus_block9(b8)
        convr = self.convr(b9)
        convr = self.C2(convr)

        denoise_out = torch.squeeze(convr, 1)

        weights = Variable(torch.linspace(0, 1, steps=denoise_out.size()[1]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax(denoise_out)
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        return denoise_out, soft_argmax
