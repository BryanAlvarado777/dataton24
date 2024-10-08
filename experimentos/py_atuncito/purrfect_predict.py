# %% [markdown]
# ## Carga modulo comun

# %%
import sys
import os
sys.path.append(os.path.abspath('../../common'))

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

#from purrfect import load_partition,save_partition, create_train_valid_loaders, RandomTransform
from purrfect.dataset import load_partition,save_partition, create_train_valid_loaders, RandomTransform

from purrfect.training import train_model
import torch.optim as optim

from purrfect.active_learning import create_new_partition, test_model

from sklearn.model_selection import train_test_split
#from purrfect.submission import create_submission

# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
USE_AUTOCAST = False
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_GRACE_PERIOD = 8
PARTITION_SIZE = 5

# %% [markdown]
# ## Definición modelo

# %%
class GradientMagnitude(nn.Module):
    def __init__(self):
        super(GradientMagnitude, self).__init__()
        # Define Sobel filters for computing gradients in x and y directions
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,device=DEVICE)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,device=DEVICE)
        
        # Sobel filters need to be reshaped for convolution: (out_channels, in_channels, height, width)
        # Apply the filters across all channels by expanding them to shape (C, 1, 3, 3)
        self.sobel_x = sobel_x.view(1, 1, 3, 3)
        self.sobel_y = sobel_y.view(1, 1, 3, 3)
    
    def forward(self, x):
        B, C, W, H = x.shape
        
        # Apply Sobel filters to compute gradients in x and y directions for all channels
        grad_x = F.conv2d(x, self.sobel_x.expand(C, 1, 3, 3), groups=C, padding=1)
        grad_y = F.conv2d(x, self.sobel_y.expand(C, 1, 3, 3), groups=C, padding=1)
        
        # Compute gradient magnitude: sqrt(grad_x^2 + grad_y^2)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        return grad_mag
def aggregate_similarity_cosine(tensor):
    # tensor is of shape (B, 2, H, W)
    #B, C, H, W = tensor.shape
    #assert C == 2, "The channel dimension must represent 2D vectors (x, y components)"

    # Define shifts for neighboring vectors (up, down, left, right) and stack them
    shift_up = F.pad(tensor[:, :, 1:, :], (0, 0, 0, 1))  # Shift up
    shift_down = F.pad(tensor[:, :, :-1, :], (0, 0, 1, 0))  # Shift down
    shift_left = F.pad(tensor[:, :, :, 1:], (0, 1, 0, 0))  # Shift left
    shift_right = F.pad(tensor[:, :, :, :-1], (1, 0, 0, 0))  # Shift right

    # Stack shifted neighbors into a single tensor of shape (B, 2, H, W, 4)
    neighbors = torch.stack((shift_up, shift_down, shift_left, shift_right), dim=4)  # (B, 2, H, W, 4)
    # Calculate dot product for all neighbors
    dot_product = (tensor.unsqueeze(4) * neighbors).sum(dim=1,keepdim=True)  # (B, 1, H, W, 4)
    # Calculate mean, max, and min along the neighbor dimension (dim=4)
    #mean_similarity = dot_product.mean(dim=4, keepdim=True)  # (B, 1, H, W, 1)
    max_similarity = dot_product.max(dim=4, keepdim=True).values  # (B, 1, H, W, 1)
    min_similarity = dot_product.min(dim=4, keepdim=True).values  # (B, 1, H, W, 1)

    # Concatenate mean, max, and min similarities along the channel dimension
    aggregate_similarity = torch.cat((max_similarity, min_similarity), dim=1)  # (B, 3, H, W,1)

    return aggregate_similarity.squeeze(-1)  # (B, 3, H, W)
class ChannelAdder(nn.Module):
    def __init__(self):
        super(ChannelAdder, self).__init__()
        self.grad_magnitude = GradientMagnitude()

    def forward(self, x):
        
        # Extract the first two channels (e1 and e2) and the third channel (delta) directly
        e = x[:, :2, :, :]  # Shape (B, 2, H, W)
        delta = x[:, 2:3, :, :]  # Shape (B, 1, H, W)

        # Calculate the magnitude in one step (B, 1, H, W)
        magnitude = torch.norm(e, dim=1, keepdim=True)  # Efficient norm calculation

        # Calculate the angle using atan2 to avoid division by zero and handle quadrant
        angle = 0.5 * torch.atan2(e[:, 0, :, :], e[:, 1, :, :]).unsqueeze(1)  # (B, 1, H, W)

        # Compute the weighted components e1_weighted and e2_weighted
        e_norm = e / magnitude  # Split along channel dimension

        cross_product = e[:, 0:1, :, :] * e[:, 1:2, :, :]  # Efficient cross-product (B, 1, H, W)

        simmilarity = aggregate_similarity_cosine(e_norm)
        # Concatenate all the channels (original and new) into the output tensor
        output = torch.cat([
            e,
            magnitude,
            angle,
            e_norm,
            cross_product,
            delta,
            simmilarity #min max and mean simmilarity
        ], dim=1)
        output = torch.cat([output, self.grad_magnitude(output)], dim=1)

        return output

# %%
BN_MOMENTUM = 0.1
def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
class PA(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PA, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()
    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool_mask(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)
        return mask_ch

    def channel_pool_mask(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)
        return mask_sp
    def forward(self, x, y):
        sp_mask = self.spatial_pool_mask(y)
        ch_mask = self.channel_pool_mask(y)
        return x * sp_mask * ch_mask
    

class PSA(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA, self).__init__()
        self.pa = PA(inplanes, planes, kernel_size, stride)

    def forward(self, x):
        return self.pa(x,x)

class ConBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=True,use_bn=True,use_relu=True):
        super(ConBnRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias)
        if use_bn and use_relu:
            self.output = nn.Sequential(
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
        elif use_bn:
            self.output = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        elif use_relu:
            self.output = nn.ReLU(inplace=True)
        else:
            self.output = nn.Identity()
        
        
    def forward(self, x):
        return self.output(self.conv(x))
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, use_attention=False):
        super(ResBlock, self).__init__()
        self.rescale = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        if use_attention:
            self.seq1 = nn.Sequential(
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                PSA(out_channels, out_channels),
                #ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                #PSA(out_channels, out_channels),
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,use_relu=False),
            )
            self.seq2 = nn.Sequential(
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                PSA(out_channels, out_channels),
                #ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                #PSA(out_channels, out_channels),
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,use_relu=False)
            )
        else:
            self.seq1 = nn.Sequential(
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,use_relu=False)
            )
            self.seq2 = nn.Sequential(
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, use_relu=False)
            )
            self.seq3 = nn.Sequential(
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                ConBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, use_relu=False)
            )

        

    def forward(self, x):
        x = self.rescale(x)
        x = F.relu(self.seq1(x) + x, inplace=True)
        x = F.relu(self.seq2(x) + x, inplace=True)
        x = F.relu(self.seq3(x) + x, inplace=True)
        return x

class UNetEncoder(nn.Module):
    def __init__(self, i_ch=16):
        super(UNetEncoder, self).__init__()
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool2d(2),  # i_chx128x128 -> i_chx64x64
            ResBlock(i_ch, i_ch * 2)  # i_chx64x64 -> (i_ch * 2)x64x64
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool2d(2),  # (i_ch * 2)x64x64 -> (i_ch * 2)x32x32
            ResBlock(i_ch * 2, i_ch * 4)  # (i_ch * 2)x32x32 -> (i_ch * 4)x32x32
        )
        self.encoder3 = nn.Sequential(
            nn.AvgPool2d(2),  # (i_ch * 4)x32x32 -> (i_ch * 4)x16x16
            ResBlock(i_ch * 4, i_ch * 8)  # (i_ch * 4)x16x16 -> (i_ch * 8)x16x16
        )
        self.encoder4 = nn.Sequential(
            nn.AvgPool2d(2),  # (i_ch * 8)x16x16 -> (i_ch * 8)x8x8
            ResBlock(i_ch * 8, i_ch * 16)  # (i_ch * 8)x8x8 -> (i_ch * 16)x8x8
        )
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        return e1, e2, e3, e4
class UNetDecoder(nn.Module):
    def __init__(self, i_ch=16):
        super(UNetDecoder, self).__init__()
        # Decoder
        self.attention4 = PA(i_ch * 16, i_ch * 16)
        self.upconv4 = nn.ConvTranspose2d(i_ch * 16*2, i_ch * 16, kernel_size=2, stride=2)  # (i_ch * 16)x8x8 -> (i_ch * 8)x16x16
        self.decoder4 = ResBlock(i_ch * 16, i_ch * 8)  # (i_ch * 16)x16x16 -> (i_ch * 8)x16x16

        self.attention3 = PA(i_ch * 8, i_ch * 8)
        self.upconv3 = nn.ConvTranspose2d(i_ch * 8*2, i_ch * 8, kernel_size=2, stride=2)  # (i_ch * 8)x16x16 -> (i_ch * 4)x32x32
        self.decoder3 = ResBlock(i_ch * 8, i_ch * 4)  # (i_ch * 8)x32x32 -> (i_ch * 4)x32x32

        self.attention2 = PA(i_ch * 4, i_ch * 4)
        self.upconv2 = nn.ConvTranspose2d(i_ch * 4*2, i_ch * 4, kernel_size=2, stride=2)  # (i_ch * 4)x32x32 -> (i_ch * 2)x64x64
        self.decoder2 = ResBlock(i_ch * 4, i_ch * 2)  # (i_ch * 4)x64x64 -> (i_ch * 2)x64x64

        self.attention1 = PA(i_ch*2, i_ch*2)
        self.upconv1 = nn.ConvTranspose2d(i_ch * 2*2, i_ch*2, kernel_size=2, stride=2)  # (i_ch * 2)x64x64 -> i_chx128x128
        self.decoder1 = ResBlock(i_ch * 2, i_ch)  # (i_ch * 2)x128x128 -> i_chx128x128
    def forward(self, x, e1, e2, e3, e4):
        d4 = torch.cat((self.attention4(e4, x), x), dim=1)
        d4 = self.upconv4(d4)
        d4 = self.decoder4(d4)

        d3 = torch.cat((self.attention3(e3, d4), d4), dim=1)
        d3 = self.upconv3(d3)
        d3 = self.decoder3(d3)

        
        d2 = torch.cat((self.attention2(e2, d3), d3), dim=1)
        d2 = self.upconv2(d2)
        d2 = self.decoder2(d2)

        
        d1 = torch.cat((self.attention1(e1, d2), d2), dim=1)
        d1 = self.upconv1(d1)
        d1 = self.decoder1(d1)
        return d1
class UNetBottleneck(nn.Module):
    def __init__(self,input_ch=16):
        super(UNetBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            ResBlock(input_ch * 16, input_ch*16),
            ResBlock(input_ch * 16, input_ch*16)
        )
    def forward(self, x):
        return self.bottleneck(x)
class MultiHeadUNet(nn.Module):
    def __init__(self, in_channels, out_channels, i_ch=16):
        super(MultiHeadUNet, self).__init__()
        
        self.embedder1 = nn.Sequential(
            ResBlock(in_channels, i_ch),  # in_channelsx128x128 -> i_chx128x128
            ResBlock(i_ch, i_ch),  # i_chx128x128 -> i_chx128x128
        )
        self.embedder2 = nn.Sequential(
            ResBlock(in_channels, i_ch),  # in_channelsx128x128 -> i_chx128x128
            ResBlock(i_ch, i_ch),  # i_chx128x128 -> i_chx128x128
        )
        self.embedder3 = nn.Sequential(
            ResBlock(in_channels, i_ch),  # in_channelsx128x128 -> i_chx128x128
            ResBlock(i_ch, i_ch),  # i_chx128x128 -> i_chx128x128
        )

        self.embedder4 = nn.Sequential(
            ResBlock(in_channels, i_ch),  # in_channelsx128x128 -> i_chx128x128
            ResBlock(i_ch, i_ch),  # i_chx128x128 -> i_chx128x128
        )

        self.embedder5 = nn.Sequential(
            ResBlock(in_channels, i_ch),  # in_channelsx128x128 -> i_chx128x128
            ResBlock(i_ch, i_ch),  # i_chx128x128 -> i_chx128x128
        )
        
        self.encoder1 = UNetEncoder(i_ch)
        self.encoder2 = UNetEncoder(i_ch)
        self.encoder3 = UNetEncoder(i_ch)
        self.encoder4 = UNetEncoder(i_ch)
        self.encoder5 = UNetEncoder(i_ch)

        self.compress_e1 = nn.Conv2d(i_ch * 2 * 5, i_ch * 2, kernel_size=1)
        self.compress_e2 = nn.Conv2d(i_ch * 4 * 5, i_ch * 4, kernel_size=1)
        self.compress_e3 = nn.Conv2d(i_ch * 8 * 5, i_ch * 8, kernel_size=1)
        self.compress_e4 = nn.Conv2d(i_ch * 16 * 5, i_ch * 16, kernel_size=1)

        self.bottleneck = UNetBottleneck(i_ch)
        self.decoder = UNetDecoder(i_ch)
        self.output = nn.Conv2d(i_ch, out_channels, kernel_size=1)
    def freeze_encoder(self,freeze=True):
        for param in self.encoder1.parameters():
            param.requires_grad = not freeze
        for param in self.encoder2.parameters():
            param.requires_grad = not freeze
        for param in self.encoder3.parameters():
            param.requires_grad = not freeze
        for param in self.encoder4.parameters():
            param.requires_grad = not freeze
        for param in self.encoder5.parameters():
            param.requires_grad = not freeze
    def forward(self, x1,x2,x3,x4,x5):
        e1_1,e2_1,e3_1,e4_1 = self.encoder1(self.embedder1(x1))
        e1_2,e2_2,e3_2,e4_2 = self.encoder2(self.embedder2(x2))
        e1_3,e2_3,e3_3,e4_3 = self.encoder3(self.embedder3(x3))
        e1_4,e2_4,e3_4,e4_4 = self.encoder4(self.embedder4(x4))
        e1_5,e2_5,e3_5,e4_5 = self.encoder5(self.embedder5(x5))

        e1 = torch.cat([e1_1,e1_2,e1_3,e1_4,e1_5],dim=1)
        e2 = torch.cat([e2_1,e2_2,e2_3,e2_4,e2_5],dim=1)
        e3 = torch.cat([e3_1,e3_2,e3_3,e3_4,e3_5],dim=1)
        e4 = torch.cat([e4_1,e4_2,e4_3,e4_4,e4_5],dim=1)

        e1 = self.compress_e1(e1)
        e2 = self.compress_e2(e2)
        e3 = self.compress_e3(e3)
        e4 = self.compress_e4(e4)

        b = self.bottleneck(e4)
        d = self.decoder(b, e1, e2, e3, e4)
        return self.output(d)

class UNet(nn.Module):
    def __init__(self,in_channels, i_ch=16):
        super(UNet, self).__init__()
        self.embedder = nn.Sequential(
            ResBlock(in_channels, i_ch),  # in_channelsx128x128 -> i_chx128x128
            ResBlock(i_ch, i_ch),  # i_chx128x128 -> i_chx128x128
        )
        self.encoder = UNetEncoder(i_ch)
        self.bottle = UNetBottleneck(i_ch)
        self.decoder = UNetDecoder(i_ch)
        self.output = nn.Sequential(
            ResBlock(i_ch, i_ch),  # in_channelsx128x128 -> i_chx128x128
            nn.Conv2d(i_ch, 1, kernel_size=1)
        )
    def freeze_encoder(self,freeze=True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        for param in self.embedder.parameters():
            param.requires_grad = not freeze
    def forward(self, x):
        e1,e2,e3,e4 = self.encoder(self.embedder(x))
        b = self.bottle(e4)
        d = self.decoder(b, e1, e2, e3, e4)
        return self.output(d)
        
        
class KappaPredictor(nn.Module):
    def __init__(self):
        super(KappaPredictor, self).__init__()
        self.channel_adder = ChannelAdder()
        self.bn = nn.BatchNorm2d(20)
        self.unet = UNet(20,i_ch=16)#MultiHeadUNet(2, 1,i_ch=16)
    def freeze_encoder(self,freeze=True):
        if freeze:
            for param in self.bn.parameters():
                param.requires_grad = not freeze
        self.unet.freeze_encoder(freeze)
    def forward(self, x):
        x = self.channel_adder(x)
        x = self.bn(x)
        #x1,x2,x3,x4,x5 = torch.split(x, 2, dim=1)
        out = self.unet(x)
        return out

# %% [markdown]
# ## Creación particion inicial

# %%
def create_next_partitions(current_partition,k=PARTITION_SIZE):#Creacion de particiones train y valid
    init_partition = []
    for i in range(k):
        init_partition += load_partition(f"partition_{k*current_partition+(i+1)}.json")
        #print(f"partition_{k*current_partition+(i+1)}.json")

    train_partition, val_partition = train_test_split(init_partition, test_size=0.2, random_state=42)
    save_partition(f"partition_{current_partition+1}_train.json","partitions",train_partition)
    save_partition(f"partition_{current_partition+1}_val.json","partitions",val_partition)

# %%
def load_best_model_so_far(model, last_saved_partition):
    best_loss = float('inf')
    best_partition = 0
    for partition in range(1,last_saved_partition+1):
        checkpoint = torch.load(f"models/last_checkpoint_partition_{partition}.pth",weights_only=False)
        if checkpoint['best_loss'] < best_loss:
            best_loss = checkpoint['best_loss']
            best_partition = partition
    model.load_state_dict(torch.load(f"models/best_model_partition_{best_partition}.pth",weights_only=True))
    print(f"Loaded best model from partition {best_partition} with loss {best_loss}")
def load_last_best_model(model, last_saved_partition):
    #for from last_saved_partition to 1 looking if exists a best model file. if there is then load it and return
    for partition in range(last_saved_partition,0,-1):
        if os.path.exists(f"models/best_model_partition_{partition}.pth"):
            model.load_state_dict(torch.load(f"models/best_model_partition_{partition}.pth",weights_only=True))
            print(f"Loaded best model from partition {partition}")
            return
    print(f"There is no best model saved")

# %%
def many_partitions_v2(start,end,model,criterion,transform=None,full_frecuency=5):
    for current_partition in range(start,end):
        if current_partition %full_frecuency==1:
            print(f"Partition {current_partition}: training full")
            model.freeze_encoder(False)
        else:
            print(f"Partition {current_partition}: training decoder")
            model.freeze_encoder(True)
        load_last_best_model(model,current_partition-1)
        create_next_partitions(current_partition-1)
        train_loader, val_loader = create_train_valid_loaders(
            f"partition_{current_partition}_train.json",
            f"partition_{current_partition}_val.json",
            "partitions",
            batch_size=BATCH_SIZE,
            transform=transform,
        )
        best_model_path = os.path.join(
            "models", f"best_model_partition_{current_partition}.pth"
        )
        last_checkpoint_path = os.path.join(
            "models", f"last_checkpoint_partition_{current_partition}.pth"
        )
        optimizer = optim.AdamW(model.parameters())
        train_model(
            model,
            train_loader,
            val_loader,
            best_model_path,
            last_checkpoint_path,
            criterion,
            optimizer,
            num_epochs=100,
            device=DEVICE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            use_autocast=USE_AUTOCAST,
            early_stopping_grace_period=EARLY_STOPPING_GRACE_PERIOD,
        )
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path,weights_only=True))
            test_model(model,criterion,device=DEVICE,batch_size=BATCH_SIZE,experiment_name=f"adamw_atunet_freeze__{current_partition}")
        else:
            print("No best model found in partition",current_partition)
        

# %% [markdown]
# ## Carga modelo

# %% [markdown]
# 

# %%
class L2LogLoss(nn.Module):
    def __init__(self):
        super(L2LogLoss, self).__init__()
    def rescale(self, x):
        return x.sign()*(x.abs().log1p())
    def forward(self, y_pred, y_true):
        return F.mse_loss(self.rescale(y_pred),self.rescale(y_true))
class L1LogLoss(nn.Module):
    def __init__(self):
        super(L1LogLoss, self).__init__()
    def rescale(self, x):
        return x.sign()*(x.abs().log1p())
    def forward(self, y_pred, y_true):
        return F.l1_loss(self.rescale(y_pred),self.rescale(y_true))

# %%
#Define model
model = KappaPredictor().to(DEVICE)
# Define Loss
criterion = nn.L1Loss()
transform = RandomTransform()

# %%
print("Parametros entrenables del modelo: ",sum([p.numel() for p in model.parameters() if p.requires_grad]))

# %%
many_partitions_v2(1,10,model,criterion,transform=transform)


