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
PARTITION_SIZE = 2

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

# %%
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
            self.grad_magnitude(e),
            e_norm,
            cross_product,
            delta,
            simmilarity #min max and mean simmilarity
        ], dim=1)

        return output
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Channel Adder
        self.channel_adder = ChannelAdder()
        # Encoder
        self.encoder1 = DoubleConv(in_channels+10, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    def freeze_encoder(self,freeze=True):
        for param in self.encoder1.parameters():
            param.requires_grad = not freeze
        for param in self.encoder2.parameters():
            param.requires_grad = not freeze
        for param in self.encoder3.parameters():
            param.requires_grad = not freeze
        for param in self.encoder4.parameters():
            param.requires_grad = not freeze
        for param in self.bottleneck.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        # Agregar features
        x = self.channel_adder(x)
        
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)

        # Final output
        out = self.final_conv(d1)
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
model = UNet(3,1).to(DEVICE)
# Define Loss
criterion = nn.L1Loss()
transform = None#RandomTransform()

# %%
print("Parametros entrenables del modelo: ",sum([p.numel() for p in model.parameters() if p.requires_grad]))

# %%
many_partitions_v2(1,10,model,criterion,transform=transform)


