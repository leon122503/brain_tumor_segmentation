import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block with two 3D convolutions and batch normalization.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First convolution
        x = self.relu(self.bn1(self.conv1(x)))
        # Second convolution
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResidualConvBlock(nn.Module):
    """
    Convolutional block with two 3D convolutions, batch normalization, and a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection: 1x1 conv if channels change, identity otherwise
        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
            self.residual_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x

        # First convolution
        out = self.relu(self.bn1(self.conv1(x)))

        # Second convolution
        out = self.bn2(self.conv2(out)) # Apply BN before adding residual

        # Apply residual connection
        if hasattr(self, 'residual_bn'): # Check if 1x1 conv was used
             residual = self.residual_bn(self.residual_conv(residual))
        else:
             residual = self.residual_conv(residual) # Apply identity if channels match

        out += residual
        out = self.relu(out) # Apply final ReLU

        return out


class DepthwiseSeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolutional block for 3D data.
    Consists of depthwise 3D convolution followed by pointwise 1x1x1 convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConvBlock, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels
        )
        self.bn1 = nn.BatchNorm3d(in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Second depthwise separable conv
        self.depthwise2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            groups=out_channels,
        )
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.pointwise2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.bn4 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First depthwise separable convolution
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Second depthwise separable convolution
        x = self.depthwise2(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.pointwise2(x)
        x = self.bn4(x)
        x = self.relu(x)

        return x


class AttentionGate(nn.Module):
    """
    Attention Gate for 3D UNet to focus on relevant features.
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class DownSample(nn.Module):
    """
    Downsample using max pooling.
    """

    def __init__(self):
        super(DownSample, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)


class UpSample(nn.Module):
    """
    Upsample using transposed convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNet3D(nn.Module):
    """
    3D U-Net model for brain tumor segmentation.
    """

    def __init__(self, in_channels=4, out_channels=4, init_features=16):
        super(UNet3D, self).__init__()

        # Initial number of features
        features = init_features

        # Encoder pathway
        self.encoder1 = ConvBlock(in_channels, features)
        self.down1 = DownSample()

        self.encoder2 = ConvBlock(features, features * 2)
        self.down2 = DownSample()

        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.down3 = DownSample()

        # Adding back the fourth layer
        self.encoder4 = ConvBlock(features * 4, features * 8)
        self.down4 = DownSample()

        # Bottom
        self.bottom = ConvBlock(features * 8, features * 16)

        # Decoder pathway
        self.up4 = UpSample(features * 16, features * 8)
        self.decoder4 = ConvBlock(features * 16, features * 8)

        self.up3 = UpSample(features * 8, features * 4)
        self.decoder3 = ConvBlock(features * 8, features * 4)

        self.up2 = UpSample(features * 4, features * 2)
        self.decoder2 = ConvBlock(features * 4, features * 2)

        self.up1 = UpSample(features * 2, features)
        self.decoder1 = ConvBlock(features * 2, features)

        # Final layer
        self.final = nn.Conv3d(features, out_channels, kernel_size=1)

        # Print model parameters for debugging
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {total_params/1e6:.2f}M parameters")

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        x = self.down1(enc1)

        enc2 = self.encoder2(x)
        x = self.down2(enc2)

        enc3 = self.encoder3(x)
        x = self.down3(enc3)

        # Fourth encoder level
        enc4 = self.encoder4(x)
        x = self.down4(enc4)

        # Bottom
        x = self.bottom(x)

        # Decoder path with skip connections
        x = self.up4(x)
        x = torch.cat((x, enc4), dim=1)
        x = self.decoder4(x)

        x = self.up3(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.decoder3(x)

        x = self.up2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.decoder2(x)

        x = self.up1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.decoder1(x)

        # Final layer
        x = self.final(x)

        return x


class AttentionUNet3D(nn.Module):
    """
    3D U-Net model with attention gates and residual connections for brain tumor segmentation.
    """

    def __init__(self, in_channels=4, out_channels=4, init_features=16):
        super(AttentionUNet3D, self).__init__()

        # Initial number of features
        features = init_features

        # Encoder pathway using ResidualConvBlock
        self.encoder1 = ResidualConvBlock(in_channels, features)
        self.down1 = DownSample()

        self.encoder2 = ResidualConvBlock(features, features * 2)
        self.down2 = DownSample()

        self.encoder3 = ResidualConvBlock(features * 2, features * 4)
        self.down3 = DownSample()

        self.encoder4 = ResidualConvBlock(features * 4, features * 8)
        self.down4 = DownSample()

        # Bottom using ResidualConvBlock
        self.bottom = ResidualConvBlock(features * 8, features * 16)

        # Attention Gates - Fix the channel dimensions here
        self.attention4 = AttentionGate(
            F_g=features * 8, F_l=features * 8, F_int=features * 4
        )
        self.attention3 = AttentionGate(
            F_g=features * 4, F_l=features * 4, F_int=features * 2
        )
        self.attention2 = AttentionGate(
            F_g=features * 2, F_l=features * 2, F_int=features
        )
        self.attention1 = AttentionGate(
            F_g=features, F_l=features, F_int=features // 2 if features > 1 else 1
        )

        # Decoder pathway using ResidualConvBlock
        self.up4 = UpSample(features * 16, features * 8)
        self.decoder4 = ResidualConvBlock(features * 16, features * 8)

        self.up3 = UpSample(features * 8, features * 4)
        self.decoder3 = ResidualConvBlock(features * 8, features * 4)

        self.up2 = UpSample(features * 4, features * 2)
        self.decoder2 = ResidualConvBlock(features * 4, features * 2)

        self.up1 = UpSample(features * 2, features)
        self.decoder1 = ResidualConvBlock(features * 2, features)

        # Final layer
        self.final = nn.Conv3d(features, out_channels, kernel_size=1)

        # Print model parameters for debugging
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Residual Attention U-Net initialized with {total_params/1e6:.2f}M parameters")

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        x = self.down1(enc1)

        enc2 = self.encoder2(x)
        x = self.down2(enc2)

        enc3 = self.encoder3(x)
        x = self.down3(enc3)

        enc4 = self.encoder4(x)
        x = self.down4(enc4)

        # Bottom
        x = self.bottom(x)

        # Decoder path with attention gates and skip connections
        x = self.up4(x)
        # Apply attention to the skip connection
        enc4_att = self.attention4(g=x, x=enc4)
        x = torch.cat((x, enc4_att), dim=1)
        x = self.decoder4(x)

        x = self.up3(x)
        enc3_att = self.attention3(g=x, x=enc3)
        x = torch.cat((x, enc3_att), dim=1)
        x = self.decoder3(x)

        x = self.up2(x)
        enc2_att = self.attention2(g=x, x=enc2)
        x = torch.cat((x, enc2_att), dim=1)
        x = self.decoder2(x)

        x = self.up1(x)
        enc1_att = self.attention1(g=x, x=enc1)
        x = torch.cat((x, enc1_att), dim=1)
        x = self.decoder1(x)

        # Final layer
        x = self.final(x)

        return x


class DepthwiseSeparableUNet3D(nn.Module):
    """
    3D U-Net model using depthwise separable convolutions for brain tumor segmentation.
    """

    def __init__(self, in_channels=4, out_channels=4, init_features=16):
        super(DepthwiseSeparableUNet3D, self).__init__()

        # Initial number of features
        features = init_features

        # Encoder pathway
        self.encoder1 = DepthwiseSeparableConvBlock(in_channels, features)
        self.down1 = DownSample()

        self.encoder2 = DepthwiseSeparableConvBlock(features, features * 2)
        self.down2 = DownSample()

        self.encoder3 = DepthwiseSeparableConvBlock(features * 2, features * 4)
        self.down3 = DownSample()

        self.encoder4 = DepthwiseSeparableConvBlock(features * 4, features * 8)
        self.down4 = DownSample()

        # Bottom
        self.bottom = DepthwiseSeparableConvBlock(features * 8, features * 16)

        # Decoder pathway
        self.up4 = UpSample(features * 16, features * 8)
        self.decoder4 = DepthwiseSeparableConvBlock(features * 16, features * 8)

        self.up3 = UpSample(features * 8, features * 4)
        self.decoder3 = DepthwiseSeparableConvBlock(features * 8, features * 4)

        self.up2 = UpSample(features * 4, features * 2)
        self.decoder2 = DepthwiseSeparableConvBlock(features * 4, features * 2)

        self.up1 = UpSample(features * 2, features)
        self.decoder1 = DepthwiseSeparableConvBlock(features * 2, features)

        # Final layer
        self.final = nn.Conv3d(features, out_channels, kernel_size=1)

        # Print model parameters for debugging
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"Depthwise Separable U-Net initialized with {total_params/1e6:.2f}M parameters"
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        x = self.down1(enc1)

        enc2 = self.encoder2(x)
        x = self.down2(enc2)

        enc3 = self.encoder3(x)
        x = self.down3(enc3)

        enc4 = self.encoder4(x)
        x = self.down4(enc4)

        # Bottom
        x = self.bottom(x)

        # Decoder path with skip connections
        x = self.up4(x)
        x = torch.cat((x, enc4), dim=1)
        x = self.decoder4(x)

        x = self.up3(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.decoder3(x)

        x = self.up2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.decoder2(x)

        x = self.up1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.decoder1(x)

        # Final layer
        x = self.final(x)

        return x


def dice_loss(pred, target):
    """
    Dice loss for segmentation.

    Args:
        pred: Predicted tensor, shape [B, C, D, H, W]
        target: Target tensor, shape [B, C, D, H, W]

    Returns:
        Dice loss value
    """
    smooth = 1.0

    # Flatten the tensors
    pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
    target_flat = target.reshape(target.size(0), target.size(1), -1)

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Calculate Dice loss
    dice_loss = 1.0 - dice.mean(dim=1)

    return dice_loss.mean()


def combined_loss(pred, target, alpha=0.5):
    """
    Combined loss function with Dice loss and cross-entropy loss.

    Args:
        pred: Predicted tensor, shape [B, C, D, H, W]
        target: Target tensor, shape [B, C, D, H, W]
        alpha: Weight for dice loss (1-alpha will be weight for CE loss)

    Returns:
        Combined loss value
    """
    # Dice loss
    dice = dice_loss(F.softmax(pred, dim=1), target)

    # Cross-entropy loss
    batch_size, num_classes = pred.size(0), pred.size(1)
    pred_flat = (
        pred.view(batch_size, num_classes, -1)
        .permute(0, 2, 1)
        .contiguous()
        .view(-1, num_classes)
    )
    target_flat = torch.argmax(target, dim=1).view(-1)
    ce_loss = F.cross_entropy(pred_flat, target_flat)

    # Combine losses
    return alpha * dice + (1 - alpha) * ce_loss


def get_model(in_channels=4, out_channels=4, device="cuda", model_type="unet"):
    """
    Initialize and return the selected 3D model.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        device: Device to place model on ('cuda' or 'cpu')
        model_type: Type of model to use ('unet', 'attention_residual_unet', or 'depthwise_unet')

    Returns:
        Initialized 3D U-Net model of the specified type
    """
    if model_type == "attention_residual_unet":
        model = AttentionUNet3D(
            in_channels=in_channels, out_channels=out_channels, init_features=16
        )
    elif model_type == "depthwise_unet":
        model = DepthwiseSeparableUNet3D(
            in_channels=in_channels, out_channels=out_channels, init_features=16
        )
    else:  # default to standard UNet3D
        model = UNet3D(
            in_channels=in_channels, out_channels=out_channels, init_features=16
        )

    model = model.to(device)
    return model
