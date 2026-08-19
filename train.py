from datasets import load_dataset
from datasets.utils.logging import set_verbosity_info

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

set_verbosity_info()

print("downloading...")
ds = load_dataset("sayakpaul/nyu_depth_v2", split="train", token=token, trust_remote_code=True)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def depth_to_tensor(depth):
    # Resize depth map too
    depth = TF.resize(
        depth,
        size=(224, 224),
        interpolation=InterpolationMode.BILINEAR
    )

    depth = np.array(depth, dtype=np.float32)
    depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

    return depth

def apply_transform(batch):
    batch["image"] = [
        image_transform(img.convert("RGB"))
        for img in batch["image"]
    ]

    batch["depth_map"] = [
        depth_to_tensor(depth)
        for depth in batch["depth_map"]
    ]

    return batch

ds.set_transform(apply_transform)

# train/test split:
train_size = int(0.8 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = random_split(ds, [train_size, val_size])

train_loader = DataLoader(train_ds, shuffle=True, batch_size=8)
val_loader = DataLoader(val_ds, shuffle=True, batch_size=8)

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(16)

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=96, kernel_size=5, stride=2, padding=2) # 224x224 input
        self.bn1 = nn.BatchNorm2d(96)        

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=2) # 112x112 input
        self.bn2 = nn.BatchNorm2d(256)        

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=5, stride=1, padding=2) # 56x56 input
        self.bn3 = nn.BatchNorm2d(384)        

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=2048, kernel_size=5, stride=4, padding=2) # 14x14 input
        self.bn4 = nn.BatchNorm2d(2048)        

        self.conv5 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=5, stride=1, padding=2) # 14x14 input
        self.bn5 = nn.BatchNorm2d(2048)        

        self.conv6 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=5, stride=1, padding=2) # 14x14 input
        self.bn6 = nn.BatchNorm2d(1024)        

        ### ---------- Decoder ----------

        self.upscale1 = nn.ConvTranspose2d(in_channels=1024, out_channels=720, kernel_size=5 ,stride=2, padding=2, output_padding=1) 
        self.bn7 = nn.BatchNorm2d(720)

        self.smoothConv1 = nn.Conv2d(in_channels=720, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.smoothbn1 = nn.BatchNorm2d(512)

        self.upscale2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5 ,stride=2, padding=2, output_padding=1) 
        self.bn8 = nn.BatchNorm2d(256)        

        self.smoothConv2 = nn.Conv2d(in_channels=256, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.smoothbn2 = nn.BatchNorm2d(192)

        self.upscale3 = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=5 ,stride=2, padding=2, output_padding=1) 
        self.bn9 = nn.BatchNorm2d(96)        

        self.smoothConv3 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.smoothbn3 = nn.BatchNorm2d(64)

        self.upscale4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=5 ,stride=2, padding=2, output_padding=1) 
        self.bn10 = nn.BatchNorm2d(16)

        self.output = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(1)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.bn0(self.conv0(x))) # 224x224x16
        identity0 = x

        x = self.relu(self.bn1(self.conv1(x)))
        identity1 = x # 112x112x96

        x = self.relu(self.bn2(self.conv2(x)))
        identity2 = x # 56x56x256

        x = self.relu(self.bn3(self.conv3(x)))

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))

        ### ---------- Decoder ----------

        x = self.relu(self.bn7(self.upscale1(x)))               # 28x28x512
        x = self.relu(self.smoothbn1(self.smoothConv1(x)))

        x = self.relu(self.bn8(self.upscale2(x)) + identity2)   # 56x56x256
        x = self.relu(self.smoothbn2(self.smoothConv2(x)))
        
        x = self.relu(self.bn9(self.upscale3(x)) + identity1)   # 112x112x96
        x = self.relu(self.smoothbn3(self.smoothConv3(x)))

        x = self.relu(self.bn10(self.upscale4(x)) + identity0)  # 224x224x16

        x = self.relu(self.output(x)) # 224x224x1

        return x


model = FCN().to(device)

LR = 0.001
EPOCHS = 30
CRIT = nn.SmoothL1Loss()
OPT = optim.AdamW(lr=LR, params=model.parameters())
PLOT_INTERVAL = 100
VALIDATION_TESTS = 35
SAVE_EACH = 5

from tqdm.notebook import tqdm

def plot_loss(history, val_history, epoch):
    plt.plot(history, color='orange', label="train")
    plt.plot(val_history, color='blue', label="validation")
    plt.ylabel("Loss")
    plt.xlabel("Training")
    plt.legend()
    plt.savefig(f"figures/loss_{epoch}.jpg")

def test(sample, epoch):
    model.eval()
            
    with torch.no_grad():
        pred = model(val_ds[sample]["image"].to(device).unsqueeze(0))

    pred = pred.squeeze().cpu()
    true = val_ds[sample]["depth_map"].squeeze().cpu()

    def unnormalize_image(img):
        # img: [3, H, W]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        img = img.cpu() * std + mean
        img = img.clamp(0, 1)

        return img.permute(1, 2, 0)  # [H, W, 3]

    img_vis = unnormalize_image(val_ds[0]["image"])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_vis)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap="plasma")
    plt.title("Predicted Depth")
    plt.axis("off")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(true, cmap="plasma")
    plt.title("True Depth")
    plt.axis("off")
    plt.colorbar()

    plt.savefig(f"figures/validation{epoch}.jpg")


print("training model")
loss_history = []
val_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        leave=True,
        dynamic_ncols=True
    )

    progress_bar_val = tqdm(
        val_loader,
        total=VALIDATION_TESTS,
        leave=True,
        dynamic_ncols=True
    )

    for i, batch in enumerate(progress_bar):
        images = batch["image"].to(device)
        labels = batch["depth_map"].to(device)

        OPT.zero_grad()

        predictions = model(images)  

        loss = CRIT(predictions, labels)

        loss.backward()
        OPT.step()

        epoch_loss += loss.item()

        avg_loss = epoch_loss / (i + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        if epoch + 1 % SAVE_EACH == 0:
            torch.save(model.state_dict(), f"models/epoch_{epoch}.pth")

        if i % PLOT_INTERVAL == 0:
            print(avg_loss)
            loss_history.append(avg_loss)

            model.eval()
            val_loss = 0.0
            val_count = 0

            with torch.no_grad():
                progress_bar_val = tqdm(
                    val_loader,
                    total=VALIDATION_TESTS,
                    desc="Validation",
                    leave=False,
                    dynamic_ncols=True
                )

                for j, batch in enumerate(progress_bar_val):
                    if j >= VALIDATION_TESTS:
                        break

                    val_images = batch["image"].to(device)
                    val_labels = batch["depth_map"].to(device)

                    val_pred = model(val_images)
                    loss_v = CRIT(val_pred, val_labels)

                    val_loss += loss_v.item()
                    val_count += 1

                    progress_bar_val.set_postfix(val_loss=f"{val_loss / val_count:.4f}")

            val_history.append(val_loss / val_count)

            model.train()


    test(0, epoch)
    plot_loss(loss_history, val_history, epoch)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader):.8f}")

torch.save(model.state_dict(), "models/final.pth")

