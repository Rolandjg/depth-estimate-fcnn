import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = FCN()
model.load_state_dict(torch.load("final.pth", map_location=device))
model.to(device)
model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("test.jpg").convert("RGB")
x = image_transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(x)


depth = pred.squeeze().unsqueeze(0).unsqueeze(0)
depth = F.interpolate(
    depth,
    size=(img.height, img.width),
    mode="bilinear",
    align_corners=False
)

depth = depth.squeeze().cpu().numpy()

example_inputs = (torch.randn(1, 3, 224, 224),)
onnx = torch.onnx.export(model, example_inputs, dynamo=True)
onnx.save("model.onnx")

plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("depth")
plt.imshow(depth, cmap="PuRd")


plt.show()
