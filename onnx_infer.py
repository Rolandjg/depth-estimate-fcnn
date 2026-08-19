import onnxruntime
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import time

print("loading model")
session = onnxruntime.InferenceSession("model.onnx")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

img = Image.open("test.jpg").convert("RGB")
x = np.array(image_transform(img).unsqueeze(0))

start = time.perf_counter()

outputs = session.run(
    None, 
    {session.get_inputs()[0].name: x}
)

end = time.perf_counter()
print((end - start) / 50 * 1000, "ms")

time.sleep(1)


start = time.perf_counter()

outputs = session.run(
    None, 
    {session.get_inputs()[0].name: x}
)

end = time.perf_counter()
print((end - start) / 50 * 1000, "ms")

depth = outputs[0][0]
depth = depth.squeeze()
resize = cv2.resize(depth, dsize=(img.width, img.height), interpolation=cv2.INTER_LINEAR)

plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("depth")
plt.imshow(resize, cmap="PuRd")

plt.show()
