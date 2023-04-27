import os
import json
import glob
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from dataset import HuskyDataset

#img_path = "./HuskyTrainingData/HuskyTrainingData/2023-02-08-14-51-49/front_images/"
#measurement_path = "./HuskyTrainingData/HuskyTrainingData/2023-02-08-14-51-49/measurements/"

img_path = "./bc_april4/bc_april4/front_images/"
measurement_path = "./bc_april4/bc_april4/measurements/"

img = glob.glob(img_path + "*.jpg")
img.sort()
measure = glob.glob(measurement_path + "*.txt")
measure.sort()

img_names = []
linear = []
angular = []
measurements = []

img_0 = mpimg.imread(img[0])
plt.imshow(img_0)
plt.show()

for filename in img:
      img_names.append(filename)

for filename in measure:
      my_file = open(filename)
      json_data = my_file.read()
      try:
          json_objects = json_data.split("}{")
          first_obj = json.loads(json_objects[0] + "}")
          linear.append(first_obj["linear"])
          angular.append(first_obj["angular"])
      except:
          obj = json.loads(json_data)
          measurements.append(obj)
          linear.append(obj["linear"])
          angular.append(obj["angular"])

data = {
    "image": img_names,
    "measurements": measurements
}

print(data.keys())

dataframe = pd.DataFrame(data)

transform = transforms.Compose([
    transforms.Resize((200, 88)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = HuskyDataset(dataframe, transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)