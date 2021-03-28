import numpy as np



data = np.load("C:/Users/tlseh/PycharmProjects/flask3/bottleneck_features/CatResnet50Data.npz")
lst = data.files
for item in lst:
        print(item)
        print(data[item])