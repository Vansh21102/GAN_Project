import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

file_path = 'file path to store fake images'

for i in range(3729):
    fake_img = np.random.randint(0,255,(256,256,3))
    filename = f'fake_{i}.png'
    cv2.imwrite(os.path.join(file_path,filename),fake_img)
