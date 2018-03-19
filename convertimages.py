import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import os
import cv2
from math import floor, ceil, pi
from PIL import Image



folder = './Marcel-Train-Augmenting/A'

def get_image_paths():

    files = os.listdir(folder)
    files.sort()
    files = ['{}/{}'.format(folder, file) for file in files]
    return files

#Convert to jpg images
files = os.listdir(folder)
files.sort()
filePaths = ['{}/{}'.format(folder, file) for file in files]
for index, img in enumerate(filePaths):
    f, e = os.path.splitext(files[index])
    outfile = f + ".jpg"
    im = Image.open(img).save("/home/mnguyen/OpenCvProject/ImageAugmentation/Marcel-Train-Augmenting/A_jpg/"+outfile)
    #im.save(outfile, folder)

