from imageIO import *
import os
from distribute import distribute
from combine_A_and_B import combine
# Resize the image to smaller size

for root, dirs, files in os.walk("SourceImage"):
    for file in files:
        try:
            if not os.path.exists(os.path.join("ResizedImage", file)):
                imsave(os.path.join("ResizedImage", file), imresize(os.path.join(root, file)))
                print("Resize Image: " + file)
        except:
            print("Not Vilid Image: " + file)

s = input("Now please run the matlab script: smoothing.m.\nPress y after finishing, n for exit: (y/n)\n")
if not (s == "y" or s == "Y"):
    exit()

s = input("Now please run the matlab script: sketching.m.\nPress y after finishing, n for exit: (y/n)\n")
if not (s == "y" or s == "Y"):
    exit()

distribute()
combine()
