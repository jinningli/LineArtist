from PIL import Image
import scipy.misc
import numpy as np

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

def imresize(path, tosize = 512):
    img = scipy.misc.imread(path).astype(np.float)
    shape = img.shape
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    rate = 0.0
    if shape[0] > shape[1]:
        rate = tosize / shape[0]
    else:
        rate = tosize / shape[1]
    img = scipy.misc.imresize(img, float(rate))
    return img

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)
