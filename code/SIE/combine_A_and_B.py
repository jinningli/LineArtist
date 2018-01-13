import os
import numpy as np
import cv2
from distribute import mkdir


def combine():
    fold_A = os.path.join("Cache", "SketchImage")
    fold_B = os.path.join("Cache", "ResizedImage")
    mkdir("Datasets")
    fold_AB = "Datasets/Dataset_AB"

    splits = os.listdir(fold_A)

    for sp in splits:
        img_fold_A = os.path.join(fold_A, sp)
        img_fold_B = os.path.join(fold_B, sp)
        if sp == ".DS_Store":
            continue
        img_list = os.listdir(img_fold_A)

        num_imgs = len(img_list)
        print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
        img_fold_AB = os.path.join(fold_AB, sp)
        if not os.path.isdir(img_fold_AB):
            os.makedirs(img_fold_AB)
        print('split = %s, number of images = %d' % (sp, num_imgs))
        for n in range(num_imgs):
            name_A = img_list[n]
            path_A = os.path.join(img_fold_A, name_A)
            name_B = name_A
            path_B = os.path.join(img_fold_B, name_B)
            if os.path.isfile(path_A) and os.path.isfile(path_B):
                if name_A == ".DS_Store" or name_B == ".DS_Store":
                    continue
                name_AB = name_A
                path_AB = os.path.join(img_fold_AB, name_AB)
                im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
                im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)
    os.system("rm -r Cache")
