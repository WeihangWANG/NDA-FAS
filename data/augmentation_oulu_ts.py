from imgaug import augmenters as iaa
import random
import math
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
from einops import rearrange
from .prepare_data import *

from torchvision import transforms as T
transforms = T.Compose([
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0),
                    T.ToTensor(),
                    # T.Normalize(mean=[0.485, 0.456, 0.406],
                    #             std=[0.229, 0.224, 0.225])
                ])

transforms_one = T.Compose([
                    # T.RandomPosterize(4,p=1),
                    # T.GaussianBlur((3,3),sigma=(0.1,2.0)),
                    T.RandomEqualize(p=1),
                    T.ToTensor(),
                    # T.Normalize(mean=[0.485, 0.456, 0.406],
                    #             std=[0.229, 0.224, 0.225])
                ])

transforms_two = T.Compose([
                    T.RandomPosterize(5,p=1),
                    T.ToTensor(),
                    # T.Normalize(mean=[0.485, 0.456, 0.406],
                    #             std=[0.229, 0.224, 0.225])
                ])

transforms_pos = T.Compose([
                    T.ToTensor(),
                    # T.Normalize(mean=[0.485, 0.456, 0.406],
                    #             std=[0.229, 0.224, 0.225])
                ])

def TTA_9_cropps_color(image, target_shape=(48, 48, 3)):
    # image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    _, width, height = image.shape
    target_w, target_h, _ = target_shape
    #
    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2
    # start_x = 40
    # start_y = 40
    #
    starts = [[start_x - target_w, start_y - target_w],[start_x - target_w, start_y],[start_x - target_w, start_y + target_w],
              [start_x, start_y - target_w],[start_x, start_y],[start_x, start_y + target_w],
              [start_x + target_w, start_y - target_w],[start_x + target_w, start_y],[start_x + target_w, start_y + target_w],
              ]

    images = []

    for start_index in starts:
        # image_ = image.copy()
        image_ = image.clone()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w - 1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h - 1

        zeros = image_[:, x:x + target_w, y: y + target_h]

        # image_ = zeros.copy()
        # images.append(image_.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
        images.append(zeros.reshape([1, target_shape[2], target_shape[0], target_shape[1]]))

    return images

def TTA_9_cropps(image, target_shape=(48, 48, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    _, width, height = image.shape
    target_w, target_h, _ = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2
    # start_x = 40
    # start_y = 40
    #
    starts = [[start_x - target_w, start_y - target_w], [start_x - target_w, start_y],
              [start_x - target_w, start_y + target_w],
              [start_x, start_y - target_w], [start_x, start_y], [start_x, start_y + target_w],
              [start_x + target_w, start_y - target_w], [start_x + target_w, start_y],
              [start_x + target_w, start_y + target_w],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w - 1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h - 1

        zeros = image_[x:x + target_w, y: y + target_h]

        image_ = zeros.copy()

        images.append(image_.reshape([1, target_shape[0], target_shape[1], 1]))

    return images

def CutOut(img, length=20):
    h, w = img.shape[0], img.shape[1]    # Tensor [1][2],  nparray [0][1]
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    length_new = np.random.randint(1, length)

    y1 = np.clip(y - length_new // 2, 0, h)
    y2 = np.clip(y + length_new // 2, 0, h)
    x1 = np.clip(x - length_new // 2, 0, w)
    x2 = np.clip(x + length_new // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    img[mask == 0.]= 0.
    return img

def RandomErasing(img, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):

    if random.uniform(0, 1) < probability:
        attempts = np.random.randint(1, 3)
        for attempt in range(attempts):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)

                img[x1:x1+h, y1:y1+w, 0] = mean[0]
                img[x1:x1+h, y1:y1+w, 1] = mean[1]
                img[x1:x1+h, y1:y1+w, 2] = mean[2]
    return img

def random_resize(img):
    # if random.uniform(0, 1) > probability:
    #     return img

    # minRatio = 0.2
    # ratio_h = random.uniform(minRatio, 1.0)
    # ratio_w = random.uniform(minRatio, 1.0)
    #
    # h = img.shape[0]
    # w = img.shape[1]
    #
    # new_h = int(h*ratio_h)
    # new_w = int(w*ratio_w)

    img = cv2.resize(img, (80, 80))
    img = cv2.resize(img, (112, 112))
    return img

def crop_tensor(img):
    patch_size = 48
    stride_size = 32
    channel = 3
    img = torch.unsqueeze(img, 0)

    soft_split = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(stride_size, stride_size))
    img = soft_split(img)

    img = rearrange(img, 'b (c h w) n -> b n c h w', h = patch_size, w = patch_size, c=channel)
    return img

def augumentor_1(img, target_shape=(32, 32, 3)):

    augment_img_neg = iaa.Sequential([
        # iaa.Fliplr(0.5),
        iaa.Add(value=(-20,20),per_channel=True),
        iaa.LinearContrast((0.5,1.5)),
    ])

    augment_img_pos = iaa.Sequential([
        iaa.Fliplr·(0.5),
    ])

    distortion = iaa.Sequential([
        iaa.PiecewiseAffine(scale=(0.01, 0.1)),
    ])

    # mask = np.zeros_like(img)
    # # wid = np.random.randint(1,5)
    # stride = np.random.randint(2,10)
    # ind = np.arange(0, img.shape[1], 2*stride)
    # for x in ind:
    #     mask[:,x:x+stride] = (255,255,255)

    # fil = img.copy()

    zeros = np.zeros_like(img)
    # if random.random() <= 0.5:  # comment for omic
    a = random.uniform(1.2, 1.5)  #0.5,1.5
    b = random.randint(5, 15)  #+/-20
    # fil = cv2.addWeighted(fil.astype(np.uint8), a, zeros, 1-a, b)
    fil = cv2.addWeighted(img, a, zeros, 1-a, b)

    if random.random() <= 0.05:    # 0.05 for ocim & ocmi
        fil = distortion.augment_image(fil)
    
    img_bri = augment_img_neg.augment_image(fil.astype(np.uint8))

    bg = np.zeros((img.shape[0],img.shape[1],3))
    bg[:,:,0] = 0#255#1.0
    bg[:,:,2] = 255#0#0.0
    bg[:,:,1] = random.randint(100,180)#random.uniform(100/255,180/255)
    bg = bg.astype(np.uint8)
    ratio = random.uniform(0.1,0.15)
    # img_color = fil*(1-ratio) + bg*ratio
    img_color = img_bri*(1-ratio) + bg*ratio

    if random.random() <= 0.5:
        img_out = fil
    else:
        img_out = img_bri
    # img_out = fil
    img_color = img_out

    # # if random.random() <= 0.0001:
    # #     # tiaowen
    # #     if random.random() <= 0.5:
    # #         img_out = img_out*0.9 + mask * 0.1
    # #     else:
    # #         img_color = img_color * 0.9 + mask * 0.1

    img = cv2.resize(img, (112,112))
    img_out = cv2.resize(img_out, (112,112))
    img_color = cv2.resize(img_color, (112,112))

    img = Image.fromarray(img)
    img_out = Image.fromarray(img_out.astype(np.uint8))
    img_color = Image.fromarray(img_color.astype(np.uint8))

    img = transforms_pos(img)
    if random.random() <= 0.5:
        img_out = transforms(img_out)
        img_color = transforms(img_color)
    else:
        img_out = transforms_pos(img_out)
        img_color = transforms_pos(img_color)       

    # color = (img - 127.5) / 128.
    # img_color = (img_color - 127.5) / 128.
    # img_out = (img_out - 127.5) / 128.

    color = crop_tensor(img)
    img_clr = crop_tensor(img_color)
    img_out = crop_tensor(img_out)

    return color, img_clr, img_out

## augment for validation and test
def augumentor_2(color, target_shape=(32, 32, 3)):

    # augment_img = iaa.Sequential([
    #     iaa.Fliplr(0.5),
    #     # iaa.Add(value=(-10,10),per_channel=True), #protocol 3&4 wo
    #     # iaa.GammaContrast(gamma=(0.9, 1.1)), #protocol 3&4 wo
    # ])
    # color = augment_img.augment_image(color)
    # color = cv2.resize(color, (112,112))

    color = Image.fromarray(color)
    color = transforms_pos(color)
    # color = (color - 127.5) / 128.
    # color = random_resize(color)
    #color = random_resize(color)
    
    #color = color / 255.0
    ## segmentation
    # src
    # color = image_into_patches(color, target_shape, 0.25)
    # color = TTA_9_cropps_color(color, target_shape)
    color = crop_tensor(color)

    return color

def augumentor_3(img, target_shape=(32, 32, 3)):

    # img = cv2.resize(img, (112,112))
    # img_out = cv2.resize(img_out, (112,112))
    # img_color = cv2.resize(img_color, (112,112))

    img = Image.fromarray(img)
    # img_out = Image.fromarray(img_out.astype(np.uint8))
    # img_color = Image.fromarray(img_color.astype(np.uint8))

    img_out = transforms_one(img)
    img_color = transforms_one(img)
    img = transforms_pos(img)

    # color = (img - 127.5) / 128.
    # img_color = (img_color - 127.5) / 128.
    # img_out = (img_out - 127.5) / 128.

    color = crop_tensor(img)
    img_clr = crop_tensor(img_color)
    img_out = crop_tensor(img_out)

    return color, img_clr, img_out

def image_into_patches(image, target_patch=32, overlap=0.25):
    # print("*****image shape*******")
    # print(image.shape)
    # print(target_patch)
    width, height, _ = image.shape
    overlap_size = int(target_patch[0] * (1 - overlap))

    patch_num = (width - target_patch[0]) // overlap_size + 1

    starts = []
    for i in range(patch_num):
        for j in range(patch_num):
            starts.append([overlap_size * i, overlap_size * j])

    images = []
    for start_index in starts:
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_patch[0] >= width:
            x = width - target_patch[0] - 1
        if y + target_patch[0] >= height:
            y = height - target_patch[0] - 1

        patch = image[x:x + target_patch[0], y: y + target_patch[0]]
        img_ = patch.copy()
        images.append(img_.reshape([1,target_patch[0],target_patch[1],target_patch[2]]))
        
    return images
