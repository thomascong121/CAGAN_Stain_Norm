import random
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.color import rgb2hed, hed2rgb
from sklearn.utils import shuffle
from torch.autograd import Variable
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, keep_dim=True):
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if not keep_dim and image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) * 127.5
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy.astype(imtype)


def img2tensor(image):
    # print('image type ',type(image))
    aug = transforms.Compose([
        transforms.ToTensor(),
    ])
    return aug(image)
    # image = np.array(image)
    # image = image/127.5 - 1
    # return torch.tensor(image).permute(2, 0, 1)


def hed_to_rgb(h, ed):
    """
    Takes a batch of images
    """
    hed = torch.cat([h, ed], dim=1).permute(0, 2, 3, 1).cpu().detach().float().numpy()
    rgb_imgs = []

    for img in hed:
        img_rgb = hed2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def shuffleDf(df):
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)
    return df


def color_transform(opt, image):
    if opt['use_color'] == 'gray':
        to_gray = transforms.Grayscale(3)
        transImage = img2tensor(to_gray(image))
        mask = img2tensor(image)
    elif opt['use_color'] == 'hed':
        Hed = rgb2hed(image)
        transImage = img2tensor(Hed[..., [0]])
        mask = img2tensor(Hed[..., [1, 2]])
    elif opt['use_color'] == 'ycc':
        imgYCC = image.convert('YCbCr')
        y, cb, cr = imgYCC.split()
        transImage = img2tensor(y)
        # cb = img2tensor(cb)
        # cr = img2tensor(cr)
        mask = img2tensor(y.copy())
    else:
        Hed = rgb2hed(image)
        H_comp = Hed[:, :, 0]
        transImage = img2tensor((np.repeat(H_comp[:, :, np.newaxis], 3, -1)))
        mask = img2tensor(image)
    return transImage, mask


def base_aug(opt, image):
    aug_list = []
    if opt['crop']:
        aug_list.append(transforms.CenterCrop(opt['fineSize']))
    else:
        aug_list.append(transforms.Resize((opt['fineSize'], opt['fineSize'])))
    aug = transforms.Compose(aug_list)
    image = aug(image)

    image_transform, mask = color_transform(opt, image)

    return image_transform, mask


def Hed_Aug(img):
    img = np.array(img)
    Hed = rgb2hed(img)
    H = Hed[..., [0]]
    E = Hed[..., [1]]
    D = Hed[..., [2]]

    alpha1 = np.clip(random.random(), a_min=0.9, a_max=1)
    beta1 = np.clip(random.random(), a_min=0, a_max=0.01)

    alpha2 = np.clip(random.random(), a_min=0.9, a_max=1)
    beta2 = np.clip(random.random(), a_min=0, a_max=0.01)

    alpha3 = np.clip(random.random(), a_min=0.9, a_max=1)
    beta3 = np.clip(random.random(), a_min=0, a_max=0.01)

    H = H * alpha1 + beta1
    E = E * alpha2 + beta2
    D = D * alpha3 + beta3

    Hed_cat = np.concatenate((H, E, D), axis=-1)
    Hed_cat = hed2rgb(Hed_cat)
    Hed_cat = np.clip(Hed_cat, a_min=0, a_max=1)
    Hed_cat = Image.fromarray(np.uint8(Hed_cat * 255))
    return Hed_cat


def histo_aug(opt, image):
    if opt['crop']:
        aug_list = [transforms.CenterCrop(opt['fineSize'])]
    else:
        aug_list = [transforms.Resize((opt['fineSize'], opt['fineSize']))]
    aug_list += [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)
    ]
    # apply basic augmentation:
    aug_base = transforms.Compose(aug_list)
    image = aug_base(image)
    image_org, mask_org = color_transform(opt, image)  # y, y_label

    # apply extra augmentations - 1:
    # if random.random() < opt.Training.aug_prob:
    # apply extra augmentations:
    if 0 < random.random() < 0.2:
        image_transform = transforms.GaussianBlur(3)(image)
        image_transform = transforms.functional.adjust_saturation(image_transform, 1.5)
        image_transform = transforms.functional.adjust_contrast(image_transform, 1.5)
        image_transform = Hed_Aug(image_transform)
    elif 0.2 < random.random() < 0.8:
        image_transform = Hed_Aug(image)
    else:
        image_transform = image
    image_transform, mask_transform = color_transform(opt, image_transform)  # y, y_label
    # print('range of images ',torch.min(image_transform), torch.max(image_transform), torch.min(mask_transform), torch.max(mask_transform))
    return image_transform, mask_org  # y_trans, y_label


def image_read(opt, imageRow, augment_fn, img_index=0, opposite=False):
    img_path = imageRow.iloc[0, img_index]
    image = Image.open(img_path)

    if augment_fn == 'None':
        image_aug, rgb_aug = base_aug(opt, image)
    elif augment_fn == 'histo':
        image_aug, rgb_aug = histo_aug(opt, image)
    else:
        raise Exception('Augmentation %s not implemented'%augment_fn)
    return image_aug, rgb_aug


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images

def NMI(file_pth):
    NMI_lst = []
    slide_lst = {}
    target_lst = os.listdir(file_pth)
    for i in tqdm(range(len(target_lst))):
        img_pth = file_pth + '/%s'%(target_lst[i])
        # slide_id = target_lst[i].split('+')[0].split('-')[1]
        img_np = np.asarray(Image.open(img_pth).convert('RGB'))
        img_hsv = np.asarray(Image.open(img_pth).convert('HSV'))

        color_thresh_R, color_thresh_G, color_thresh_B, color_thresh_H = thresh_cal(img_np, img_hsv)
        tissue_mask = _tissue_mask(img_np, img_hsv, color_thresh_R, color_thresh_G, color_thresh_B, color_thresh_H)

        img_tissue = []
        for layer in range(3):
            extract = img_np[:, :, layer]
            img_tissue.append(extract[np.nonzero(tissue_mask)])
        img_tissue = np.array(img_tissue)

        img_mean = np.mean(img_tissue, axis=0)
        img_median = np.median(img_mean)
        img_95_per = np.percentile(img_mean, 95)
        # print(img_median, img_95_per)
        NMI_lst.append(img_median / img_95_per)
        # if slide_id not in slide_lst:
        #     slide_lst[slide_id] = [img_median / img_95_per]
        # else:
        #     slide_lst[slide_id].append(img_median / img_95_per)

    overall_mean = np.mean(NMI_lst)
    overall_std = np.std(NMI_lst)
    overall_cv = overall_std / overall_mean
    print('Overall std %.3f; cv %.3f' % (overall_std, overall_cv))

    # slide_std = 0
    # slide_cv = 0
    # for slide in slide_lst:
    #     one_mean = np.mean(slide_lst[slide])
    #     one_std = np.std(slide_lst[slide])
    #     one_cv = one_std / one_mean
    #     slide_std += one_std
    #     slide_cv += one_cv
    # print('Overall slide-level std %.3f; cv %.3f'%(slide_std/len(slide_lst), slide_cv/len(slide_lst)))


def thresh_cal(img_np, img_hsv):
    color_thresh_R = threshold_otsu(img_np[:, :, 0])
    color_thresh_G = threshold_otsu(img_np[:, :, 1])
    color_thresh_B = threshold_otsu(img_np[:, :, 2])
    color_thresh_H = threshold_otsu(img_hsv[:, :, 1])
    return color_thresh_R,color_thresh_G,color_thresh_B,color_thresh_H

def _tissue_mask(image_np_trans, img_hsv, color_thresh_R, color_thresh_G, color_thresh_B, color_thresh_H):
    background_R = image_np_trans[:, :, 0] > color_thresh_R
    background_G = image_np_trans[:, :, 1] > color_thresh_G
    background_B = image_np_trans[:, :, 2] > color_thresh_B
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_hsv[:, :, 1] > color_thresh_H
    min_R = image_np_trans[:, :, 0] > 50
    min_G = image_np_trans[:, :, 1] > 50
    min_B = image_np_trans[:, :, 2] > 50
    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B  ###############tissue mask

    return tissue_mask  # levl4