import pandas as pd
import os, cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu

def nmi(in_pth, dataset='tcga'):
    if dataset == 'tcga':
        sub_folder = ['_Train_All', '_Valid_All', '_Test_All']
    elif dataset == 'breakhis':
        # sub_folder = ['train', 'test']
        sub_folder = ['benign', 'malignant']
    elif dataset == 'cam16':
        # sub_folder = ['test']
        # sub_folder = ['test.csv']
        sub_folder = ['Raboud']
    else:
        raise Exception('need implementation %s'%dataset)
    nmi_lst = []
    slide_lst = {}

    # test_csv = os.path.join(in_pth, sub_folder[0])
    df = pd.read_csv(in_pth)
    for i in tqdm(range(len(df))):
        img_pth = df.iloc[i, 0]
        img_np = np.asarray(Image.open(img_pth).convert('RGB'))
        img_hsv = np.asarray(Image.open(img_pth).convert('HSV'))

        color_thresh_R,color_thresh_G,color_thresh_B,color_thresh_H = thresh_cal(img_np, img_hsv)
        tissue_mask = _tissue_mask(img_np, img_hsv, color_thresh_R,color_thresh_G,color_thresh_B,color_thresh_H)

        img_tissue = []
        for layer in range(3):
            extract = img_np[:,:,layer]
            img_tissue.append(extract[np.nonzero(tissue_mask)])
        img_tissue = np.array(img_tissue)
        try:
            img_mean = np.mean(img_tissue, axis=0)
            img_median = np.median(img_mean)
            img_95_per = np.percentile(img_mean,95)
            # print(img_median, img_95_per)
            nmi_lst.append(img_median/img_95_per)
        except:
            print('Wrong: ',img_pth)
    # for fold in sub_folder:
    #     fold_pth = os.path.join(in_pth, fold)
    #     fold_pth_fold = os.listdir(fold_pth)
    #     for i in tqdm(range(len(fold_pth_fold))):
    #         # img_pth = df.iloc[i, 0]
    #         img_pth = os.path.join(fold_pth, fold_pth_fold[i])
    #         img_np = np.asarray(Image.open(img_pth).convert('RGB'))
    #         img_hsv = np.asarray(Image.open(img_pth).convert('HSV'))
    #
    #         color_thresh_R,color_thresh_G,color_thresh_B,color_thresh_H = thresh_cal(img_np, img_hsv)
    #         tissue_mask = _tissue_mask(img_np, img_hsv, color_thresh_R,color_thresh_G,color_thresh_B,color_thresh_H)
    #
    #         img_tissue = []
    #         for layer in range(3):
    #             extract = img_np[:,:,layer]
    #             img_tissue.append(extract[np.nonzero(tissue_mask)])
    #         img_tissue = np.array(img_tissue)
    #         try:
    #             img_mean = np.mean(img_tissue, axis=0)
    #             img_median = np.median(img_mean)
    #             img_95_per = np.percentile(img_mean,95)
    #             # print(img_median, img_95_per)
    #             nmi_lst.append(img_median/img_95_per)
    #         except:
    #             print('Wrong: ',img_pth)

    print(nmi_lst)
    overall_mean = np.mean(nmi_lst)
    overall_std = np.std(nmi_lst)
    overall_cv = overall_std / overall_mean
    print('Overall std %.3f; cv %.3f'%(overall_std, overall_cv))

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


def change_csv_path(csv_pth):
    current_pth = '/home/congz3414050/cong/media/data/'
    df = pd.read_csv(csv_pth)
    for i in tqdm(range(len(df))):
        original_pth = df.iloc[i, 0]
        original_pth_split = original_pth.split('/')
        original_pth_remain = ('/').join(original_pth_split[7:])
        new_pth = current_pth + original_pth_remain
        df.at[i, 'image'] = new_pth
    df.to_csv(csv_pth, index=False)

# from sklearn.cluster import KMeans
# import pandas as pd
# import numpy as np
# from tqdm.notebook import tqdm
# from p_tqdm import p_map
# from PIL import Image
# from collections import Counter, defaultdict
# from multiprocessing import Pool, Value, Lock
#
# img_pool = []
# def gen_target(img_pth):
#     img_pil = Image.open(img_pth)
#     img_np = np.array(img_pil)
#     img_mean = np.mean(img_np, axis=(0, 1))
#     return img_mean
#
# def run(csv_pth):
#     df = pd.read_csv(csv_pth)
#     opts_list = df['image'].tolist()
#
#     pool = Pool(processes=8)
#     img_pool = p_map(gen_target, opts_list)
#     kmeans = KMeans(n_clusters=8, random_state=0).fit(img_pool)
#     print(Counter(kmeans.labels_))
#     return kmeans
#
# cluster = run('/home/congz3414050/cong/media/data/fold5/train.csv')
# def gen_csv(cluster, target_label, csv_pth, root):
#     df = pd.read_csv(csv_pth)
#     source_df_pth = root + '/source.csv'
#     source_df_content = {'image':[]}
#     target_df_pth = root + '/target.csv'
#     target_df_content = {'image':[]}
#
#     for i in range(len(df)):
#         if cluster.labels_[i] == target_label:
#             target_df_content['image'].append(df.iloc[i,0])
#         else:
#             source_df_content['image'].append(df.iloc[i,0])
#     target_csv = pd.DataFrame.from_dict(target_df_content)
#     source_csv = pd.DataFrame.from_dict(source_df_content)
#     print('target len: ',len(target_csv))
#     print('source len: ',len(source_csv))
#     target_csv.to_csv(target_df_pth, index=False)
#     source_csv.to_csv(source_df_pth, index=False)


# csv_pth = '/home/congz3414050/cong/media/data/fold5/train.csv'
# root = '/home/congz3414050/cong/media/data/fold5'
# gen_csv(cluster, 1, csv_pth, root)
# change_csv_path('/home/congz3414050/cong/media/data/fold5/train.csv')
# nmi('/home/congz3414050/cong/media/data/fold5/target.csv', dataset='breakhis')
# nmi('/content/drive/MyDrive/TCGA_926_Converted_Patched_Filtered', dataset='tcga')
# nmi('/home/congz3414050/cong/media/data/CAMELYON16', dataset='cam16')




#
# # model.test(test_loader, stage='test', save=True)
# # NMI('/home/congz3414050/cong/media/results/unet_resunet_camelyon/normalised/60/test')
#
df = pd.read_csv('/home/congz3414050/cong/media/data/TCGA_IDH/new_source.csv')
print(len(df))
df = pd.read_csv('/home/congz3414050/cong/media/data/TCGA_IDH/new_target.csv')
print(len(df))