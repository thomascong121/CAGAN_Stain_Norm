import torch
from torch.utils.data import Dataset
from utils.util import image_read
import torchvision.transforms as transforms
import pandas as pd
import cv2


class singlestainData(Dataset):
    """
    :param:
        source_dataframe
        target_dataframe
    :return:
        input: A=target, B=source, #A_paths, B_paths#
    """

    def __init__(self, opt, stage):
        self.opt = opt
        self.df = pd.read_csv(opt['test_dataframe']) if stage == 'test' else pd.read_csv(opt['train_dataframe'])
        if self.opt['name'] == 'breakhis':
            self.label_map = {'B': 0.0, 'M': 1.0}
        elif self.opt['name'] == 'tcga':
            self.label_map = {'MU': 0.0, 'WT': 1.0}
        elif self.opt['name'] in ['cam16', 'cam17']:
            self.label_map = {'NORMAL': 0.0, 'TUMOR': 1.0}
        # else:
        #     raise Exception('%s dataset not used' % self.opt['name'])
        self.label_map = {'NORMAL': 0, 'TUMOR': 1}

    @staticmethod
    def name():
        return 'single dataset for test'

    def __len__(self):
        return int(len(self.df))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[[idx % len(self.df)]]
        label_str = row.iloc[0, self.opt['label_index']]
        # label = label_str
        transform, rgb = image_read(self.opt, row, augment_fn='None', img_index=self.opt['image_index'])
        # print(label_str, self.label_map, self.label_map[label], torch.FloatTensor(self.label_map[label]), torch.tensor(2))
        return transform, rgb, torch.tensor(self.label_map[label_str])


class alignedstainData(Dataset):
    """
    :param:
        source_dataframe
        target_dataframe
    :return:
        input: A=target, B=source, #A_paths, B_paths#
    """

    def __init__(self, opt):
        self.opt = opt
        self.source_df = pd.read_csv(opt['source_dataframe'])
        self.target_df = pd.read_csv(opt['target_dataframe'])
        print('source vs target ', len(self.source_df), len(self.target_df))

    @staticmethod
    def name():
        return 'aligned dataset for Pix2pix-based'

    def __len__(self):
        target_length = int(len(self.target_df))
        source_length = int(len(self.source_df))
        return target_length if target_length > source_length else source_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        target_row = self.target_df.iloc[[idx % len(self.target_df)]]
        source_row = self.source_df.iloc[[idx % len(self.source_df)]]
        target_transform, target_rgb = image_read(self.opt, target_row, augment_fn=self.opt['augment_fn'],
                                                  img_index=self.opt['image_index'])
        source_transform, source_rgb = image_read(self.opt, source_row, augment_fn=self.opt['augment_fn'],
                                                  img_index=self.opt['image_index'])
        return {'target': (target_transform, target_rgb), 'source': (source_transform, source_rgb)}
