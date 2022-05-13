import pandas as pd
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from collections import Counter
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import copy
import albumentations as A
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models


class WSIDataset(Dataset):
    """Generate dataset."""

    def __init__(self, filepath, transform=None, bag=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_data = filepath
        self.transform = transform
        self.patient = {}

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.all_data.iloc[idx, 2]
        label_idx = self.all_data.iloc[idx, 0]
        p_id = self.all_data.iloc[idx, 1]
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        label = torch.nn.functional.one_hot(torch.LongTensor([label_idx]), num_classes=2)
        if p_id not in self.patient:
            self.patient[p_id] = len(self.patient) + 1
        patient_idx = self.patient[p_id]
        sample = {'image': image, 'label': label, 'pid': int(patient_idx)}  # , 'pth':img_path}
        if self.transform:
            sample = self.transform(sample)

        return sample

class My_Transform(object):
    '''
    My transform:
    '''

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, pid = sample['image'], sample['label'], sample['pid']
        aug = A.Compose([
            A.Resize(256, 256, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

        augmented = aug(image=image)
        image_medium = augmented['image']
        return {'image': image_medium, 'label': label, 'pid': pid}


class My_Normalize(object):
    '''
    My Normalize (TRail)
    '''

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, pid = sample['image'], sample['label'], sample['pid']
        normal_aug = A.Compose([
            A.Resize(256, 256, p=1),
            A.Normalize()
        ])
        augmented_img = normal_aug(image=image)
        image = augmented_img['image']
        # image = image/255.0
        # print('normal ',image.shape)
        return {'image': image, 'label': label, 'pid': pid}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print(list(sample.keys()))
        image, label, pid = sample['image'], sample['label'], sample['pid']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'label': label.squeeze(0),
                'pid': torch.FloatTensor([pid])}

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        # model_ft = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 1024
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def test_model(t_model, testloader, device='cpu', weight_trained=None):
    since = time.time()
    if weight_trained:
        t_model.load_state_dict(torch.load(weight_trained))
    t_model.eval()
    p_classification = defaultdict(list)
    p_classification_prob = defaultdict(list)
    p_label = {}
    TP = TN = FP = FN = 1e-4
    slide_label = []
    slide_prob = []
    for data in tqdm(testloader):
        inputs = data['image'].to(device=device, dtype=torch.float)
        labels = data['label'].to(device=device, dtype=torch.int64)
        _, label_index = torch.max(labels, 1)
        p_id = data['pid']
        # print('pid ',p_id)
        output = t_model(inputs)
        _, prediction = torch.max(output, 1)
        for i in range(output.size()[0]):
            curr_p_id = p_id[i].item()
            p_label[curr_p_id] = label_index[i].cpu().item()
            p_classification[curr_p_id].append(prediction[i].cpu().item())
            p_classification_prob[curr_p_id].append([output[i][0].cpu().item(), output[i][1].cpu().item()])
            if prediction[i].cpu().item() == label_index[i].cpu().item() == 0:
                TN += 1
            elif prediction[i].cpu().item() == label_index[i].cpu().item()  == 1:
                TP += 1
            elif prediction[i].cpu().item() == 0 and label_index[i].cpu().item()  == 1:
                FN += 1
            else:
                FP += 1
            slide_label.append(label_index[i].cpu().item())
            slide_prob.append(output[i][1].cpu().item())
    Specificity = TN / (TN + FP)#Specificity = TN / (TN + FP)
    Sensitivity = TP / (FN + TP)#Sensitivity = TP / (FN + TP)
    Precision = TP / (TP + FP)#Precision = TP / (TP + FP)
    F1_Score = 2*(Precision * Sensitivity) / (Precision + Sensitivity)
    AUC = roc_auc_score(slide_label, slide_prob)
    print(TP, TN, FN, FP)
    print('Patch level Statistic: ')
    print('Specificity: ', Specificity)
    print('Sensitivity: ', Sensitivity)
    print('Precision: ', Precision)
    print('Acc: ', (TP+TN)/(TP+TN+FN+FP))
    print('F1-Score: ',F1_Score)
    print('AUC ',AUC)
    return p_label, p_classification, p_classification_prob

def patent_class(p_label, p_classification, p_classification_prob, FP_lst, FN_lst):
    p_classification_check = defaultdict(list)

    TP = TN = FP = FN = 1e-4
    pat_label = []
    pat_pred = []
    pat_prob = []
    for k in p_classification:
        #classification of slide for one patient
        p_list = p_classification[k]
        x = Counter(p_list)
        s = sum(p_list)
        #majority vote for patient level classification
        p_pred = x.most_common(1)[0][0]
        #keep the prob for the positive label
        p_prob = p_classification_prob[k]
        p_pos = list(map(lambda x:x[1], p_prob))
        p_pos = sum(p_pos)/len(p_pos)
        pat_prob.append(p_pos)
        p_gt = p_label[k]
        if p_pred == p_gt == 0:
            TN += 1
        elif p_pred == p_gt  == 1:
            TP += 1
        elif p_pred == 0 and p_gt  == 1:
            FN += 1
            if k not in FN_lst:
                FN_lst[k] = 1
            else:
                FN_lst[k] += 1
        else:
            FP += 1
            if k not in FP_lst:
                FP_lst[k] = 1
            else:
                FP_lst[k] += 1
        pat_label.append(p_gt)
        pat_pred.append(p_pred)
    Specificity = TN / (TN + FP)#Specificity = TN / (TN + FP)
    Sensitivity = TP / (FN + TP)#Sensitivity = TP / (FN + TP)
    Precision = TP / (TP + FP)#Precision = TP / (TP + FP)
    F1_Score = 2*(Precision * Sensitivity) / (Precision + Sensitivity)
    # print(pat_label)
    # print(pat_prob)
    AUC = roc_auc_score(pat_label, pat_prob)
    ns_fpr, ns_tpr, _ = roc_curve(pat_label, pat_prob)
    print(TP, TN, FN, FP)
    print('FN ',FN_lst)
    print('FP ',FP_lst)
    print('Statistic: ')
    print('Specificity: ', Specificity)
    print('Sensitivity: ', Sensitivity)
    print('Precision: ', Precision)
    print('Acc: ', (TP+TN)/(TP+TN+FN+FP))
    print('F1-Score: ',F1_Score)
    print('AUC ',AUC)
    return (TP+TN)/(TP+TN+FN+FP), F1_Score, AUC, FP_lst, FN_lst

@hydra.main(config_path='./configs', config_name='config.yaml')
def train_model(cfg: DictConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('\n Using : ', device)
    if cfg.dataset.opt_dataset['name'] == 'cam17':
        all_data_csv = cfg.dataset.opt_dataset['all_data_dataframe']
        all_data = pd.read_csv(all_data_csv)
        all_data = all_data.sample(frac=1).reset_index(drop=True)
        train_data = all_data.head(int(0.7 * len(all_data)))
        teat_data = all_data.tail(int(0.3 * len(all_data)))
    else:
        train_data = pd.read_csv(cfg.dataset.opt_dataset['train_dataframe'])
        teat_data = pd.read_csv(cfg.dataset.opt_dataset['test_dataframe'])
    train_wsi_dataset = WSIDataset(train_data, transform=transforms.Compose([My_Transform(),
                                                                             My_Normalize(),
                                                                             ToTensor()]))
    test_wsi_dataset = WSIDataset(teat_data, transform=transforms.Compose([My_Normalize(),
                                                                           ToTensor()]))

    trainloader = DataLoader(train_wsi_dataset,
                             batch_size=cfg.run.opt_run['batchSize'],
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True)
    testloader = DataLoader(test_wsi_dataset,
                            batch_size=cfg.run.opt_run['batchSize'],
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)

    model, _ = initialize_model(cfg.model.classifier['which_model'],
                                            cfg.dataset.opt_dataset['num_class'],
                                            False, use_pretrained=True)
    since = time.time()
    model = model.to(device)
    dataloaders = {'train': trainloader, 'test': testloader}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    test_acc_history = []
    train_acc_history = []
    FP_lst = {}
    FN_lst = {}
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(cfg.run.opt_run['n_epoch']):
        print('Epoch {}/{}'.format(epoch, cfg.run.opt_run['n_epoch'] - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val1']:
            # for phase in ['val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data in tqdm(dataloaders[phase]):
                    inputs = data['image'].to(device)
                    labels = data['label'].to(device=device, dtype=torch.float32)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    _, label_index = torch.max(labels, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == label_index.data)
                    # break
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                scheduler.step(epoch_loss)

            else:
                model.eval()  # Set model to evaluate mode
                p_label, p_classification, p_classification_prob = test_model(model, testloader, device)
                acc, f1, auc, FP_lst, FN_lst = patent_class(p_label, p_classification, p_classification_prob, FP_lst,
                                                            FN_lst)
                print('Current ACC ', acc)
                print('Best ACC ', best_acc)
                # deep copy the model
                if acc > best_acc:
                    best_acc = acc

            if phase == 'test':
                test_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history

if __name__ == '__main__':
    model_trained, train_acc_history, val_acc_history = train_model()
