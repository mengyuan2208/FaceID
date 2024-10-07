import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import argparse

class FileListDataset(Dataset):
    def __init__(self, filelist, transform=None):
        # self.img_lst, self.lb_lst, self.num_classes = self.build_dataset(filelist)
        self.img_lst = self.build_dataset(filelist)
        self.num = len(self.img_lst)
        self.transform = transform

    def build_dataset(self, filelist):
        img_lst = []
        # lb_lst = []
        # lb_max = -1
        # lb_mapping = {}
        with open(filelist) as f:
            for x in f.readlines():
                img_path = x.strip()
                # lb_name = img_path.split('/')[1]
                # if lb_name not in lb_mapping:
                #     lb_mapping[lb_name] = len(lb_mapping)
                # lb_max = max(lb_max, lb_mapping[lb_name])
                img_lst.append(img_path)
        #         lb_lst.append(lb_mapping[lb_name])
        # assert len(img_lst) == len(lb_lst)
        # return img_lst, lb_lst, lb_max
        return img_lst

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        fn = self.img_lst[idx]
        img = Image.open(open(fn, 'rb')).convert('RGB')
        img = self.transform(img)
        return img
    
    # def save_labels_to_file(self, filename):
    #     with open(filename, 'w') as f:
    #         for label in self.lb_lst:
    #             f.write(f"{label}\n")

class IdentityMapping(nn.Module):
    def __init__(self, base):
        super(IdentityMapping, self).__init__()
        self.base = base
        self.base.fc = nn.Linear(self.base.fc.in_features, feature_dim)

    def forward(self, x):
        x = self.base(x)
        return x

def extract(test_loader, model):
    model.eval()
    features = []
    with torch.no_grad():
        for _, x in enumerate(test_loader):
            # compute output
            output = model(x)
            features.append(output.data.cpu().numpy())
    return np.vstack(features)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Preprocess images and extract features")
    parser.add_argument('--filelist', type=str, required=True, help="Path to the file containing the list of images")
    parser.add_argument('--output_feature_path', type=str, required=True, help="Path to save the extracted features")

    args = parser.parse_args()

    # Parameters
    workers = 1
    batch_size = 128
    input_size = 224
    feature_dim = 256
    filelist = args.filelist
    output_feature_path = args.output_feature_path
    # output_label_path = args.output_label_path

    print(f"=> creating model resnet50")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = IdentityMapping(model)

    dataset = FileListDataset(
        filelist,
        transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    
    # print(f'saving labels to {output_label_path}')
    # dataset.save_labels_to_file(output_label_path)

    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=workers,
                             pin_memory=True)

    features = extract(test_loader, model)
    assert features.shape[0] == dataset.num
    assert features.shape[1] == feature_dim

    print(f'saving extracted features to {output_feature_path}')
    folder = os.path.dirname(output_feature_path)
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    features.tofile(output_feature_path)