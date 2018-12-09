# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class Hand(data.Dataset):
    
    def __init__(self,root, img_mean, img_std, img_size, transforms=None,train=True ):
        '''
        Get images, divide into train/val set
        '''

        self.train = train
        self.images_root = root

        self._read_txt_file()
    
        if transforms is None:
            normalize = T.Normalize(mean= img_mean,
                                    std= img_std )

            if not train: 
                self.transforms = T.Compose([
                    T.Resize( img_size ),
                    T.CenterCrop( img_size ),
                    T.ToTensor(),
                    normalize
                    ]) 
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop( img_size ),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                    ])
                
    def _read_txt_file(self):
        self.images_path = []
        self.images_labels = []

        if self.train:
            txt_file = self.images_root + "./images/train.txt"
        else:
            txt_file = self.images_root + "./images/test.txt"

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_labels.append(item[1])

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = self.images_root+self.images_path[index]
        label = self.images_labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, int(label)
    
    def __len__(self):
        return len(self.images_path)
