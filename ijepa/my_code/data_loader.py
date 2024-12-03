from torch.utils.data import Dataset as TorchDataset
import torch


class Transformed_Cifar(TorchDataset):
    def __init__(self, data, transform):
        self.data   = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        img  = self.data[i]['img']
        label  = self.data[i]['label']
        transformed_img = self.transform(img)
        return transformed_img, label
