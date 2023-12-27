from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        #print(dataset_path)
        len = 0
        data_list = []
        d = {"background": 0, "kart": 1,"pickup": 2,"nitro": 3,"bomb": 4,"projectile": 5}
        with open(dataset_path+'/labels.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                #print(row['file'], row['label'])
                len += 1
                l = d.get(row['label'])
                data_list.append([row['file'],l])
        self.dataset_path = dataset_path
        self.len = len
        self.data_list = data_list

        #raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        return self.len

        #raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        img = Image.open(self.dataset_path+"/"+self.data_list[idx][0])
        label = self.data_list[idx][1]
        
        x = transforms.functional.to_tensor(img)
        #print(x.shape)
        #print(img.size)

        res = (x, label)
        return(res)
        #raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
