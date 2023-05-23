import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    '''
    This will convert x and y data to torch dataset
    '''
    def __init__(self,data,target,transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x,y

    def __len__(self):
        return len(self.data)

def convert_to_tensor_dataset(train_x,train_y,val_x,val_y):
    '''
    Will return torch dataset
    :param train_x:
    :param train_y:
    :param val_x:
    :param val_y:
    :return:
    '''
    # test_x,test_y
    train_data = MyDataset(train_x, train_y)
    val_data = MyDataset(val_x, val_y)
    # test_data = MyDataset(test_x, test_y)
    # , test_data
    return train_data,val_data


def dataloader(train_data,val_data,batch_size):
    '''
    Data loader
    :param train_data:
    :param val_data:
    :param batch_size:
    :return:
    '''
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader