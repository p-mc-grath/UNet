import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image 

from os import listdir
from os.path import join, isdir, isfile
from pathlib import Path
import random

from torch.utils.data import Dataset, DataLoader

from ..utils.U_Net_utils import load_json_file, save_json_file

class MISDataset(Dataset):
    def __init__(self, mode, config):
        '''
        Saves relative paths of files in files variable
        full path when joined with self.root

        Arguments:
            mode: One of 'train', 'val', 'test'
            config: as defined in utils
        '''

        super().__init__()
        self.root = config.dir.data.root

        # to load or to save crawled data
        json_file_path = join(self.root, mode + '_' + config.dataset.file_list_name)
       
        # load from json file if possible
        if isfile(json_file_path):
            print('Loading dataset from file: ' + mode + '_' + config.dataset.file_list_name)
            self.files = load_json_file(json_file_path)

        # crawl directories
        else:
            print('Crawling directories for data')

            # allocation
            self.files = {}
            for datatype in config.dataset.datatypes:
                self.files[datatype] = []

            # crawl 
            for datatype in config.dataset.datatypes:
                current_dir_no_root = join(mode, datatype)
                current_dir = join(self.root, current_dir_no_root)
                self.files[datatype] = self.files[datatype] + [join(current_dir_no_root, f) for f in listdir(current_dir)]
            
            print('Your %s dataset consists of %d images' %(mode, len(self.files['rgb'])))

            # TODO check if file names match and are sorted the same

            # save for next time
            Path(self.root).mkdir(exist_ok=True)
            save_json_file(json_file_path, self.files)

    def __getitem__(self, idx):
        '''
        return: 
            dataset items at idx
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load data corresponding to idx
        rgb = transforms.ToTensor()(TF.center_crop(Image.open(join(self.root, self.files['rgb'][idx])), 508))
        depth = transforms.ToTensor()(TF.center_crop(Image.open(join(self.root, self.files['depth'][idx])), 324))
        normals= transforms.ToTensor()(TF.center_crop(Image.open(join(self.root, self.files['normals'][idx])), 324))
    
        return rgb, depth, normals

    def __len__(self):
        '''
        return: 
            number of samples
        '''
        return len(self.files['rgb'])

class MISDataset_Loader(DataLoader):
    def __init__(self, config):
        '''
        Creates MIS dataset(s) and calls pytorch default dataloader
        '''
        
        self.mode = config.loader.mode

        if self.mode == 'train':
            # dataset
            train_set = MISDataset('train', config)
            valid_set = MISDataset('val', config)

            # actual loader
            self.train_loader = DataLoader(train_set, 
                batch_size=config.loader.batch_size, 
                num_workers=config.loader.num_workers,
                pin_memory=config.loader.pin_memory,
                drop_last=config.loader.drop_last)
            self.valid_loader = DataLoader(valid_set, 
                batch_size=config.loader.batch_size,
                num_workers=config.loader.num_workers,
                pin_memory=config.loader.pin_memory,
                drop_last=config.loader.drop_last)
            
            # iterations
            self.train_iterations = (len(train_set) + config.loader.batch_size) // config.loader.batch_size
            self.valid_iterations = (len(valid_set) + config.loader.batch_size) // config.loader.batch_size

        elif self.mode == 'test':
            # dataset
            test_set = MISDataset('test', config)

            # loader also called VALID -> In Agent: valid function == test function; TODO find better solution 
            # !! dataset input is different
            self.valid_loader = DataLoader(test_set, 
                batch_size=config.loader.batch_size,
                num_workers=config.loader.num_workers,
                pin_memory=config.loader.pin_memory,
                drop_last=config.loader.drop_last)
            
            # iterations
            self.valid_iterations = (len(test_set) + config.loader.batch_size) // config.loader.batch_size

        else:
            raise ValueError('Please choose a one of the following modes: train, val, test')
    