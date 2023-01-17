import sys

import torch.utils.data as data
import PIL.Image as Image
import pandas as pd
import numpy as np


class Loader(data.DataLoader):
    def __init__(self, DB_ROOT_PATH, RUN_TYPE='train', transform=None):
        """
        :param root: DATABASE PATH
        :param run_type: Model run type (train, test, valid)
        :param transform: Image style transform module
        :var
        """
        super(Loader, self).__init__()
        self.DB_ROOT_PATH = DB_ROOT_PATH
        self.RUN_TYPE = RUN_TYPE
        self.transform = transform

        try:
            data_info = pd.read_csv(f'{self.DB_ROOT_PATH}/{self.RUN_TYPE}.csv')
        except Exception as e:
            if e == FileNotFoundError:
                print("해당 경로에 파일이 존재하지 않습니다!!!!")
                print("The file does not exist in that path!!!!")
                sys.exit()
            else:
                print(f'Exception Type : {e}')
                sys.exit()

        assert self.RUN_TYPE == 'train' or self.RUN_TYPE == 'test' or self.RUN_TYPE == 'valid', \
            'Only train, test, and valid are available for run_type.'

        if self.RUN_TYPE == 'test':
            self.path_list = self.get_paths(data_info)

        elif self.RUN_TYPE == 'train' or self.RUN_TYPE == 'valid':
            self.path_list = self.get_paths(data_info)
            self.label_list = self.get_labels(data_info)

    def get_paths(self, data_info):
        """
        :param data_info: DataFrame, DB info [folder, ]
        :return:
        """
        paths_list = []
        paths_info = data_info.iloc[:, 0:2].values
        return self.make_path_list(paths_info, paths_list)

    def make_path_list(self, path_info, paths_list: list):
        for folder, image_name in path_info:
            full_path = f'{self.DB_ROOT_PATH}/{self.RUN_TYPE}/{folder}/{image_name}'
            paths_list.append(full_path)

        return paths_list

    def get_labels(self, data_info):
        return data_info.iloc[:, 2:].values