# -*- coding: utf-8 -*-


from torch.utils.data.dataset import Dataset


class Mydatasets(Dataset):
    def __init__(self, img_X, img_y, bsz):
        self.X = img_X
        self.y = img_y
        assert len(img_X) == len(img_y)  # 查看x和y的数量是否相同
        # 长度只能为bsz的整数倍
        self.len = len(self.X)
        length = self.len // bsz * bsz
        self.len = length
        self.X = self.X[:self.len]
        self.y = self.y[:self.len]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        return X, y
