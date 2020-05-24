

import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm


class COVIDsample:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __str__(self):
        return self.label


class COVIDdataset:
    def __init__(self):
        self.dataset = {}
        #self.minsize = -1
        self.minsize = 100
        self.X = np.array([])
        self.y = np.array([])

    def add(self, sample):
        size = sample['img'].shape
        leastdirection = size[1]
        if leastdirection > size[2]:
            leastdirection = size[2]
        f = sample['img'].reshape((size[1],size[2]))
        image_resized = f[(size[1]-leastdirection)//2:(size[1]+leastdirection)//2,(size[2]-leastdirection)//2:(size[2]+leastdirection)//2]

        self.dataset[sample['idx']] = COVIDsample(image_resized,sample['lab'][2])

        if self.minsize == -1:
            self.minsize = leastdirection

        if leastdirection < self.minsize:
            self.minsize = leastdirection

    def normalize(self):
        print()
        print(40 * "=")
        print("Resizing images")
        print(40 * "=")
        for c in tqdm(self.dataset.keys()):
            self.dataset[c].data = resize(self.dataset[c].data,
                                          (self.dataset[c].data.shape[0] * self.minsize // self.dataset[c].data.shape[0],
                                           self.dataset[c].data.shape[1] * self.minsize // self.dataset[c].data.shape[1]),
                                          anti_aliasing=True)

    def vectorize(self):
        for c in self.dataset.keys():
            self.dataset[c].data = self.dataset[c].data.flatten()

    def generateMatrices(self):
        print(40 * "=")
        print("Generating matrices")
        print(40 * "=")
        for c in tqdm(self.dataset.keys()):
            data = self.dataset[c].data.flatten()
            self.X = np.append(self.X, data)
            self.y = np.append(self.y, self.dataset[c].label)
        self.X = self.X.reshape((len(self.dataset.keys()),self.minsize**2))
        return

    def __str__(self):
        return("Size of set: {}".format(len(self.dataset)))
