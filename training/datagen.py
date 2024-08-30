import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
import albumentations as A

IMG_SIZE = 128

# Customized DataGenerator

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_ids, batch_size=32, dim=(IMG_SIZE, IMG_SIZE), n_channels=4, n_classes=4, shuffle=True, augmentation=None):
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indexes]
        X, y = self.__data_generation(list_ids_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype = np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype = np.float32)

        for i, ID in enumerate(list_ids_temp):
            images = {}
            for name in ['t1', 't1ce', 't2', 'flair', 'seg']:
                images[name] = np.load(f'dataset/{name}/img_{ID}.npy')

            if self.augmentation:
                augmented = self.augmentation(image=images['flair'], mask=images['seg'])
                images['flair'] = augmented['image']

            X[i,] = np.stack((images['t1'], images['t1ce'], images['t2'], images['flair']), axis=-1)
            images['seg'][images['seg'] == 4] = 3
            y[i,] = to_categorical(images['seg'], num_classes=self.n_classes)
            
        return X, y
    

augmentation = A.Compose([
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=None, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)                  
    ], p=0.8),
    A.RandomBrightnessContrast(p=0.8)
])