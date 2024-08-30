import nibabel as nib
import numpy as np
import os
import cv2
from tqdm import tqdm


def loadimg(imgpath):
    return nib.load(imgpath).get_fdata()


def ccentre(imgslice, x, y):   
    # Tried multiple resize methods .. found cv2 works best
    return cv2.resize(imgslice,(x,y))

def normalizer(imgslice):
    return (imgslice - imgslice.mean()) / imgslice.std()

def pathstoids(dirList):
    x = []
    for i in range(len(dirList)):
        x.append(dirList[i].split("/")[-1])
    return x


def create_slices():
    
    train_path = "brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    
    # all data directories
    all_directories = [f.path for f in os.scandir(train_path) if f.is_dir()]

    # This dir has misnaming for the seg file
    all_directories.remove(os.path.join(train_path,'BraTS20_Training_355'))
    all_ids = pathstoids(all_directories)

    types = ['t1','t1ce','t2','flair','seg']

    # Make dirs for each type of data
    for tp in types:
        os.makedirs(f'dataset/{tp}', exist_ok = True)

    slices_created = 0
    imgs = {}
    imgslice = {}

    for i in tqdm(range(len(all_ids))): # total no of dirs
        current_path = os.path.join(train_path ,all_ids[i])
        
        for tp in types: # seg t1 t1ce t2 flair
            imgs[tp] = loadimg(os.path.join(current_path, f'{all_ids[i]}_{tp}.nii')) # imgs['seg'] = npy for brats20_001_flair.nii
        
        for j in range(155):  # as there 155 instances for each .nii file
            
            for name in imgs: # seg t1 t1ce t2 flair
                imgslice[name] = imgs[name][:, :, j] # imgslice['seg'] = (240 240) of npy for brats20_001_seg.nii
                imgslice[name] = ccentre(imgslice[name], 128, 128) # imgslice['seg'] = (128 128) of npy for brats20_001_seg.nii
            
            # Only takes slices which have masking data
            if imgslice['seg'].max() > 0:
                for name in ['t1','t2','t1ce','flair']:
                    imgslice[name] = normalizer(imgslice[name]) # normalize the npy image
                for name in imgslice:
                    np.save(f'./dataset/{name}/img_{slices_created}.npy', imgslice[name])
                slices_created += 1


    print(f'No of slices : {slices_created}')
    
if __name__ == "__main__":
    create_slices()