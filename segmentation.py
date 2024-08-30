
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm


def load_nifti(imgpath):
    return nib.load(imgpath).get_fdata()


def ccentre(imgslice, x, y):   
    # Tried multiple resize methods .. found cv2 works best
    return cv2.resize(imgslice,(x,y))

def normalizer(imgslice):
    return (imgslice - imgslice.mean()) / imgslice.std()


def segmentation(flair, t1, t2, t1ce, model):
    '''
    Returns a mask of the same size as the input.
    '''
    # Slice each input into 155 slices of size 240x240
    flair = load_nifti(flair)
    t1 = load_nifti(t1)
    t2 = load_nifti(t2)
    t1ce = load_nifti(t1ce)
    
    x,y,z = flair.shape
    
    combined_mask = np.zeros((x, y, z))
    one_batch_combined_slice = np.zeros((1,128,128,4))
    
    for i in tqdm(range(155)):
        
        flair_slice = normalizer(ccentre(flair[:, :, i],128,128))
        t1_slice = normalizer(ccentre(t1[:, :, i],128,128))
        t2_slice = normalizer(ccentre(t2[:, :, i],128,128))
        t1ce_slice = normalizer(ccentre(t1ce[:, :, i],128,128))
        
    
        # Combine the slices to form a 4-channel image
        combined_slice = np.stack([t1_slice, t1ce_slice, t2_slice, flair_slice], axis=-1)
        
        print(f'combined shape {combined_slice.shape}')

        one_batch_combined_slice[0,:,:,:] = combined_slice
        
        # pred
        predicted_mask = model.predict(one_batch_combined_slice)
        
        print(f'predicted mask shape {predicted_mask.shape}')
    
        # Argmax the predicted mask to obtain a hot encoded mask
        hot_encoded_mask = np.argmax(predicted_mask[0], axis=-1)   # That zero works maybe, maybe not do some testing (works .. was right LOL)
        
        print(f'hot encoded shape {hot_encoded_mask.shape}')
        
        hot_encoded_mask = cv2.resize(hot_encoded_mask, (x,y), interpolation=cv2.INTER_NEAREST)
    
        
        combined_mask[:, :, i] = hot_encoded_mask
        
    return combined_mask
    
