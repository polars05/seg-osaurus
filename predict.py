import os
import math
import numpy as np
import matplotlib.pyplot as plt
import PIL

import keras

from segmentation_models import Unet

import utils

PATH_TO_WEIGHTS = "/media/dh/DATA/ndsc-advanced/forTry/seg_weights.15-0.52.hdf5"
PATCH_SZ = 64   # should be divisible by 32
N_CLASSES = 3   # roads, buildings, background

def main():
    out_dir = "/media/dh/DATA/ndsc-advanced/forTry/results"
    img_path = "/media/dh/DATA/ndsc-advanced/forTry/results/22828990_15.tiff"
    img = PIL.Image.open(img_path)
    img = img.convert('RGB')
    img = np.asarray(img) #[H, W, C]
    img = utils.normalize(img)

    model = keras.models.load_model(PATH_TO_WEIGHTS)
    #model.compile()
    
    preds = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
    print (preds.shape)

    map = picture_from_mask(preds, 0.7)

    map = int(255/map.shape[2]) * np.sum(map, axis=2)
    print (map.shape)
    plt.imshow(map)
    plt.show()

    filename_w_ext = os.path.basename(img_path)
    filename, ext = os.path.splitext(filename_w_ext)
    plt.savefig(os.path.join(out_dir, filename + "_out.png"))

    #img = PIL.Image.fromarray(map, mode="I")    
    #img.save(os.path.join(out_dir, filename + "_out.png"), "PNG")

def predict(x, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
   
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    
    return prediction[:img_height, :img_width, :]

def picture_from_mask(mask, threshold=0.5):
    pict = np.ones(shape=(mask.shape[0], mask.shape[1], mask.shape[2]), dtype=np.uint8)
    
    for i in range(mask.shape[2]):
        pict[:,:,i] = i*(mask[:,:,i] > threshold).astype(int)
        #plt.imshow(pict[:,:,i])
        #plt.show()
 
    return pict



if __name__ == '__main__':
    main()