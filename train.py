import os
import PIL
import numpy as np

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

import keras
#from keras.preprocessing.image import ImageDataGenerator

import utils

########## logging ##########
import logging

logging.basicConfig(
	format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
	datefmt='%m/%d/%Y %H:%M:%S',
	level=logging.INFO
)
logger = logging.getLogger(__name__)
########## /logging ##########



#N_BANDS = 8
N_CLASSES = 3  # buildings, roads, background
#CLASS_WEIGHTS = [0.5, 0.5]
N_EPOCHS = 250
UPCONV = True
PATCH_SZ = 64   # should be divisible by 32
BATCH_SIZE = 192
TRAIN_SZ = 40000  # train size (no. of patches to be generated)
VAL_SZ = 10000 # validation size



X_DICT_TRAIN = dict()
Y_DICT_TRAIN = dict()
X_DICT_VALIDATION = dict()
Y_DICT_VALIDATION = dict()

#PATH_TO_IMAGES = "/media/dh/DATA/ndsc-advanced/forTry/data/train/images"
PATH_TO_IMAGES = "/media/dh/DATA/ssai-cnn/data/mass_merged/train/sat"
#PATH_TO_LABELS = "/media/dh/DATA/ndsc-advanced/forTry/data/train/labels"
PATH_TO_LABELS = "/media/dh/DATA/ssai-cnn/data/mass_merged/train/map"

def main():
    WEIGHTS_PATH = 'weights'

    logger.info("Reading images")
    for file in os.listdir(PATH_TO_IMAGES):
        if file.endswith(".tiff"):
            ##### /read an image in the dataset #####
            img_path = os.path.join(PATH_TO_IMAGES, file)

            img = PIL.Image.open(img_path)
            img = img.convert('RGB')
            img = np.asarray(img) #[H, W, C]
            img = utils.normalize(img)
            ##### /read an image in the dataset #####

            ##### read a label in the dataset #####
            f_name, f_ext = os.path.splitext(file)
            label_path = os.path.join(PATH_TO_LABELS, f_name + ".tif")
            assert os.path.isfile(label_path)
            label = PIL.Image.open(label_path)
            #label = label.convert('RGB')
            label = np.asarray(label) #[H, W, num_classes]
            ##### /read a label in the dataset #####
            
            train_xsz = int(3/4 * img.shape[0])  # use 75% of image as train and 25% for validation
            X_DICT_TRAIN[file] = img[:train_xsz, :, :]
            #print (X_DICT_TRAIN[file].shape)
            Y_DICT_TRAIN[file] = label[:train_xsz, :, :]
            #print (Y_DICT_TRAIN[file].shape)
            X_DICT_VALIDATION[file] = img[train_xsz:, :, :]
            Y_DICT_VALIDATION[file] = label[train_xsz:, :, :]

    logger.info("Training set: {} images".format(len(X_DICT_TRAIN)))
    logger.info("Training set: {} labels".format(len(Y_DICT_TRAIN)))
    logger.info("Validation set: {} images".format(len(X_DICT_VALIDATION)))
    logger.info("Validation set: {} labels".format(len(Y_DICT_VALIDATION)))

    x_train, y_train = utils.get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
    assert len(x_train) == len(y_train)
    logger.info("Generated {} patches for training".format(len(x_train)))

    x_val, y_val = utils.get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
    assert len(x_val) == len(y_val)
    logger.info("Generated {} patches for validation".format(len(x_val)))


    logger.info("########## Training ##########")
    # define model
    model = Unet()
    model = Unet(backbone_name='resnet34', input_shape=(PATCH_SZ, PATCH_SZ, 3), classes=N_CLASSES, encoder_weights=None)
    
    model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

    # load weights (if specified)
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs(WEIGHTS_PATH)
    WEIGHTS_PATH += "/seg_weights.{epoch:02d}-{val_iou_score:.2f}.hdf5"

    ########## define callbacks ##########
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        WEIGHTS_PATH, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        period=5
    )
    
    csv_logger = keras.callbacks.CSVLogger(
        'log_unet.csv', 
        append=True, 
        separator=';'
    )

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='./tensorboard_unet/', 
        write_graph=True, 
        write_images=True
    )
    ########## /define callbacks ##########

    # fit model
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        verbose=1, #show an animated progress bar
        shuffle=True,
        callbacks=[
            model_checkpoint, 
            csv_logger, 
            tensorboard
        ],
        validation_data=(x_val, y_val),
    )



if __name__ == "__main__":
    main()
    # tensorboard --logdir=logs/