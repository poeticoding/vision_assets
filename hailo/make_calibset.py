import numpy as np
import os
from PIL import Image
from random import shuffle
import tensorflow as tf

IMAGES_COUNT = 200
CALIBRATION_IMAGES_DIR = "val2017/"
OUTPUT_DIR = "output_npy"
BATCH_SIZE = 100


image_paths = os.listdir(CALIBRATION_IMAGES_DIR)
shuffle(image_paths)

# first 200
sub_image_paths = image_paths[:IMAGES_COUNT]

# hwc
def open_image(image_path):
	img = Image.open(os.path.join(CALIBRATION_IMAGES_DIR, image_path))
	img_array = np.asarray(img)

	return img_array

def resize_image(image_arr):
    return tf.keras.ops.image.resize(
        image_arr, (640, 640), 
        crop_to_aspect_ratio=False, 
        pad_to_aspect_ratio=True, 
        fill_value=114,
        data_format="channels_last"
    )

calib_dataset = np.zeros(len(sub_image_paths), 640, 640, 3)

for idx, img_name in enumerate(sub_image_paths):
	resized_img_arr = resize_image(open_image(img_name))
	calib_dataset[idx, :, :, :] = resized_img_arr.numpy()

np.save(os.path.join(OUTPUT_DIR, 'calib_set.npy'), calib_dataset)
