import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import os
import cv2
from math import floor, ceil, pi
from PIL import Image

def get_image_paths():
    folder = './Cars'

    files = os.listdir(folder)
    files.sort()
    files = ['{}/{}'.format(folder, file) for file in files]
    return files

X_img_paths = get_image_paths()

IMAGE_SIZE = 220

def tf_resize_images(X_img_file_paths):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index, file_path in enumerate(X_img_paths):
            img = mpimg.imread(file_path)[:, :, :3]
            resized_img = sess.run(tf_img, feed_dict={X:img})
            X_data.append(resized_img)

    X_data = np.array(X_data, dtype=np.float32)  # Convert to numpy
 # Convert to numpy
    return X_data

#Resize images to a common size
X_imgs = tf_resize_images(X_img_paths)
print(X_imgs.shape)


#
# matplotlib.rcParams.update({'font.size': 14})
#
# fig, ax = plt.subplots(figsize = (12, 12))
# plt.subplot(1, 2, 1)
# plt.imshow(mpimg.imread(X_img_paths[0])[:,:,:3])
# plt.title('Original Image')
# plt.subplot(1, 2, 2)
# plt.imshow(X_imgs[0])
# plt.title('Resized Image')
# plt.show()



from math import ceil, floor


def get_translate_parameters(index):
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2:  # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE))
    else:  # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE

    return offset, size, w_start, w_end, h_start, h_end



#Translate image to the left, right, up and down

def translate_images(X_imgs):

    offsets = np.zeros((len(X_imgs), 2), dtype=np.float32)
    n_translations = 4
    X_translated_arr = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            X_translated.fill(1.0)  # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)

            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], w_start: w_start + size[1], :] = glimpses

            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype=np.float32)
    return X_translated_arr



# translated_imgs = translate_images(X_imgs)
# print(translated_imgs.shape)

#
# gs = gridspec.GridSpec(1, 5)
# gs.update(wspace = 0.30, hspace = 2)
#
# fig, ax = plt.subplots(figsize = (16, 16))
# plt.subplot(gs[0])
# plt.imshow(X_imgs[0])
# plt.title('Base Image')
# plt.subplot(gs[1])
# plt.imshow(translated_imgs[0])
# plt.title('Left 20 percent')
# plt.subplot(gs[2])
# plt.imshow(translated_imgs[1329])
# plt.title('Right 20 percent')
# plt.subplot(gs[3])
# plt.imshow(translated_imgs[2658])
# plt.title('Top 20 percent')
# plt.subplot(gs[4])
# plt.imshow(translated_imgs[3987])
# plt.title('Bottom 20 percent')
# plt.show()
#



#Rotate images with predefined angles
def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    radian = tf.placeholder(tf.float32, shape=(len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict={X: X_imgs, radian: radian_arr})

            # for i, file  in enumerate(files):
            #     cv2.imwrite(os.path.join(rotated_dir + "/" + "rotated_" + str(degrees_angle) + "_" + files[i]), rotated_imgs[i])
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate


# rotated_imgs = rotate_images(X_imgs, -90, 90, 10)
# print(rotated_imgs.shape)




# matplotlib.rcParams.update({'font.size': 11})
#
# fig, ax = plt.subplots(figsize = (16, 16))
# gs = gridspec.GridSpec(3, 5)
# gs.update(wspace = 0.30, hspace = 0.0002)
#
# plt.subplot(gs[0])
# plt.imshow(X_imgs[5])
# plt.title('Base Image')
#
# for i in range(14):
#     plt.subplot(gs[i + 1])
#     plt.imshow(rotated_imgs[5 + 12 * i])
#     plt.title('Rotate {:.2f} degrees'.format(-90 + 13.846 * i))
# plt.show()




def flip_images(X_imgs):
    # curr_dir = './Marcel-Train-Augmenting/A_jpg'
    # files = os.listdir(curr_dir)
    # files.sort()
    # flipped_dir = "./Marcel-Train-Augmenting/FlippedImages"
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, img in enumerate(X_imgs):
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            # for i, file  in enumerate(flipped_imgs):
            #     cv2.imwrite(os.path.join(flipped_dir + "/" + "flipped_" + str(i) + "_" + files[index]), flipped_imgs[i])

            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip


# flipped_images = flip_images(X_imgs)
# print(flipped_images.shape)


# matplotlib.rcParams.update({'font.size': 14})
#
# fig, ax = plt.subplots(figsize = (10, 10))
# plt.subplot(2, 2, 1)
# plt.imshow(X_imgs[6])
# plt.title('Base Image')
# plt.subplot(2, 2, 2)
# plt.imshow(flipped_images[18])
# plt.title('Flip left right')
# plt.subplot(2, 2, 3)
# plt.imshow(flipped_images[19])
# plt.title('Flip up down')
# plt.subplot(2, 2, 4)
# plt.imshow(flipped_images[20])
# plt.title('Transpose')
# plt.show()




# change perspective
def get_mask_coord(imshape):
    vertices = np.array([(0.09 * imshape[1], 0.99 * imshape[0]),
                          (0.43 * imshape[1], 0.32 * imshape[0]),
                          (0.56 * imshape[1], 0.32 * imshape[0]),
                          (0.85 * imshape[1], 0.99 * imshape[0])], dtype=np.int32)
    return vertices


def get_perspective_matrices(X_img):
    offset = 0
    img_size = (X_img.shape[1], X_img.shape[0])

    # Estimate the coordinates of object of interest inside the image.
    src = np.float32(get_mask_coord(X_img.shape))
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                       [img_size[0] - offset, img_size[1]]])


    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    return perspective_matrix


def perspective_transform(X_img):
    # Doing only for one type of example
    perspective_matrix = get_perspective_matrices(X_img)
    warped_img = cv2.warpPerspective(X_img, perspective_matrix,
                                     (X_img.shape[1], X_img.shape[0]),
                                     flags=cv2.INTER_LINEAR)
    return warped_img


# X_img = X_imgs[0]
# perspective_img = perspective_transform(X_img)
# print(perspective_img.shape)
#
# fig, ax = plt.subplots(figsize = (12, 12))
# plt.subplot(1, 2, 1)
# plt.imshow(X_imgs[0])
# plt.title('Original Image')
# plt.subplot(1, 2, 2)
# plt.imshow(perspective_img)
# plt.title('Different View of Image')
# plt.show()