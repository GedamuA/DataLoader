# Gedam A. 2020-01-17
# Center for future media lab 
# UESTC, chengdu china  
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import slim
import cv2

train_batch = 16
# Dataset Path
input_path1 = "/home/gede/VcGAN/Dataset/Train_image/"
test_path = "/home/gede/VcGAN/Dataset/ForTest/"
input_path2 = "/home/gede/VcGAN/Dataset/ForTrain/"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True


#  Dataset Loader
class datareader():
    def __init__(self, train_input1, test_data):
        self.train_input1 = train_input1
        self.test_data = test_data

        self.train_input_image1 = []
        self.train_input_label1 = []
        self.train_input_image_name1 = []

        self.test_data_image = []
        self.test_data_label = []
        self.test_data_image_name = []

        nums1 = 0
        nums2 = len(os.listdir(train_input1))
        list_pic = os.listdir(train_input1)
        list_pic.sort()
        for pic_name in list_pic:
            # generate
            nums1 += 1
            print('Train_image_read:%s %d/%d' % (pic_name, nums1, nums2))
            image_path = os.path.join(train_input1, pic_name)
            self.train_input_image1.append(read_image(image_path))
            self.train_input_image_name1.append(pic_name)
            a = pic_name.split('_')
            self.train_input_label1.append(getlabel(int(a[0][1:3]), 40))

        nums3 = 0
        nums4 = len(os.listdir(test_data))
        for pic_name in os.listdir(test_data):
            nums3 += 1
            print('test_image_read:%s %d/%d' % (pic_name, nums3, nums4))
            image_path = os.path.join(test_data, pic_name)
            self.test_data_image.append(read_image(image_path))
            self.test_data_image_name.append(pic_name)
            a = pic_name.split('_')
            self.test_data_label.append(getlabel(int(a[0][1:3]), 40))

        # init params
        self.test_ptr = 0
        self.test_size = len(self.test_data_label)

    def get_train_data_batch(self, list_id):
        data_image1 = [self.train_input_image1[i] for i in list_id]
        data_label1 = [self.train_input_label1[i] for i in list_id]
        return data_image1, data_label1

    def get_batch_test(self, batch_size):
        if self.test_ptr + batch_size < self.test_size:
            data = self.test_data_image[self.test_ptr:self.test_ptr + batch_size]
            label = self.test_data_label[self.test_ptr:self.test_ptr + batch_size]
            self.test_ptr += batch_size
        else:
            data = self.test_data_image[self.test_ptr:]
            label = self.test_data_label[self.test_ptr:]
            self.test_ptr = 0
        return data, label


def getlabel(label_id, class_num):
    label = [0] * class_num
    label[label_id] = 1
    return label


def read_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 127.5 - 1
    return image


image_real = tf.placeholder(tf.float32, [None, 128, 128, 3])
label_real = tf.placeholder(tf.float32, [None, 40])
# data_loader
dataset = datareader(input_path1, test_path)
train_list = [i for i in range(len(dataset.train_input_image1))]  # mark ==> train_image1
test_list = [i for i in range(len(dataset.test_data_image))]
input_queue = tf.train.slice_input_producer([train_list, test_list], num_epochs=None, shuffle=True, capacity=128)
train_id = tf.train.batch(input_queue[0], batch_size=train_batch, num_threads=2, capacity=128,
                          allow_smaller_final_batch=False)
test_id = tf.train.batch(input_queue[1], batch_size=train_batch, num_threads=2, capacity=128,
                         allow_smaller_final_batch=False)

init = tf.global_variables_initializer()
with tf.Session(config=config) as sess:
    print('Init variable')
    sess.run(tf.local_variables_initializer())
    sess.run(init)
    max_acc = 0

    print('======= start train============')
    print("=====================================")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    train_id_data = sess.run(train_id)
    test_id_data = sess.run(test_id)
    train_data_input1, train_input_label1 = dataset.get_train_data_batch(train_id_data)
    test_data_input, test_data_label = dataset.get_batch_test(test_id)
    print(len(train_input_label1))
    print(len(test_data_label))
