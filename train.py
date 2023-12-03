import random
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout, Input
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import tensorflow as tf
from PIL import Image, ImageFile
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.metrics import top_k_categorical_accuracy
from sklearn.utils import shuffle
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S as EfficientNetV2
import os
import sklearn.metrics as sklm
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow_addons.metrics import F1Score
from sklearn.model_selection import train_test_split
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

batch_size = 4
img_width, img_height= 500, 500

LABELS = ["Covid_19", "Tuberclosis", "Viral Pnuemonia", "Bacterial Pneumonia", "Lung Opacity", "Normal"]
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('validation.csv')
def balance_df(dataframe, columns=LABELS):
    label_dict = {}
    max_sum = []
    dataframe_out = dataframe
    for col in columns:
        label_dict[col] = dataframe[col].sum()
        max_sum.append(dataframe[col].sum())
    label_dict = dict(sorted(label_dict.items(), key=lambda item: item[1]))
    for pathology,val in label_dict.items():
        fraction = max((max(max_sum)/val - 2),0)
        sampled_df = dataframe[dataframe[pathology] > 0.3]
        upsampled_df = sampled_df.sample(frac=fraction, replace=True)
        dataframe_out = pd.concat([dataframe_out, upsampled_df])
    label_dict = {}
    for col in columns:
        label_dict[col] = dataframe_out[col].sum()
    print('upsampling')
    print(label_dict)
    return dataframe_out

df_train = balance_df(df_train)
df_test = balance_df(df_test)

def compute_weights(df,class_list=LABELS):
    weights = []
    for each_class in class_list:
        weights.append(list(compute_sample_weight(class_weight='balanced', y = df[each_class])))
    return np.max(np.array(weights),axis = 0)

# df_train['sample_weight'] = compute_weights(df_train)
df_train['sample_weight'] = compute_weights(df_train)
df_test['sample_weight'] = compute_weights(df_test)

for i in range(100):
    df_train = shuffle(df_train)

import math

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_frame, batch_size=10, img_shape=None, augmentation=True, subset = 'training', num_classes=None):
        self.data_frame = data_frame
        self.train_len = len(data_frame)
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.subset = subset
        print(f"Found {self.data_frame.shape[0]} images belonging to {self.num_classes} classes")

    def __len__(self):
        ''' return total number of batches '''
        self.data_frame = shuffle(self.data_frame)
        return math.ceil(self.train_len / self.batch_size)

    def on_epoch_end(self):
        ''' shuffle data after every epoch '''
        # fix on epoch end it's not working, adding shuffle in len for alternative
        pass

    def __data_augmentation(self, img, mode = 'rgb'):
        ''' function for apply some data augmentation '''
        flip_prob = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if flip_prob > 0.5:
            img = tf.image.transpose(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.5)
        img = tf.image.random_contrast(img, 0.5, 1.5)
        img = tf.image.random_saturation(img, 0.5, 1.5)
        if mode == 'rgb':
            img = tf.image.random_hue(img, 0.5)
        img = tfa.image.rotate(img, random.uniform(-90,90) * math.pi / 180)
        
        return tf.image.random_jpeg_quality(img, 30,100)

    def __get_image(self, file_id, crop_height = img_height, crop_width = img_height, crop_prob = 0):
        """ open image with file_id path and apply data augmentation """
        file_id = 'data/'+file_id+'.tiff'
        
        if self.subset!= 'training':
            img = Image.open(file_id).convert('RGB')
            img = np.asarray(img.resize((self.img_shape[0], self.img_shape[1])))
        else:
            if random.randint(0,5) > 3:
                img = Image.open(file_id).convert('L').convert('RGB')
                img = np.asarray(img.resize((self.img_shape[0], self.img_shape[1])))
                if self.augmentation:
                    img = self.__data_augmentation(img, mode = 'gray')
            else:
                img = Image.open(file_id).convert('RGB')
                img = np.asarray(img.resize((self.img_shape[0], self.img_shape[1])))
                if self.augmentation: 
                    img = self.__data_augmentation(img)
        if crop_prob == 1:
            img = tf.image.random_crop(img, size=[crop_height, crop_width, 3])
        img = tf.cast(img, tf.float32)
        return img

    def __getitem__(self, idx):
        batch_x = self.data_frame["Image"][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_vector = []
        for label in LABELS:
            annotation = self.data_frame[label][idx * self.batch_size:(idx + 1) * self.batch_size].values
            batch_y_vector.append(annotation)
        '''
        batch_y_vector= [self.data_frame["Normal Fundus"][idx * self.batch_size:(idx + 1) * self.batch_size].values]
        '''
        crop_height = random.randint(img_height*.7,img_height*1)
        crop_width = random.randint(img_width*.7,img_width*1)
        if self.subset == 'training':
            crop_prob = random.randint(0,2)
        else:
            crop_prob = 0
        x = [self.__get_image(file_id,crop_height = crop_height, crop_width = crop_width, crop_prob = crop_prob) for file_id in batch_x]
        batch_y_vector = np.array(batch_y_vector).T.tolist()
        y_vector = [label_id for label_id in batch_y_vector]
        y_vector = tf.cast(y_vector, tf.int32)
        batch_weights = [self.data_frame["sample_weight"][idx * self.batch_size:(idx + 1) * self.batch_size].values]
        batch_weights = np.array(batch_weights).T.tolist()
        weight_vector = [label_id for label_id in batch_weights]
        weight_vector = tf.cast(weight_vector, tf.float32)
        return tf.convert_to_tensor(x), tf.convert_to_tensor(y_vector), tf.convert_to_tensor(weight_vector)

custom_train_generator = CustomDataGenerator(data_frame = df_train, batch_size = batch_size, img_shape = (img_height, img_width, 3))
custom_test_generator = CustomDataGenerator(data_frame = df_test, batch_size = batch_size//2, img_shape = (img_height, img_width, 3), subset = 'test', augmentation = False)
strategy = tf.distribute.MirroredStrategy()
adm= Adam(learning_rate = 1e-4)
loss = BinaryCrossentropy()
with strategy.scope():
    input_tensor = Input(shape=(None,None,3))
    conv_base = EfficientNetV2(weights='imagenet',include_top=False,input_tensor=input_tensor)
    a_map = layers.Conv2D(512, 1, strides=(1, 1), padding="same", activation='relu')(conv_base.output)
    a_map = layers.Conv2D(1, 1, strides=(1, 1), padding="same", activation='relu')(a_map)
    a_map = layers.Conv2D(1280, 1, strides=(1, 1), padding="same", activation='sigmoid')(a_map)
    res = layers.Multiply()([conv_base.output, a_map])
    x = GlobalAveragePooling2D()(res)
    x = Dropout(0.5)(x)
    predictions = Dense(len(LABELS), activation='sigmoid', name='final_output')(x)
    model = Model(input_tensor, predictions)
    del conv_base
    del a_map
    del res
    model.compile(optimizer=adm, loss=loss,metrics=['accuracy'])

mcp_save = ModelCheckpoint('chest_diseases_{epoch:03d}--{loss:03f}--{accuracy:03f}--{val_loss:03f}--{val_accuracy:03f}.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')
model.fit(
    custom_train_generator,
    validation_data = custom_test_generator,
    epochs = 1000,
    verbose = 1,
    workers = 16,
    max_queue_size = 1,
    callbacks=[mcp_save])