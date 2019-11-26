'''https://www.nexmo.com/blog/2018/12/04/dog-breed-detector-using-machine-learning-dr'''
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, Input
from keras.applications import xception
from keras.utils import to_categorical

data_path = os.getcwd()+'/Data-dog-breed-identification/'
df = pd.read_csv(data_path+'labels.csv')

breed_count = df.groupby(by='breed', as_index=False).agg({'id': pd.Series.nunique})
breed_select = breed_count[:12]
breed_select_list =  list(breed_select['breed'])
selectdf = df[df['breed'].isin(breed_select_list)]

label_enc = LabelEncoder()
np.random.seed(seed=7)
rnd = np.random.random(len(selectdf))

y_total = label_enc.fit_transform(selectdf["breed"].values)
train_idx = rnd < 0.9
valid_idx = rnd >= 0.9

y_train = y_total[train_idx]
y_valid = y_total[valid_idx]

x_total = selectdf["id"].values
x_train = x_total[train_idx]
x_valid = x_total[valid_idx]

input_image_train = []
input_image_valid = []
resize_dim = (299,299)
for i in range(0,len(x_train)) :
    img = cv2.imread(data_path+'train/'+x_train[i]+'.jpg')
    img_resize = cv2.resize(img,resize_dim)
    input_image_train.append(img_resize)
input_image_train = np.asarray(input_image_train)
for i in range(0,len(x_valid)) :
    img = cv2.imread(data_path+'train/'+x_valid[i]+'.jpg')
    img_resize = cv2.resize(img,resize_dim)
    input_image_valid.append(img_resize)
input_image_valid = np.asarray(input_image_valid)


y_train_onehot = to_categorical(y_train, num_classes=12)
y_valid_onehot = to_categorical(y_valid, num_classes=12)
y_train_onehot

base_model = xception.Xception(weights=os.getcwd()+'/pretrained_model/xception_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False,input_shape=(299,299,3))
base_model.summary()

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output  ## <tf.Tensor 'block14_sepconv2_act_5/Relu:0' shape=(?, ?, ?, 2048) dtype=float32>

x = BatchNormalization()(x)  ### <tf.Tensor 'batch_normalization_27/cond/Merge:0' shape=(?, ?, ?, 2048) dtype=float32>

x = GlobalAveragePooling2D()(x)  ###<tf.Tensor 'global_average_pooling2d_8/Mean:0' shape=(?, 2048) dtype=float32>

x = Dropout(1.5)(x)
x = Dense(1500,activation = 'relu')(x)
x = Dropout(1.5)(x)

x = Dense(1024,activation = 'relu')(x)

x = Dropout(1.5)(x)

predictions = Dense(12,activation = 'softmax')(x)
##<tf.Tensor 'input_1:0' shape=(?, 299, 299, 3) dtype=float32>  base_model.input
model = Model(inputs = base_model.input, outputs = predictions) # This model will include all layers required in the computation of predictions given base_model.input.

optimizer = RMSprop(lr=0.01, rho=0.9)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=["accuracy"])

batch_size=50
filepath=os.getcwd()+"/model_xception_dogbreed.hdf5"       ##----------------------
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(x=input_image_train, y=y_train_onehot, batch_size=batch_size, epochs=100, verbose=1, callbacks=callbacks_list, validation_data=(input_image_valid, y_valid_onehot), shuffle=True)
