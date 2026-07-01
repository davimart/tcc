"""
Model fine-tunning
"""


import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
from utils.tf_image_data_loader import get_data_loader
from utils.pd_data_loaders import LFWA
import in_rv1

import os
import time
import numpy as np

np.random.seed(1)
tf.random.set_seed(2)

######################################################
######################################################

######################################################
## General parameters
######################################################

PRUEBAS = False

main_data_folder = "data"+os.sep+"LFWa"+os.sep
images_folder = 'lfw_funneled'
data_SV_path = 'results/SV/lfwa_sv/lfwa_SV.pkl'
reweigthing_metric = 'SV_tp_diff' # column of previous df
SCALED = False
TRUNCATED = True
OPPOSITE = False

assert not (True == TRUNCATED and SCALED==True)

NEW_MODEL_NAME = "reweighted_sv_tp_diff_truncated_lfwa"
NEW_RES_PATH = 'results'+os.path.sep+NEW_MODEL_NAME+'_T'+time.strftime('%d_%m_%y__%H_%M')

pretrained_model_path = "results/celebA_inception_resnet_v1_pretraining_T09_02_22__18_41/"
pretrained_model_name = "celebA_inception_resnet_v1_pretraining.h5"


device = "GPU"
gpu_n = 0

EPOCHS = 100
BATCH_SIZE = 128

INPUT_SHAPE = (160,160)
lr = 0.0005

AUTOTUNE = tf.data.experimental.AUTOTUNE

if not os.path.exists(NEW_RES_PATH) and not PRUEBAS:
    os.makedirs(NEW_RES_PATH)

# custom tf execution
physical_devices = tf.config.list_physical_devices(device)
print(physical_devices)
if device == "GPU" and "GPU" in physical_devices[gpu_n]:
    tf.config.set_visible_devices(physical_devices[gpu_n], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_n],True)
else:
    tf.config.set_visible_devices(physical_devices[:], 'CPU')
    tf.config.experimental.set_memory_growth(physical_devices[:],True)

######################################################
######################################################

pretrained_inception = tf.keras.models.load_model(pretrained_model_path+pretrained_model_name) # custom_objects={'InceptionResNetV1': InceptionResNetV1})

# Freeze all the layers but the last one
for layer in pretrained_inception.layers[:-1]:
    layer.trainable = False

#####################
# Load LFW datasets
print('##Get LFWA Data Loader')

data_pd_loader = LFWA(main_data_folder, images_folder)
df_data = data_pd_loader.load_dataset()
df_SV = pd.read_pickle('results/SV/lfwa_sv/lfwa_SV.pkl')

if OPPOSITE:
    df_SV[reweigthing_metric] = df_SV[reweigthing_metric]*-1
if SCALED:
    df_SV[reweigthing_metric] = df_SV[reweigthing_metric]-df_SV[reweigthing_metric].min()
if TRUNCATED:
    df_SV[reweigthing_metric][df_SV[reweigthing_metric]<0]=0


df_train = df_data[df_data["partition"]==0]
assert df_train.shape[0] == df_SV.shape[0]
df_train = pd.concat([df_train, df_SV[reweigthing_metric]], axis=1) #add SV column to DF
df_train[reweigthing_metric] = df_train[reweigthing_metric].astype('float64')
df_val = df_data[df_data["partition"]==1]

print("### Generating data loaders")
train_ds = get_data_loader(df_train, label='Male', weight=reweigthing_metric,
                            partition="train",
                            input_shape=INPUT_SHAPE,
                            batch_size=BATCH_SIZE,
                            augment=True)

print("#### Generated train data loader")
validation_ds = get_data_loader(df_val, label='Male',
                                partition="val",
                                input_shape=INPUT_SHAPE,
                                batch_size=BATCH_SIZE,
                                augment=False)


#Train parameters for model.fit with generators
STEP_SIZE_TRAIN = len(df_train) // BATCH_SIZE
STEP_SIZE_VALID = len(df_val) // BATCH_SIZE

#Compile, save diagram and fit
my_callbacks = [CSVLogger(NEW_RES_PATH+os.path.sep+NEW_MODEL_NAME+'.csv', 
                          separator=";",
                          append=False),

                ModelCheckpoint(filepath=NEW_RES_PATH+os.path.sep+NEW_MODEL_NAME+'.h5', #.{epoch:02d}-{val_loss:.2f}
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True),
                                
                EarlyStopping(monitor='val_loss', mode='min',
                              verbose=1, patience=50, min_delta=5e-4),

                ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=3, min_lr=1e-7, 
                                          min_delta=1e-7,
                                          verbose=1)
                ]

print("#### Generating model...")
pretrained_inception.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.Accuracy(),
                         tf.keras.metrics.FalsePositives(),
                         tf.keras.metrics.FalseNegatives(),
                         tf.keras.metrics.TruePositives(),
                         tf.keras.metrics.TrueNegatives()])

print("### Checking trainable layers")
for layer in pretrained_inception.layers:
    if layer.trainable == True:
        print(layer.trainable,
         " - Layer name:", layer.name, 
         " - # Params:", layer.trainable_weights[0].shape)

print("#### Model compiled!")

print("#### Starting training")
#tf.keras.utils.plot_model(model, to_file=RES_PATH+os.path.sep+MODEL_NAME+".png", show_shapes=True, show_layer_names=True, rankdir="TD")
history =  pretrained_inception.fit(train_ds,
                     epochs=EPOCHS,
                     validation_data = validation_ds,
                     callbacks = my_callbacks,
                     #batch_size=train_loader.batch_size,
                     steps_per_epoch = STEP_SIZE_TRAIN,
                     validation_steps = STEP_SIZE_VALID,
                     verbose=1,
                     max_queue_size = 300)


pretrained_inception.save(NEW_RES_PATH+os.path.sep+'FINAL_'+NEW_MODEL_NAME+'.h5')