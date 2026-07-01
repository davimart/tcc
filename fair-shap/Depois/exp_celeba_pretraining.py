"""
Model training
"""

import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from utils.tf_image_data_loader import get_data_loader
from utils.pd_data_loaders import CelebA
#from inception_resnet_v1 import InceptionResNetV1
from in_rv1 import build_inception_rn_v1

import os
import time

start = time.time()

######################################################
## General parameters
######################################################

PRUEBAS = False

main_data_folder = "data"+os.sep+"celebA"+os.sep
images_folder = main_data_folder+os.sep+'img_align_celeba'+os.sep+'img_align_celeba'+os.sep

device = "GPU"
gpu_n = 0

EPOCHS = 100
BATCH_SIZE = 128

INPUT_SHAPE = (160,160)
lr = 0.001

AUTOTUNE = tf.data.experimental.AUTOTUNE

MODEL_NAME = "celebA_irv1_pretraining"
RES_PATH = 'results'+os.path.sep+MODEL_NAME+'_T'+time.strftime('%d_%m_%y__%H_%M') 
if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)

# custom tf execution
physical_devices = tf.config.list_physical_devices(device)
print(physical_devices)
if device == "GPU" and "GPU" in physical_devices[gpu_n]:
    tf.config.set_visible_devices(physical_devices[gpu_n], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_n],True)
else:
    tf.config.set_visible_devices(physical_devices[:], 'CPU')
    #tf.config.experimental.set_memory_growth(physical_devices[:],True)"""

######################################################
######################################################


## import the data set that include the attribute for each picture
pd_loader = CelebA(main_data_folder, images_folder)
df_train = pd_loader.df_train
df_validation = pd_loader.df_validation

print("### Generating data loaders")
train_ds = get_data_loader(df_train, label='Male',
                            partition="train",
                            input_shape=INPUT_SHAPE,
                            batch_size=BATCH_SIZE,
                            augment=True,
                            crop='CelebA')
print("#### Generated train data loader")
validation_ds = get_data_loader(df_validation, label='Male',
                                partition="val",
                                input_shape=INPUT_SHAPE,
                                batch_size=BATCH_SIZE,
                                augment=False,
                                crop='CelebA')

print("#### Generated validation and test data loaders")

end = time.time()
print(end-start)

#Train parameters for model.fit with generators
STEP_SIZE_TRAIN = len(df_train) // BATCH_SIZE
STEP_SIZE_VALID = len(df_validation) // BATCH_SIZE

#Compile, save diagram and fit
my_callbacks = [CSVLogger(RES_PATH+os.path.sep+MODEL_NAME+'.csv', 
                          separator=";",
                          append=False),

                ModelCheckpoint(filepath=RES_PATH+os.path.sep+MODEL_NAME+'.h5', #.{epoch:02d}-{val_loss:.2f}
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True),
                                
                EarlyStopping(monitor='val_loss', mode='min',
                              verbose=1, patience=20, min_delta=5e-4),

                ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=4, min_lr=1e-7, 
                                          min_delta=1e-7,
                                          verbose=1)
                ]

print("#### Generating model...")
#model = InceptionResNetV1(input_shape=INPUT_SHAPE+(3,), bottleneck_layer_size=1)
model = build_inception_rn_v1(input_shape=INPUT_SHAPE+(3,), out_dim=1)

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.FalsePositives(),
                         tf.keras.metrics.FalseNegatives(),
                         tf.keras.metrics.TruePositives(),
                         tf.keras.metrics.TrueNegatives()])

print("#### Model compiled!")
end = time.time()
print(end-start)
print("#### Starting training")
#tf.keras.utils.plot_model(model, to_file=RES_PATH+os.path.sep+MODEL_NAME+".png", show_shapes=True, show_layer_names=True, rankdir="TD")
history =  model.fit(train_ds,
                     epochs=EPOCHS,
                     validation_data = validation_ds,
                     callbacks = my_callbacks,
                     #batch_size=train_loader.batch_size,
                     steps_per_epoch = STEP_SIZE_TRAIN,
                     validation_steps = STEP_SIZE_VALID,
                     verbose=1,
                     max_queue_size = 300)


model.save(RES_PATH+os.path.sep+'FINAL_'+MODEL_NAME+'.h5')

"""
model = keras.models.load_model('path_to_my_model.h5')

or

# Recrea el modelo
model.save_weights('path_to_my_weights', save_format='tf')

new_model = get_model()
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop())

# Esto inicializa las variables utilizadas por los optimizadores,
# así como cualquier variable métrica con estado
new_model.train_on_batch(x_train[:1], y_train[:1])

# Cargue el estado del modelo anterior.
new_model.load_weights('path_to_my_weights')
"""

