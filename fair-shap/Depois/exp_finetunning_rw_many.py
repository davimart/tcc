"""
Model fine-tunning
"""

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
from utils.tf_image_data_loader import get_data_loader
from utils.pd_data_loaders import LFWA, CelebA, FairFaces
import in_rv1

import os
import time
import numpy as np

np.random.seed(1)
tf.random.set_seed(2)

# custom tf execution
device = "GPU"
gpu_n = 0
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

######################################################
## General parameters
######################################################
PRUEBAS = False

pretrained_model_path = "results/celebA_irv1_pretraining_T28_03_22__16_45/celebA_irv1_pretraining.h5"
LAYER_POS = -7

data_SV_path = 'results/SV/lfwa_sv/lfwa_SV.pkl'
TRUNCATED = True #exps only with truncated measure
reweigthing_metrics_names = ['SV_acc', 'SV_EOp', 'SV_EOp_bounded', 'SV_tp_diff', 'SV_max_acc_disp_log', 'SV_max_acc_disp_diff']
# ['SV_acc', 'SV_tpr', 'SV_tnr', 'SV_max_acc_disp_diff',
#  'SV_max_acc_disp_log', 'SV_tp_diff', 'SV_fpr', 'SV_fnr', 'SV_EOp', 'SV_EOp_bounded']

TRAIN_DATASET = 'LFWA' #'LFWA' or 'FairFaces' or 'celebA'

EPOCHS = 100
BATCH_SIZE = 128

INPUT_SHAPE = (160,160)
lr = 0.0005

AUTOTUNE = tf.data.experimental.AUTOTUNE
######################################################
######################################################

for reweigthing_metric in reweigthing_metrics_names:
    tf.keras.backend.clear_session()

    if reweigthing_metric is None:
        NEW_MODEL_NAME = TRAIN_DATASET+'_irv1_finetunning'
    else: 
        NEW_MODEL_NAME = "reweighted_"+reweigthing_metric.lower()+"_truncated_lfwa"
    
    NEW_RES_PATH = 'results'+os.path.sep+NEW_MODEL_NAME+'_T'+time.strftime('%d_%m_%y__%H_%M')
    if not os.path.exists(NEW_RES_PATH) and not PRUEBAS:
        os.makedirs(NEW_RES_PATH)
    
    ### Load pretrained model
    pretrained_inception = tf.keras.models.load_model(pretrained_model_path) # custom_objects={'InceptionResNetV1': InceptionResNetV1})
    # Freeze all the layers but the last one - FREEZE LAYERS
    for layer in pretrained_inception.layers[:LAYER_POS]:
        layer.trainable = False

    # LOAD DATASET AND SV IF NEEDED
    CROP = None
    main_data_folder = "data"+os.sep+TRAIN_DATASET+os.sep
    if TRAIN_DATASET=='FairFaces':
        images_folder = main_data_folder
        pd_loader = FairFaces(main_data_folder)

    elif TRAIN_DATASET=='celebA':
        images_folder = main_data_folder+os.sep+'img_align_celeba'+os.sep+'img_align_celeba'+os.sep
        pd_loader = CelebA(main_data_folder, images_folder)
        CROP = 'CelebA'

    elif TRAIN_DATASET=='LFWA':
        main_data_folder = "data"+os.sep+"LFWa"+os.sep
        images_folder = 'lfw_funneled'
        pd_loader = LFWA(main_data_folder, images_folder)
        CROP = 'LFWA'
    _ = pd_loader.load_dataset()

    print(f'##Get {TRAIN_DATASET} Data Loader')
    df_train = pd_loader.df_train
    df_val = pd_loader.df_val

    if not reweigthing_metric is None:
        df_SV = pd.read_pickle(data_SV_path)
        if TRUNCATED:
            df_SV.loc[df_SV[reweigthing_metric]<0,reweigthing_metric]=0
        #if reweigthing_metric=='SV_max_acc_disp_log':
        #    OPPOSITE = True
        assert df_train.shape[0] == df_SV.shape[0]
        df_train = pd.concat([df_train, df_SV[reweigthing_metric]], axis=1) #add SV column to DF
        df_train[reweigthing_metric] = df_train[reweigthing_metric].astype('float64')
   

    print("### Generating data loaders")
    train_ds = get_data_loader(df_train, label='Male', weight=reweigthing_metric,
                                partition="train",
                                input_shape=INPUT_SHAPE,
                                batch_size=BATCH_SIZE,
                                augment=True,
                                crop=CROP)

    print("#### Generated train data loader")
    validation_ds = get_data_loader(df_val, label='Male',
                                    partition="val",
                                    input_shape=INPUT_SHAPE,
                                    batch_size=BATCH_SIZE,
                                    augment=False,
                                    crop=CROP)


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
                metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                            tf.keras.metrics.Accuracy(name='accuracy'),
                            tf.keras.metrics.FalsePositives(name='false_positives'),
                            tf.keras.metrics.FalseNegatives(name='false_negatives'),
                            tf.keras.metrics.TruePositives(name='true_positives'),
                            tf.keras.metrics.TrueNegatives(name='true_negatives')])

    print("### Checking trainable layers")
    for layer in pretrained_inception.layers:
        if layer.trainable == True:
            print(layer.trainable,
            " - Layer name:", layer.name, 
            " - # Params:", layer.output_shape)

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