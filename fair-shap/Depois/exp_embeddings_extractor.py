import tensorflow as tf
import numpy as np
import pandas as pd
import os
import in_rv1
from utils.tf_image_data_loader import get_data_loader
from utils.pd_data_loaders import FairFaces, LFWA

######## HW CONFIGURATION ########
#Random seed
np.random.seed(1)
tf.random.set_seed(2)
# GPU configuration
device='GPU'
gpu_n=1
physical_devices = tf.config.list_physical_devices(device)
print(physical_devices)
if device == "GPU" and "GPU" in physical_devices[gpu_n]:
    tf.config.set_visible_devices(physical_devices[gpu_n], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_n],True)
#################################
############PARAMETERS###########

RESPATH = 'results'+os.sep+'embeddings_lfwa'+os.sep
if not os.path.exists(RESPATH):
    os.makedirs(RESPATH)

LAYER_POS = -8
BATCH_SIZE = 128
INPUT_SHAPE = (160,160)

#model
results = "results/LFWA_irv1_finetunning_T29_03_22__11_33/"
model_saved_path = results+'LFWA_irv1_finetunning.h5'

#################################
#################################

model = tf.keras.models.load_model(model_saved_path) # custom_objects={'InceptionResNetV1': InceptionResNetV1})
# Create embedding model
emb_layer = model.layers[LAYER_POS]
print(f"""\n##Getting embeddings from layer {emb_layer.name} and info:\n###{emb_layer.output}""")
emb_model = tf.keras.Model(inputs = model.input,
                           outputs = emb_layer.output)


print('##Get LFWA Data Loader')
lfwa_data_folder = "data"+os.sep+"LFWa"+os.sep
lfwa_images_folder = 'lfw_funneled'
lfwa_pd_loader = LFWA(lfwa_data_folder, lfwa_images_folder)
df_lfwa = lfwa_pd_loader.load_dataset()
#df_lfwa = df_lfwa[(df_lfwa['partition']==0) | (df_lfwa['partition']==1)] # We get train and val shapley values
lfwa_tf_data_loader = get_data_loader(df_lfwa, label='Male',
                                      partition="test", # for neither shuffle nor repeat nor augment
                                      input_shape=INPUT_SHAPE,
                                      batch_size=BATCH_SIZE,
                                      augment=False,
                                      crop='LFWA')
print('##Get LFWA Embeddings')
lfwa_embeddings = emb_model.predict(lfwa_tf_data_loader)
df_lfwa_embeddings = pd.DataFrame(lfwa_embeddings, index= df_lfwa.index).add_prefix('e_dim_')
df_lfwa_embeddings = pd.concat([df_lfwa, df_lfwa_embeddings],axis=1) 
df_lfwa_embeddings.to_pickle(RESPATH + "lfwa_embeddings.pkl") 
print(lfwa_embeddings.shape)
print(df_lfwa_embeddings.shape)


print('##Get FairFaces Data Loader')
ff_data_folder = "data"+os.sep+"FairFaces"+os.sep
ff_pd_loader = FairFaces(ff_data_folder)
#df_ff = ff_pd_loader.df_train_val # We get the SV against train and val
df_ff = ff_pd_loader.load_dataset()
ff_tf_data_loader = get_data_loader(df_ff, label='Male',
                                      partition="test", # for neither shuffle nor repeat nor augment
                                      input_shape=INPUT_SHAPE,
                                      batch_size=BATCH_SIZE,
                                      augment=False)
print('##Get FF Embeddings')
ff_embeddings = emb_model.predict(ff_tf_data_loader)
df_ff_embeddings = pd.DataFrame(ff_embeddings, index= df_ff.index).add_prefix('e_dim_')
df_ff_embeddings = pd.concat([df_ff, df_ff_embeddings],axis=1) 
df_ff_embeddings.to_pickle(RESPATH + "ff_embeddings.pkl") 
print(ff_embeddings.shape)
print(df_ff_embeddings.shape)





