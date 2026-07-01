"""
Model test with with validation split of FairFaces
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import in_rv1

from utils.tf_image_data_loader import get_data_loader
from utils.pd_data_loaders import CelebA, LFWA, FairFaces
from fairSV.fairness_metrics import get_roc_curve, plot_same_metric_diff_df, plot_metrics, get_rates

np.random.seed(1)
tf.random.set_seed(2)

device='GPU'
gpu_n=0

physical_devices = tf.config.list_physical_devices(device)
print(physical_devices)
if device == "GPU" and "GPU" in physical_devices[gpu_n]:
    tf.config.set_visible_devices(physical_devices[gpu_n], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_n],True)

######################
######################
TEST_DATASET = 'FairFaces' # ['LFWA', 'FairFaces', 'celebA']

# Model to test with the previous test set and out path name
# CelebA Pretraining
"""results = "results/reweighted_sv_eop_truncated_lfwa_T24_03_22__18_00/"
train_history_csv = results+"reweighted_sv_eop_truncated_lfwa.csv"
model_saved_path = results+'reweighted_sv_eop_truncated_lfwa.h5'
out_path = results+'metrics'+os.sep"""


# LFWA Raw finetunning
"""results = "results/lfwa_inception_resnet_v1_finetunning_T15_02_22__17_12/"
train_history_csv = results+"lfwa_inception_resnet_v1_finetunning.csv"
model_saved_path = results+'FINAL_PRE_lfwa_inception_resnet_v1_finetunning.h5'
out_path = results+'metrics_Final'+os.sep"""


# FairFaces Raw training
"""results = "results/fairfaces_inception_rv2_T16_02_22__17_32/" # do not forget last /
train_history_csv = results+"fairfaces_inception_rv2.csv"
model_saved_path = results+'fairfaces_inception_rv2.h5'
out_path = results+'metrics'+os.sep"""

# SV RW
results = "results/reweighted_sv_eop_truncated_lfwa_T24_03_22__18_00/"
train_history_csv = results+"reweighted_sv_eop_truncated_lfwa.csv"
model_saved_path = results+'reweighted_sv_eop_truncated_lfwa.h5'
out_path = results+'metrics'+os.sep


BATCH_SIZE = 128
INPUT_SHAPE = (160,160)
######################
######################

CROP = None
if TEST_DATASET=='FairFaces':
    main_data_folder = "data"+os.sep+"FairFaces"+os.sep
    images_folder = main_data_folder
    pd_loader = FairFaces(main_data_folder)
elif TEST_DATASET=='CelebA':
    main_data_folder = "data"+os.sep+"celebA"+os.sep
    images_folder = main_data_folder+os.sep+'img_align_celeba'+os.sep+'img_align_celeba'+os.sep
    pd_loader = CelebA(main_data_folder, images_folder)
    #CROP = 'CelebA'
elif TEST_DATASET=='LFWA':
    main_data_folder = "data"+os.sep+"LFWa"+os.sep
    images_folder = 'lfw_funneled'
    pd_loader = LFWA(main_data_folder, images_folder)
    #CROP = 'LFWA'

if not os.path.exists(out_path):
    os.makedirs(out_path)

## SAVE PLOTS: Save diagrams of loss and accuracy
df_history = pd.read_csv(train_history_csv, delimiter=";").set_index("epoch")
plot_metrics(df_history, ['loss', 'val_loss'],
                path=out_path, save=True)
plot_metrics(df_history, ['binary_accuracy', 'val_binary_accuracy'],
                path=out_path, save=True)

# Save diagrams of TPR and TNR
train_metrics_history = get_rates(df_history, all=False)
val_metrics_history = get_rates(df_history, val=True, all=False)
plot_same_metric_diff_df([train_metrics_history, val_metrics_history],
                            metrics=['TPR'], name=['tr_','val_'],
                            path=out_path, save=True)
plot_same_metric_diff_df([train_metrics_history, val_metrics_history],
                            metrics=['TNR'], name=['tr_','val_'],
                            path=out_path, save=True)

## TAKE METRICS FROM HISOTRY OF TRAINING: Construct table with metrics for Train, Validation and Test
best_epoch_id = df_history['val_loss'].idxmin()
best_epoch = pd.DataFrame(df_history.iloc[best_epoch_id]).T # 1 row x N measures
train_metrics_df = pd.DataFrame(get_rates(best_epoch, all=True))
val_metrics_df = pd.DataFrame(get_rates(best_epoch, val=True, all=True))

## TEST METRICS
model = tf.keras.models.load_model(model_saved_path) # custom_objects={'InceptionResNetV1': InceptionResNetV1})

print("#### Test metrics")
df_test_data = pd_loader.df_test

print("#### Generated test data loader")
fair_test_ds = get_data_loader(df_test_data, label='Male',
                                partition="test",
                                input_shape=INPUT_SHAPE,
                                batch_size=BATCH_SIZE,
                                augment=False,
                                crop=CROP)

#List with metrics in training
res = model.evaluate(fair_test_ds, verbose=1)
probs = model.predict(fair_test_ds, verbose=1)
df_test_data['prediction'] = probs
df_test_data.to_csv(out_path+"attrs_and_predictions.csv", float_format='%.4f')

test_metrics_df = pd.DataFrame(dict(zip(model.metrics_names, res)), index = ['test'])
test_metrics_df = pd.DataFrame(get_rates(test_metrics_df, all=True))


print("###Calculating metric for optimized thresholds")
best_threshold, opt_metrics = get_roc_curve(df_test_data['Male'].values, df_test_data['prediction'].values, path=out_path, plot_fig=True)

tp = len(df_test_data[(df_test_data['prediction']>=best_threshold) & (df_test_data['Male']==1)])
fn = len(df_test_data[(df_test_data['prediction']<best_threshold) & (df_test_data['Male']==1)])

tn = len(df_test_data[(df_test_data['prediction']<best_threshold) & (df_test_data['Male']==0)])
fp = len(df_test_data[(df_test_data['prediction']>=best_threshold) & (df_test_data['Male']==0)])

total_n = len(df_test_data)
print(tp+tn+fp+fn, '==', total_n)
assert tp+tn+fp+fn == total_n
opt_metrics['binary_accuracy'] = (tp+tn)/(tp+tn+fp+fn)
opt_metrics['true_positives'] = tp
opt_metrics['true_negatives'] = tn
opt_metrics['false_positives'] = fp
opt_metrics['false_negatives'] = fn
opt_metrics_df = pd.DataFrame(opt_metrics, index=['opt'])

final_df_metric = pd.concat([train_metrics_df,val_metrics_df,test_metrics_df,opt_metrics_df],
                            keys=['Train', 'Validation', 'Test', best_threshold]).reset_index(level=1, drop=True)
final_df_metric.T.to_csv(out_path+"test_metrics.csv", index_label="Test_metrics", float_format='%.4f')
print(model_saved_path)
print(final_df_metric.T)
print(f"""\n\n### Results in {out_path}""")



