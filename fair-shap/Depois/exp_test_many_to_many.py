"""
Test all modells agains Fair Faces
"""

from cmath import exp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import in_rv1

from utils.tf_image_data_loader import get_data_loader
from utils.pd_data_loaders import CelebA, LFWA, FairFaces
from fairSV.fairness_metrics import get_roc_curve, get_rates, plot_metrics, plot_same_metric_diff_df

np.random.seed(1)
tf.random.set_seed(2)

device='GPU'
gpu_n=0

physical_devices = tf.config.list_physical_devices(device)
print(physical_devices)
if device == "GPU" and "GPU" in physical_devices[gpu_n]:
    tf.config.set_visible_devices(physical_devices[gpu_n], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_n],True)

##################################################################
##################################################################

tests_datasets_names = ['FairFaces']
#All datasets
#tests_datasets_names = ['LFWA', 'FairFaces', 'celebA']

# Model to test with the previous test set and out path name
models =  ['results/reweighted_sv_acc_truncated_lfwa_T31_03_22__08_37/', 
            'results/reweighted_sv_eop_truncated_lfwa_T31_03_22__08_44/',
            'results/reweighted_sv_eop_bounded_truncated_lfwa_T31_03_22__08_50/']
## Use this if want to test All models
#not_models = ['SV', 'embeddings', 'embeddings_ff', '0_not_cropped']
#models = ['results'+os.sep+f+os.sep for f in os.listdir('results') if f not in not_models]

BATCH_SIZE = 128
INPUT_SHAPE = (160,160)

PLOTS = True
##################################################################
##################################################################

for m in models:
    if not os.path.exists(m):
        raise Exception(f"""Path {m} does NOT exist""")

for TEST_DATASET in tests_datasets_names:
    #Load test dataset
    CROP = None
    main_data_folder = "data"+os.sep+TEST_DATASET+os.sep
    if TEST_DATASET=='FairFaces':
        images_folder = main_data_folder
        pd_loader = FairFaces(main_data_folder)
        df_test_data = pd_loader.df_test

    elif TEST_DATASET=='celebA':
        images_folder = main_data_folder+os.sep+'img_align_celeba'+os.sep+'img_align_celeba'+os.sep
        pd_loader = CelebA(main_data_folder, images_folder)
        df_test_data = pd_loader.df_test
        CROP = 'CelebA'

    elif TEST_DATASET=='LFWA':
        main_data_folder = "data"+os.sep+"LFWa"+os.sep
        images_folder = 'lfw_funneled'
        pd_loader = LFWA(main_data_folder, images_folder)
        _ = pd_loader.load_dataset()
        df_test_data = pd_loader.df_val # VAL becasuse FF does not have test split
        CROP = 'LFWA'

    fair_test_ds = get_data_loader(df_test_data, label='Male',
                                    partition="test",
                                    input_shape=INPUT_SHAPE,
                                    batch_size=BATCH_SIZE,
                                    augment=False,
                                    crop=CROP)

    for training_path in models:
        model_versions = [training_path+f for f in os.listdir(training_path) if f.endswith('.h5')]
        train_history_csv = [training_path+f for f in os.listdir(training_path) if f.endswith('.csv')]
        assert len(train_history_csv) == 1
        train_history_csv = train_history_csv[0]

        for model_saved_path in model_versions: #Usually best and final
            is_final = '_final' if model_saved_path.split(os.sep)[-1].startswith('FINAL') else ''
            out_path = training_path+'metrics'+is_final+'_'+TEST_DATASET+os.sep
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            print(f"""\nTesting model {model_saved_path} in dataset {TEST_DATASET}...""")
            print(f"""\tResults in {out_path}...\n""")

            ############## Construct table with metrics for Train, Validation and Test ##############
            df_history = pd.read_csv(train_history_csv, delimiter=";").set_index("epoch")
            best_epoch_id = df_history['val_loss'].idxmin()
            best_epoch = pd.DataFrame(df_history.iloc[best_epoch_id]).T # 1 row x N measures
            train_metrics_df = pd.DataFrame(get_rates(best_epoch, all=True))
            val_metrics_df = pd.DataFrame(get_rates(best_epoch, val=True, all=True))

            if PLOTS:
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

            ############## TEST METRICS ##############
            model = tf.keras.models.load_model(model_saved_path)

            #Test
            res = model.evaluate(fair_test_ds, verbose=0)
            probs = model.predict(fair_test_ds, verbose=0)
            df_test_data.loc[:,'prediction'] = probs
            df_test_data.to_csv(out_path+"attrs_and_predictions.csv", float_format='%.4f')

            #Get measures with t=0.5
            test_metrics_df = pd.DataFrame(dict(zip(model.metrics_names, res)), index = ['test'])
            test_metrics_df = pd.DataFrame(get_rates(test_metrics_df, all=True))

            #Get measures th-independent and Calculating metric for optimized thresholds
            best_threshold, opt_metrics = get_roc_curve(df_test_data['Male'].values, df_test_data['prediction'].values, path=out_path, plot_fig=PLOTS)

            tp = len(df_test_data[(df_test_data['prediction']>=best_threshold) & (df_test_data['Male']==1)])
            fn = len(df_test_data[(df_test_data['prediction']<best_threshold) & (df_test_data['Male']==1)])

            tn = len(df_test_data[(df_test_data['prediction']<best_threshold) & (df_test_data['Male']==0)])
            fp = len(df_test_data[(df_test_data['prediction']>=best_threshold) & (df_test_data['Male']==0)])

            total_n = len(df_test_data)
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
            
            # print(final_df_metric.T)
            print(f"""\nTESTED model {model_saved_path} in dataset {TEST_DATASET}""")
            print(f"""\tResults in {out_path}\n""")



