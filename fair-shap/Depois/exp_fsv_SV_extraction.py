from fairSV.fair_shapley import FairShapley
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import json

RES_PATH = 'results/SV/lfwa_sv/' 
if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)

## Load embeddings or data
df_lfwa_emb = pd.read_pickle('results/embeddings/lfwa_embeddings.pkl')
df_ff_emb = pd.read_pickle('results/embeddings/ff_embeddings.pkl')
df_lfwa_emb_train = df_lfwa_emb[df_lfwa_emb.partition==0].copy()
df_ff_emb_val = df_ff_emb[df_ff_emb.partition==1].copy()

X_train = df_lfwa_emb_train.filter(regex='^e_dim').values
y_train = df_lfwa_emb_train.Male.values
print('Training samples', X_train.shape)

X_test = df_ff_emb_val.filter(regex='^e_dim').values
y_test = df_ff_emb_val.Male.values
print('Testing samples',X_test.shape)

sc = StandardScaler()
sc = sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


## Fair SV
fair_sv_extractor = FairShapley(X_train, y_train, X_test, y_test, show_plot=False, calculate_2dim=False)
k,prob_acc,probs=fair_sv_extractor.get_best_K(max_k=30)
metrics,_= fair_sv_extractor.do_knn(k=k)

SV_m = fair_sv_extractor.get_SV_matrix(K=k)

df_lfwa_emb_train['SV_acc'] = fair_sv_extractor.sv_acc
df_lfwa_emb_train['SV_tpr'] = fair_sv_extractor.sv_tpr
df_lfwa_emb_train['SV_tnr'] = fair_sv_extractor.sv_tnr
df_lfwa_emb_train['SV_max_acc_disp_diff'] = fair_sv_extractor.sv_max_acc_disp_diff
df_lfwa_emb_train['SV_max_acc_disp_log'] = fair_sv_extractor.sv_max_acc_disp_log
df_lfwa_emb_train['SV_tp_diff'] = fair_sv_extractor.sv_tp_diff
df_lfwa_emb_train['SV_EOp'] = fair_sv_extractor.sv_eop
df_lfwa_emb_train['SV_EOp_bounded'] = fair_sv_extractor.sv_eop_bounded

columns_to_get = ['Male', 'SV_acc', 'SV_tpr', 'SV_tnr',
                'SV_EOp', 'SV_EOp_bounded',    
                'SV_max_acc_disp_diff', 'SV_max_acc_disp_log', 'SV_tp_diff']
df_lfwa_emb_train[columns_to_get].to_pickle(RES_PATH + "lfwa_SV.pkl") 
pd.DataFrame(SV_m, index = df_lfwa_emb_train.index, columns=df_ff_emb_val.index).to_pickle(RES_PATH + "lfwa_SV_matrix.pkl") 

print(json.dumps(metrics, indent=4))