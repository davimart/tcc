# Towards Algorithmic Fairness by means of Data Re-weighting based on Shapley Values

### File Structure

Results starting with `exp_` are the files used for the experiment for image classification using CNNs. Instead, the ones starting with `fsv_` are the files used for the tabular datasets.

```
Fair-Shap
┣ 📦 fairSV                   # Main code for Fair Shapley Value computation
┃  ┣📄 fair_shapley.py        # Class-based implementation of with FairShapley
┃  ┣📄 fair_shalpey_sklearn   # Fast (numba) and script based implementation of FairShap
┃  ┗📄 fairness_metrics.py    # Wrapper for fairness metrics from aif360
┃
┣ 📦 utils                    # Wrappers for fairness metrics, aif360, data loaders, and baseliens
┣ 📦 data                     # Data for image classification experiments (here only csv of attributes given the space limitation)
┃
┣ 📄in_rv1.py                 # Inception Resnet v-1
┃
┣ 📄exp_celeba_pretaining.py            # Get results from model trained with CelebA
┣ 📄exp_fairfaces_training.py           # Get results from model trained with FairFaces
┣ 📄exp_lfwa_finetunning.py             # Get results from pretrain with celebA and LFWA finetunning
┣ 📄exp_lfwa_finetunning_rw_many.py     # Get results from pretrain with celebA and LFWA finetunning reweighted with a list of different SV
┣ 📄exp_lfwa_finetunning_one_to_one.py  # Get result from pretrain with celebA and LFWA finetunning reweighted with one SV
┣ 📄exp_embeddings_extractor.py         # Example of how to get embeddings from a given model
┣ 📄exp_fsv_SV_extraction.py            # Example of how to get FairShap values from a latent space
┣ 📄exp_test_many_to_many.py            # Get test results from many models against many datasets
┣ 📄exp_test_one_to_one.py              # Get test results from 1 model against 1 dataset
┃
┣ 📄fsv_EXPERIMENTS_sklearn.py          # Script for experiments with fairness benchmarks
┣ 📊fsv_exp_Synth.ipynb                 # Teaching example with synthetic datasets
┣ 📊fsv_IFRWGurobiLicTest.ipynb         # Test code of Influence Functions baseline which needs a Gurobi License.
┣ 📄fsv_time_comparison.py              # Time comparison experiment
┃
┣ 📄sh_test.sh                          # Script that launch the previous one with parameters
┃
┣ 📊*.ipynb                             # Notebook analyzing FairShap in different scenarios
┗
```



 

