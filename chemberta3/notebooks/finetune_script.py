import deepchem as dc
import numpy as np
from deepchem.models.torch_models import Chemberta
from deepchem.molnet import load_delaney
import boto3
import tempfile
import torch
import os

DATASET = os.environ['DATASET']
# DATASET = 'delaney'

load_functions = {
   'delaney': dc.molnet.load_delaney,
   'bace_regress': dc.molnet.load_bace_regression,
   'clearance': dc.molnet.load_clearance,
   'lipo': dc.molnet.load_lipo,
   'clintox': dc.molnet.load_clintox,
   'bace_class': dc.molnet.load_bace_classification,
   'bbbp': dc.molnet.load_bbbp,
   'hiv': dc.molnet.load_hiv,
   'sider': dc.molnet.load_sider,
   'tox21': dc.molnet.load_tox21
}

regression_dataset = ['delaney', 'bace_regress', 'clearance', 'lipo']
classification_dataset = ['clintox', 'bace_class', 'bbbp', 'hiv', 'sider', 'tox21']

if DATASET in regression_dataset:
   task = 'regression'
   use_max = False
   metric = dc.metrics.Metric(dc.metrics.rms_score)
   metric_out = 'rms_score'

elif DATASET in classification_dataset:
   task = 'classification'
   use_max = True
   metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
   metric_out = 'roc_auc_score'

s3 = boto3.client('s3')
# Define your S3 bucket and the file you want to download
bucket_name = 'dfs-chiron-datastore'
file_key = 'LLNL/models/chemberta_100m_mlm_epoch_4/checkpoint1.pt'
tune_dir = os.path.join(os.getcwd(), 'tuning_runs')
if not os.path.exists(tune_dir):
  os.mkdir(tune_dir)
temp_dir = tempfile.TemporaryDirectory()
log_dir= os.path.join(temp_dir.name, 'tuning_results_metadata')
os.mkdir(log_dir)
    # Define the path to save the file
temp_file_path = os.path.join(temp_dir.name, 'tmp.pt')


# Download the file from S3
s3.download_file(bucket_name, file_key, temp_file_path)
print(f"File downloaded to temporary directory: {temp_file_path}")

data = torch.load(temp_file_path)
data['model_state_dict'] = {key.replace("module.", ""): value for key, value in data['model_state_dict'].items()}

checkpoint_path = os.path.join(temp_dir.name, 'checkpoint1.pt')
with open(checkpoint_path, 'wb') as f:
  torch.save(data, f)

os.remove(temp_file_path)

loader = load_functions[DATASET](featurizer=dc.feat.DummyFeaturizer())
tasks, dataset, transformers = loader
train, val, test = dataset

print(tasks)
def load_and_restore(**params):
    ckpt_path = params['ckpt_path']
    del params['ckpt_path']
    model = Chemberta(n_tasks=len(tasks), task=task, **params)
    model.load_from_pretrained(ckpt_path)
    return model

# the parameters which are to be optimized
params = {
  'ckpt_path': [temp_dir.name],
  'learning_rate': [3e-05, 6e-04],
  'batch_size': [32, 64]
  }
# Creating optimizer and searching over hyperparameters
tuning_results = []
epochs = [10, 100, 500]
for i, epoch in enumerate(epochs):
  # set use_max to false when using regression datasets 
  optimizer = dc.hyper.GridHyperparamOpt(load_and_restore)
  result = optimizer.hyperparam_search(params, train, val, metric, transformers, nb_epoch=epoch, use_max=use_max, logfile=f'results_{i}.txt', logdir=log_dir)
  tuning_results.append(result)
  del optimizer
  torch.cuda.empty_cache()

print(tuning_results)

best_results = []
for i in range(len(epochs)):
    eval = tuning_results[i][0].evaluate(test, metrics=[metric], transformers=transformers)
    print(eval)
    best_results.append(eval['rms_score'])

results_arr = np.asarray(best_results)
best_score = min(results_arr)
index = np.argmin(results_arr)
best_params = tuning_results[index][1]
best_params['epoch'] = epochs[index]
best_model = tuning_results[index][0]
print(best_model.model_dir)
print(best_score)
print(best_params)

best_model_path = os.path.join(tune_dir, 'best_model_checkpoint')
best_model.save_checkpoint(model_dir=best_model_path)
