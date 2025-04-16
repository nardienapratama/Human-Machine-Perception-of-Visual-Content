import papermill as pm
import os

# Define your parameters
n_jobs = 24
embeddings_path = "/data/outputs_50/finetuning_all-MiniLM-L12-v2_embeddings.csv" # path to embeddings csv from notebook 1d
parameter_sets = [
    {'use_ml_obj': True, 'use_ml_capt': False, 'use_human_labels': False, 'n_jobs': n_jobs, 'embeddings_path': embeddings_path},
    {'use_ml_obj': False, 'use_ml_capt': True, 'use_human_labels': False, 'n_jobs': n_jobs, 'embeddings_path': embeddings_path},
    {'use_ml_obj': False, 'use_ml_capt': False, 'use_human_labels': True, 'n_jobs': n_jobs, 'embeddings_path': embeddings_path},
    {'use_ml_obj': True, 'use_ml_capt': True, 'use_human_labels': False, 'n_jobs': n_jobs, 'embeddings_path': embeddings_path},
    {'use_ml_obj': True, 'use_ml_capt': False, 'use_human_labels': True, 'n_jobs': n_jobs, 'embeddings_path': embeddings_path},
    {'use_ml_obj': False, 'use_ml_capt': True, 'use_human_labels': True, 'n_jobs': n_jobs, 'embeddings_path': embeddings_path},
    {'use_ml_obj': True, 'use_ml_capt': True, 'use_human_labels': True, 'n_jobs': n_jobs, 'embeddings_path': embeddings_path},
    # Add more parameter sets as needed
]


# Path to your notebook
notebook_path = '3b-income_regression_model.ipynb'
output_dir = 'model_outputs/reg-v3'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Run Papermill for each set of parameters
for i, params in enumerate(parameter_sets):
    parts = []
    # Append relevant terms based on the flags
    if params['use_ml_obj']:
        parts.append('ml_obj')
    if params['use_ml_capt']:
        parts.append('ml_capt')
    if params['use_human_labels']:
        parts.append('human_labels')
    
    # Join the parts with underscores
    annotations_used_underscore = '_'.join(parts)
    
    output_path = os.path.join(output_dir, f'reg_{annotations_used_underscore}.ipynb')
    print(f"Running notebook with parameters {params}")
    pm.execute_notebook(
        input_path=notebook_path,
        output_path=output_path,
        parameters=params
    )
