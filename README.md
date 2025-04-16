# Human-Machine-Perception-of-Visual-Content

This repository contains scripts and data for preprocessing and analyzing visual content annotations. The workflow is designed to be executed in a specific order, with each script building upon the outputs of the previous one.

## Prerequisites

1. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```

2. Download NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('words')
   ```

## Step-by-Step Workflow
Follow these steps in order to process the data and generate results:

1. Annotate Models
Run `1a-ml_models_annotations.ipynb` to annotate models using the provided data.

2. Preprocess Data
Run `1b-data_preprocessing.ipynb` to clean and preprocess the data. This includes:

- Removing noise
- Lowercasing text
- Autocorrecting typos
- Segmenting words

3. Create Training Data
Run `1c-create_train_data_embedding_model.ipynb` to generate training data for the embedding model.

4. Convert to Embeddings
Run `1d-convert_to_embeddings-aws.ipynb` to convert the processed data into embeddings and visualise them using dimensionality reduction techniques such as t-SNE.

5. Compare Embeddings
Run `2-compare_embeddings.ipynb` to visualise and compare the generated embeddings.

6. Train Classification Model
Run `3a-region_classification_model.ipynb` to train and assess a classification model using the generated embeddings.

7. Evaluate the Model
Run `3b-income_classification_model.ipynb` to train and assess a regression model using the generated embeddings.

8. Perform Significance Tests for Classification Model 
Run `4-significance_test_clsf.ipynb` to create a perform significance tests for the best-overall and worst-overall classification models. Do the same for the regression models by running `4-significance_test_reg.ipynb`.

### Optional: Automate Classification and Regression Training
Instead of manually running the classification and regression training notebooks (3a-region_classification_model.ipynb and 3b-income_classification_model.ipynb), you can use the following scripts to automate the process for all annotation combinations:

1. Classification Training Automation:
Use run_clsg_scripts.py to loop through all annotation combinations and run the classification training notebook automatically.

2. Regression Training Automation:
Use run_ref_scripts.py to loop through all annotation combinations and run the regression training notebook automatically.

These scripts streamline the workflow and eliminate the need for manual execution of the notebooks.

## Data
The data/ directory contains CSV files and other resources used throughout the preprocessing and analysis steps.

## AWS Environment
These scripts are designed to run in an AWS environment. Ensure the following:

- AWS CLI is installed and configured.
- You have access to an S3 bucket for storing intermediate and final outputs.
- Proper IAM permissions are in place for uploading and downloading files from S3.

## Citation
If you use this repository in your research, please cite the following paper:

```
@misc{pratama2025perceptionvisualcontentdifferences,
      title={Perception of Visual Content: Differences Between Humans and Foundation Models}, 
      author={Nardiena A. Pratama, Shaoyang Fan, Gianluca Demartini},
      year={2025},
      eprint={2411.18968},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.18968}, 
}
```

## License
This project is licensed under the terms of the LICENSE file.

## Contact
For questions or feedback, please contact Nardiena A. Pratama (n.pratama@uq.edu.au).