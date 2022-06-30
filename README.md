## BigIssue Benchmark and Code
 
Welcome to the BigIssue dataset GitHub repository! This is the official code repository for the BigIssues dataset and paper (paper link available soon).

### Data
The relevant data can be browsed [here](https://console.cloud.google.com/storage/browser/bigissue-research) once authenticated with Google Cloud, and downloaded from [https://storage.googleapis.com/bigissue-research](https://storage.googleapis.com/bigissue-research). One is required to have a valid Google Cloud account in order to browse data. 

In order to download the full dataset, one must have the `gsutil` command installed and active. Instructions on installation can be found [here](https://cloud.google.com/sdk/docs/install). Then one can retrieve datasets like this:

```
gsutil -m cp -r gs://bigissue-research/<name of dataset>/ .
```

There are three versions of the dataset:
* [synthetic_small](https://console.cloud.google.com/storage/browser/bigissue-research/synthetic_small/) (2048 tokens, 128 lines) 
* [synthetic_large](https://console.cloud.google.com/storage/browser/bigissue-research/synthetic/) (8192 tokens, 512 lines)
* Realistic dataset (Non-tokenized)
  * [realistic/single_file](https://console.cloud.google.com/storage/browser/bigissue-research/realistic/single_file) (Issues with changes to one Java file)
  * [realistic/multi_file](https://console.cloud.google.com/storage/browser/bigissue-research/realistic/multi_file) (Issues with changes to multiple Java files)

#### Synthetic

Synthetic data are pieces of code with infilled samples as described in the paper. Each sample is a `TFRecord` that is a concatenation of the tokenized code snippet and the label vector. We use [RobertaTokenizer](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaTokenizer) for tokenization. Labels consist of a vector of length 128 (512), where each line is marked as either 1 (buggy), 0 (not buggy), -1 (padded).

The synthetic data is already split into train, validation, and test splits.

#### Realistic

Realistic data consists of issue information for all of the issues we've collected as described in the paper. We provide the `fixed.tar.gz` and `unfixed.tar.gz` states of the repository, as well as the `diff` containing the changed diff and `issue.jsonl` with information about the issue.

### Training Code & Checkpoints

Training code is provided in the `/training` directory. Code is written for Python 3.8 and higher. All pip requirements are provided in `requirements.txt`.

Checkpoints are located on the Google Cloud Storage bucket in [https://storage.googleapis.com/bigissue-research/checkpoints](https://storage.googleapis.com/bigissue-research/checkpoints). One can download them with the command

```
wget https://storage.googleapis.com/bigissue-research/checkpoints/pooling_real_data/model
```

### Example

An example of loading a model from the checkpoint is provided in `examples/example_model_loading.py`.

### Citation

We will put a citation link once the paper is published on Arxiv.

### Questions & Feedback

If you have any feedback, please either create an Issue here on GitHub or send an email to `pkassianik@salesforce.com`.