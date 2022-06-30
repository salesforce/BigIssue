## BigIssue Dataset
 
Welcome to the BigIssue dataset GitHub repository!

The relevant data can be browsed [here](https://console.cloud.google.com/storage/browser/bigissue-research) once authenticated with Google Cloud, and downloaded from [https://storage.googleapis.com/bigissue-research](https://storage.googleapis.com/bigissue-research).

Training code is provided in the `/training` directory.

Checkpoints are located on the Google Cloud Storage bucket in [https://storage.googleapis.com/bigissue-research/checkpoints](https://storage.googleapis.com/bigissue-research/checkpoints). One can download them with the command

```
wget https://storage.googleapis.com/bigissue-research/checkpoints/pooling_real_data/model
```

An example of loading the model from the checkpoint is shown in `example_model_loading.py`.

