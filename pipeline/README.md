# RESOURCES

# DATASET
link : https://www.cfpw.io / https:/https:///www.cfpw.io/cfp-dataset.zip


# REQUERIMENTS
```
conda create -n py37 python=3.7
pip install ipython imgaug torch torchvision bunch tqdm tf-nigthly future


gcloud compute scp --project thermal-diorama-242618 --zone us-west1-b --recurse <local file or directory> tensorflow-1-vm:~/

gcloud compute instances describe --project thermal-diorama-242618 --zone us-west1-b tensorflow-1-vm | grep googleusercontent.com | grep datalab
```
