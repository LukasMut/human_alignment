mkdir -p resources/datasets
wget https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/twoafc_val.tar.gz -O ./resources/datasets/twoafc_val.tar.gz

mkdir resources/datasets/2afc/val
tar -xzf ./resources/datasets/twoafc_val.tar.gz -C ./resources/datasets/2afc

# 2AFC Train set
mkdir resources/datasets/2afc/
wget https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/twoafc_train.tar.gz -O ./resources/datasets/twoafc_train.tar.gz

mkdir resources/datasets/2afc/train
tar -xzf ./resources/datasets/twoafc_train.tar.gz -C ./resources/datasets/2afc