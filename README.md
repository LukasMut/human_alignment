<div align="center">
    <a href="https://github.com/LukasMut/human_alignment/blob/main" rel="nofollow">
        <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9-blue.svg" alt="PyPI" />
    </a>
    <a href="https://github.com/psf/black" rel="nofollow">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
</div>

# Predicting the "odd-one-out" on a triplet task

## Cifar-100 triplet task
Generate triplets by combining two samples of Cifar100 with the same coarse label (20 labels) and one "odd-one-out" 
with a different coarse label. This results for example in the following triplet:
![](images/cifar_triplet_0.png)

with bike and train being in the same coarse category and tree being the odd one out.


## Repository structure

```bash
root
├── envs
├── └── environment.yml
├── data
├── ├── __init__.py
├── ├── cifar.py
├── └── things.py
├── utils
├── ├── __init__.py
├── ├── analyses/*.py
├── ├── evaluation/*.py
├── └── probing/*.py
├── models
├── ├── __init__.py
├── ├── custom_mode.py
├── └── utils.py
├── .gitignore
├── README.md
├── main_embedding_eval.py
├── main_model_comparison.py
├── main_model_eval.py
├── main_probing.py
├── requirements.txt
├── search_temp_scaling.py
├── show_triplets.py
└── visualize_embeddings.py
```


## Usage

Run evaluation script on things triplet odd-one-out task with some pretrained model.

```python main_model_triplet_eval.py --data_root /path/to/data/name \ 
--dataset name \
--model_names resnet101 vgg11 clip_ViT-B/32 clip_RN50 vit_b_16 \
--module logits \
--overall_source thingsvision \
--sources torchvision torchvision custom custom torchvision  \
--model_dict_path /path/to/model_dict.json \
--batch_size 128 \
--distance cosine \
--out_path /path/to/results \
--device cpu \
--verbose \
--rnd_seed 42 \
```

Run evaluation script on multi-arrangement similarity judgements with some pretrained model.

```python main_model_sim_eval.py --data_root /path/to/data/name \ 
--dataset name \
--model_names resnet101 vgg11 clip_ViT-B/32 clip_RN50 vit_b_16 \
--module logits \
--overall_source thingsvision \
--sources torchvision torchvision custom custom torchvision  \
--model_dict_path /path/to/model_dict.json \
--batch_size 118 
--out_path /path/to/results \
--device cpu \
--verbose \
--rnd_seed 42 \
```
