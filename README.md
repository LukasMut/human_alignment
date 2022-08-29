![code style](https://img.shields.io/badge/code%20style-black-black)


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
├── ├── cifar.py
├── └── things.py
├── utils
├── ├── analyses/*.py
├── ├── evaluation/*.py
├── └── probing/*.py
├── models
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

Run evaluation on things triplet task with Imagenet pretrained Resnet18 and Resnet50.

```python
python main_eval.py --data_root /home/space/datasets/things \
--model_names vgg16 alexnet resnet18 resnet50 clip-ViT \
--module logits
--batch_size 128 \
--out_path /path/to/results \
--device cuda \
--distance cosine \
--rnd_seed 42 \
--verbose
```
