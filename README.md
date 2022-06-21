![code style](https://img.shields.io/badge/code%20style-black-black)
# Predicting the "odd-one-out" on a triplet task

## Cifar-100 triplet task
Generate triplets by combining two samples of Cifar100 with the same coarse label (20 labels) and one "odd-one-out" 
with a different coarse label. This results for example in the following triplet:
![](images/cifar_triplet_0.png)

with bike and train being in the same coarse category and tree being the odd one out.


## Research Questions/Ideas
* How does the pre-training task affect accuracy? (supervised vs. self-supervised learning)
* How does model size/embeddings size affect accuracy?
* Does scaling really help to peform close to human-level intelligence?
* How does model architecture affect accuracy? (e. g. VGGs vs. ResNets vs. ViTs)
* Linear probing on top of representations/finetuning
* Can we correlate accuracy with CKA similarity of representations?
* How uniformly do models make mistakes vs. humans?
* Fine-tune each model on THINGS (multi-class classification for THINGS) and then compare models against humans?
* Predict the similarity space of a different model (use predictions of one model as label).
* Do self-supervised learning on THINGS dataset and then measure the performance.

## Usage

Run evaluation on things triplet task with Imagenet pretrained Resnet18 and Resnet50.

```python
python main_eval.py --data_root /home/space/datasets/things \
--model_names vgg16 alexnet resnet18 resnet50 clip-ViT \
--module_names classifier.3 classifier.4 avgpool avgpool \
--batch_size 128 \
--out_path /path/to/results \
--device cuda \
--temperature 1.0 \
--rnd_seed 42 \
--verbose
```

## Results
| Model                 | Things (Penultimate) | Things (Logits; Dot) | Cifar-100-0 | #parameters | Imagenet Accuracy |
|-----------------------|----------------------|----------------------|-------------|-------------|-------------------|
| Efficientnet B0       |                      | 45.35                |             |             | 77.692            |
| Efficientnet B1       |                      | 43.01                |             |             | 78.642            |
| Efficientnet B2       |                      | 43.39                |             |             | 80.608            |
| Efficientnet B3       |                      | 38.90                |             |             | 82.008            |
| Efficientnet B4       |                      | 43.84                |             |             | 83.384            |
| Efficientnet B5       |                      | 44.80                |             |             | 83.444            |
| Efficientnet B6       |                      | 45.57                |             |             | 84.008            |
| Efficientnet B7       |                      | 45.53                |             |             | 84.122            |
| Resnet 18             | 47.33                |                      | 62.09       |             | 69.758            |
| Resnet 34             | 47.44                |                      |             |             | 73.314            |
| Resnet 50             | 47.74                |                      | 64.75       |             | 76.130            |
| Resnet 50 BarlowTwins | 43.84                |                      | 59.72       |             | 73.5              |
| Resnet 101            | 47.56                |                      |             |             | 77.374            |
| Resnet 152            | 47.24                |                      | 68.32       |             | 78.312            |
| Vit-B 16              | 50.89                |                      |             |             | (81.072)          |
| VGG 11                |                      | 51.96                |             |             | 69.020            |
| VGG 13                |                      | 52.19                |             |             | 69.928            |
| VGG 16                |                      | 52.06                |             |             | 71.592            |
| VGG 19                |                      | 51.83                | 66.49       |             | 72.376            |
| AlexNet               |                      |                      |             |             |                   |
