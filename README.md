
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 2fa34bb05e6fd22bcc520528cc9a3aab015eb73d
* Which models make the same mistakes?
* Linear probing on top of representations/finetuning
* Can we correlate accuracy with CKA similarity of representations?
* How uniformly do models make mistakes vs. humans?
* Fine-tune each model on THINGS (multi-class classification) and then compare models against humans?
* Predict the similarity space of a different model (use predictions of one model as label).
* Do self-supervised learning on then THINGS dataset and then measure the performance.
=======
* Which models do the same mistakes?
=======
* Which models make the same mistakes?
>>>>>>> Update README.md
* Linear probing on top of representations/finetuning
* Can we correlate accuracy with CKA similarity of representations?
* How uniformly do models make mistakes vs. humans?
* Fine-tune each model on THINGS (multi-class classification for THINGS) and then compare models against humans?
* Predict the similarity space of a different model (use predictions of one model as label).
<<<<<<< HEAD
* Do self-supervised learning on THINGS dataset and then measure the performance.
>>>>>>> Update README.md
=======
* Do self-supervised learning on then THINGS dataset and then measure the performance.
>>>>>>> Update README.md
<<<<<<< HEAD
=======
=======
* Which models make the same mistakes?
* Linear probing on top of representations/finetuning
* Can we correlate accuracy with CKA similarity of representations?
* How uniformly do models make mistakes vs. humans?
* Fine-tune each model on THINGS (multi-class classification) and then compare models against humans?
* Predict the similarity space of a different model (use predictions of one model as label).
* Do self-supervised learning on then THINGS dataset and then measure the performance.
>>>>>>> deb5af878fb99ac68f3cb4414c37fa7dbe588762
>>>>>>> 2fa34bb05e6fd22bcc520528cc9a3aab015eb73d

## Usage

Run evaluation on things triplet task with Imagenet pretrained Resnet18 and Resnet50.

```
python main_triplet_eval.py --models resnet18 resnet50 \
--dataset things \
--data_root /home/space/datasets/things \
--out-file results.csv
```


## Results
| Model                 | Things | Cifar-100-0 | #parameters | Imagenet Accuracy |
|-----------------------|--------|-------------|-------------|-------------------|
| Efficientnet B0       | 45.35  |             |             |                   |
| Efficientnet B1       | 43.01  |             |             |                   |
| Efficientnet B2       | 43.39  |             |             |                   |
| Efficientnet B3       | 38.90  |             |             |                   |
| Efficientnet B4       | 43.84  |             |             |                   |
| Efficientnet B5       | 44.80  |             |             |                   |
| Efficientnet B6       | 45.57  |             |             |                   |
| Efficientnet B7       | 45.53  |             |             |                   |
| Resnet 18             | 47.33  | 62.09       |             | 69.758            |
| Resnet 34             | 47.44  |             |             |                   |
| Resnet 50             | 47.74  | 64.75       |             | 76.130            |
| Resnet 50 BarlowTwins | 43.84  | 59.72       |             | 73.5              |
| Resnet 101            | 47.56  |             |             |                   |
| Resnet 152            | 47.24  | 68.32       |             | 78.312            |
| Vit-B 16              | 50.89  |             |             | (81.072)          |
| VGG 11                | 51.96  |             |             |                   |
| VGG 13                | 52.19  |             |             |                   |
| VGG 16                | 52.06  |             |             |                   |
| VGG 19                | 51.83  | 66.49       |             |                   |
