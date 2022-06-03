
# Predicting the "odd-one-out" on a triplet task

## Cifar-100 triplet task
Generate triplets by combining two samples of Cifar100 with the same coarse label (20 labels) and one "odd-one-out" 
with a different coarse label. This results for example in the following triplet:
![](images/cifar_triplet_0.png)

with bike and train being in the same coarse category and tree being the odd one out.


## Research Questions/Ideas
* How does the pretraining task affect accuracy?
* How does model size/embeddings size affect accuracy?
* How does model architecture affect accuracy? (e. g. Resnets vs VITs)
* Which models do the same mistakes?
* Linear probing on top of representations/finetuning
* Can we correlate accuracy with CKA similarity of representations?
* How uniformly does the model makes mistakes vs humans?
* Fine-tune each model on THINGS (multi-class classification) and then compare against humans?

## Results
| Model                 | Things | Cifar-100-0 | #parameters | Imagenet Accuracy |
|-----------------------|--------|-------------|-------------|-------------------|
| Resnet 18             | 47.33  | 62.09       |             | 69.758            |
| Resnet 50             | 47.74  | 64.75       |             | 76.130            |
| Resnet 50 BarlowTwins | 43.84  | 59.72       |             | 73.5              |
| Resnet 152            |        | 68.32       |             | 78.312            |
| Vit-B 16              | 50.89  | 75.8        |             | (81.072)          |
