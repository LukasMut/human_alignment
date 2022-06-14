
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
=======
* Which models make the same mistakes?
* Linear probing on top of representations/finetuning
* Can we correlate accuracy with CKA similarity of representations?
* How uniformly do models make mistakes vs. humans?
* Fine-tune each model on THINGS (multi-class classification) and then compare models against humans?
* Predict the similarity space of a different model (use predictions of one model as label).
* Do self-supervised learning on then THINGS dataset and then measure the performance.
>>>>>>> deb5af878fb99ac68f3cb4414c37fa7dbe588762

## Results
| Model                 | Things | Cifar-100-0 | #parameters | Imagenet Accuracy |
|-----------------------|--------|-------------|-------------|-------------------|
| Resnet 18             | 47.33  | 62.09       |             | 69.758            |
| Resnet 50             | 47.74  | 64.75       |             | 76.130            |
| Resnet 50 BarlowTwins | 43.84  | 59.72       |             | 73.5              |
| Resnet 152            |        | 68.32       |             | 78.312            |
| Vit-B 16              | 50.89  | 75.8        |             | (81.072)          |
