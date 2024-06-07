# Tools for linear evaluation using L-BFGS

This directory contains code for efficiently performing linear evaluation on a
matrix of fixed embeddings with accompanying labels in JAX, using L-BFGS.

If you find this code useful, please cite:

```
@misc{kornblith2021linear,
  title = {Tools for linear evaluation using L-BFGS},
  author = {Kornblith, Simon},
  howpublished = "\url{https://github.com/google-research/google-research/linear_eval}",
  year = {2021}
}
```

## Examples

### One-step tuning and evaluation

```python
results = linear_eval.tune_hparams_and_compute_test_accuracy(
    train_embeddings, train_labels,
    val_embeddings, val_labels,
    test_embeddings, test_labels)
```

`tune_hparams_and_compute_test_accuracy` is a convenience function that
automatically splits the training set for hyperparameter tuning (with the size
of the validation set determined by `val_fraction`), distributes it across
devices, finds the optimal hyperparameters, retrains the model on the full
combined train and val sets with these optimal hyperparameters, and computes the
accuracy on the provided test set. `results` is a Dict containing:

  - `test_accuracy`: Accuracy on test set.
  - `val_accuracy`: Accuracy of best model on val set.
  - `optimal_l2_reg`: Optimal value of L2 regularization.
  - `weights`: Linear classifier weights.
  - `biases`: Linear classifier biases.

### Training a classifier

```python
# Distribute embeddings across different accelerators/cores. If the number of
# embeddings is not evenly divisible by the number of cores, the embeddings are
# padded with zeros. `distributed_train_mask` indicates which embeddings are
# data (1) or padding (0).
((distributed_train_embeddings, distributed_train_labels),
 distributed_train_mask) = linear_eval.reshape_and_pad_data_for_devices(
     (train_embeddings, train_labels))

# Perform training. This function will not return until training is complete.
# See [the TFP documentation](https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize)
# for additional options that can be passed to control the optimizer.
weights, biases, fit_info = linear_eval.train(
    distributed_train_embeddings, distributed_train_labels,
    distributed_train_mask, l2_regularization=1e-6)
```

### Evaluating a classifier

```python
((distributed_test_embeddings, distributed_test_labels),
 distributed_test_mask) = linear_eval.reshape_and_pad_data_for_devices(
     (test_embeddings, test_labels))
accuracy = linear_eval.evaluate(
    distributed_test_embeddings, distributed_test_labels, distributed_test_mask,
    weights, biases)
```

### Tuning the L2 regularization hyperparameter

```python
optimal_l2_reg, weights, biases, accuracy = linear_eval.tune_l2_regularization(
    train_embeddings, train_labels, train_mask,
    val_embeddings, val_labels, val_mask,
    initial_range=10.0 ** np.arange(-6, 7, 2), num_steps=4)
```

This function will first train and evaluate classifiers at each point in
`initial_range`. It then searches to the left and right of the best point, at
L2 regularization values halfway between the best point and the point to either
side on a log scale. The halving process is repeated `num_steps` times.
Because the previous iteration's weights are used as a warm start, this process
is typically relatively efficient.
