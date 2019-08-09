
# Lookahead Optimizer

This repository contains implementations for [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610) in TensorFlow and PyTorch.

Lookahead improves the learning stability and lowers the variance of its inner optimizer with negligible computation and memory cost. It is simple to incorporate into an existing machine learning pipeline.

## Usage

In PyTorch:
```python
optimizer = # {any optimizer} e.g. torch.optim.Adam
if args.lookahead:
    optimizer = Lookahead(optimizer, la_steps=args.la_steps, la_alpha=args.la_alpha)
```

In TensorFlow:
```python
optimizer = # {any optimizer} e.g. tf.train.AdamOptimizer
if args.lookahead:
    optimizer = Lookahead(optimizer, la_steps=args.la_steps, la_alpha=args.la_alpha)
```

We found that evaluation performance is typically better using the slow weights.
This can be done in PyTorch with something like this in your eval loop:
```python
if args.lookahead:
    optimizer._backup_and_load_cache()
    val_loss = eval_func(model)
    optimizer._clear_and_load_backup()
```

## Experiments

Experiments in the paper were based off the following repositories.

CIFAR-10/100: [Cutout](https://github.com/uoguelph-mlrg/Cutout)

Penn Treebank: [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm)

ImageNet: [PyTorch examples](https://github.com/pytorch/examples/tree/master/imagenet)

Neural Machine Translation: [tensor2tensor](https://github.com/tensorflow/tensor2tensor)

## Additional

If you have questions or suggestions, please feel free to open an issue. Please cite as:

```
@article{zhang2019lookahead,
  title={Lookahead Optimizer: k steps forward, 1 step back},
  author={Zhang, Michael R and Lucas, James and Hinton, Geoffrey and Ba, Jimmy},
  journal={arXiv preprint arXiv:1907.08610},
  year={2019}
}
```

<img src="figs/accuracy_surface.png" width="500"> 

