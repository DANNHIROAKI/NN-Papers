# **`Pr-VIPE`**: **V**iew\-**I**nvariant **Pr**obabilistic **E**mbedding for Human Pose

We propose an approach for learning a compact view-invariant probabilistic
embedding space for 3D human poses from their 2D projections. Please refer to our [ECCV'20 paper](https://arxiv.org/abs/1912.01001), [IJCV'21 paper](https://arxiv.org/abs/2010.13321), and [website](https://sites.google.com/corp/view/pr-vipe/home) for more details.

<p align="center">
  <img src="../doc/fig/manifold.png" width=800></br>
</p>

## Contact
- [Ting Liu](https://github.com/tingliu)
- [Jennifer J. Sun](https://github.com/jenjsun)
- [Liangzhe Yuan](https://github.com/yuanliangzhe)

## Updates
- `12/03/2021`: Added [**`Temporal Pr-VIPE`**](https://github.com/google-research/google-research/tree/master/poem/pr_vipe/temporal) training support. Please refer to our [IJCV'21 paper](https://arxiv.org/abs/2010.13321) for details.
- `11/23/2021`: Added individual random keypoint dropout support. Please refer to our [IJCV'21 paper](https://arxiv.org/abs/2010.13321) for details.
- `03/25/2021`: Moved the Pr-VIPE project code into the [`pr_vipe`](https://github.com/google-research/google-research/tree/master/poem/pr_vipe) folder.
- `03/17/2021`: Fixed an [issue](https://github.com/google-research/google-research/issues/636) in camera augmentation.
- `03/04/2021`: Added a program for running model inference.
- `10/21/2020`: Added cross-view pose retrieval evaluation frame keys.
- `10/15/2020`: Added training TFRecords generation program.
- `07/02/2020`: First release. Included core TensorFlow code for model training.
