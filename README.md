# modulation_classification_early_exiting
This repository contains the public source code release for the paper <b>"Using Early Exits for Fast Inference in Automatic Modulation Classification".</b> This project is implemented mainly using PyTorch.

This code is created and maintained by [Elsayed Mohammed](https://github.com/ElSayedMMostafa).

*Slight differences between the results one can get using our code and the ones shown in the paper are expected specially in the inference time due to the differences in OS and computing resources used. The same trends are guranteed though.*

## Abstract
Automatic modulation classification (AMC) plays a critical role in wireless communications by autonomously classifying signals transmitted over the radio spectrum. Deep learning (DL) techniques are increasingly being used for AMC due to their ability to extract complex wireless signal features. However, DL models are computationally intensive and incur high inference latencies. This paper proposes the application of early exiting (EE) techniques for DL models used for AMC to accelerate inference. We present and analyze four early exiting architectures and a customized multi-branch training algorithm for this problem. Through extensive experimentation, we show that signals with moderate to high signal-to-noise ratios (SNRs) are easier to classify, do not require deep architectures, and can therefore leverage the proposed EE architectures. Our experi- mental results demonstrate that EE techniques can significantly reduce the inference speed of deep neural networks without sacrificing classification accuracy. We also thoroughly study the trade-off between classification accuracy and inference time when using these architectures. To the best of our knowledge, this work represents the first attempt to apply early exiting methods to AMC, providing a foundation for future research in this area.


## Files Description
>  **experiments.ipynb**: the main notebook implementing the 4 architectures and the main one as well as running the experiments and showing the results.

> **trainer.py**: a utility file to customizly train the EE models.

> **getData.py**: a utility file to load and process the dataset.

> **getModel.py**: a utility file to create and design the models.

## Citation
You can find our paper on [arXiv](https://arxiv.org/abs/2308.11100). \
The paper is **accepted** at 2023 IEEE Global Communications Conference: Mobile and Wireless Networks (GlobeCom 2023).

If you use this code or find it useful for your research, please consider citing:
```
@article{mohammed2023using,
  title={Using Early Exits for Fast Inference in Automatic Modulation Classification},
  author={Mohammed, Elsayed and Mashaal, Omar and Abou-Zeid, Hatem},
  journal={arXiv preprint arXiv:2308.11100},
  year={2023}
}
```
