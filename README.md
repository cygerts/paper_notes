List of interesting papers, mostly in ML and CV area. 
Rather then collecting all state-of-the art papers/benchmarks here I look for new / orthogonal / surprising ideas.
Papers biased towards my interests: robustness, uncertainty estimation, understanding neural nets, computer vision.
Currently 3 main categories: intriguing properties of NN (robustness, nn training), general CV and ML, and other (e.g. position papers).


### Intriguing properties of NN:

#### 2019:
- The lottery ticket hypothesis: finding sparse, trainable neural networks, Frankle & Carbin, ICLR 2019, [[arXiv](https://arxiv.org/pdf/1803.03635.pdf)][[code](https://github.com/google-research/lottery-ticket-hypothesis)]. For simple classification problems after network traning we can remove around 90% of neurons and still get very similar accuracy, the neurons that were not removed won "lottery ticket" - their randomly initialized weights made the traning particularly effective.
- Weight Agnostic Neural Networks, Gaiger & Ha, NeurIPS 2019, [[arXiv](https://arxiv.org/pdf/1906.04358.pdf)] [[DEMO](https://weightagnostic.github.io/)] Architecture search for randomly!! initialized neural networks which can perform reasonably well simple reinforcement learning and classifications tasks.
- Exploring Randomly Wired Neural Networks for Image Recognition, Xie et. al, [[arXiv](https://arxiv.org/pdf/1904.01569.pdf)][[code](https://github.com/seungwonpark/RandWireNN)] Neural architecure search for image recognition tasks starting from randomly wired architectures. Some bizzare looking architecture got really good results.
- Benchmarking Neural Network Robustness to Common Corruptions and Perturbations, Hendrycks et. al, ICLR 2019, [[arXiv](https://arxiv.org/pdf/1903.12261.pdf)]. Introduces new benchmark for *common corruptions* (16 different types of noise, blur, adverse weather conditions) and show drastic reduction of accuracy. Also look at Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming by Michaelis et. al [[arXiv](https://arxiv.org/pdf/1907.07484.pdf)].
- Natural Adversarial Examples, D. Hendrycks et. al, [[arXiv](https://arxiv.org/pdf/1907.07174.pdf)] "Advesarial" images in the real world. Two another variants of ImageNet dataset: first is a set of natural images which current classifiers fail to classify, and the second for out of distribution detection.

#### 2018:
- ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness, Geirhos et. al, ICLR 2019, [[arXiv](https://arxiv.org/pdf/1811.12231.pdf)][[code](https://github.com/rgeirhos/texture-vs-shape)] They show that current classification algorithms are biased towards texture, using style transfer as data augmentation helps to partially mitigate the problem.
- Rethinking ImageNet Pre-Training, He et. al, ICCV 2019, [[arXiv](http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf)]. They show that ImageNet pretraining is not as crucial as we thought.
- Same-different problems strain convolutional neural networks, Ricci et. al,  [[arXiv](https://arxiv.org/pdf/1802.03390.pdf)]. They show one of the simplest examples of visual problems that current ml algorithms cannot solve.
- Deep Image Prior, Ulyanov et. al, CVPR 2018 [[arXiv](https://arxiv.org/pdf/1711.10925.pdf)][[code](https://github.com/DmitryUlyanov/deep-image-prior)]

#### 2015:
- Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images, Nguyen et. al, CVPR 2015, [[arXiv](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf)]

### CV / ML:

#### 2020:
- PointRend: Image Segmentation as Rendering, Kirillov et. al, [[arXiv](https://arxiv.org/pdf/1912.08193v2.pdf)]. New approach to image segmentation from randomly selected point-wise features, cool stuff.
- Self-training with Noisy Student improves ImageNet classification, Xie et. al, [[arXiv](https://arxiv.org/pdf/1911.04252.pdf)]

#### 2019:
- Learning by Cheating, Chen et. al, CoRL 2019, [[arXiv](https://arxiv.org/pdf/1912.12294.pdf)][[code](https://github.com/dianchen96/LearningByCheating)]
- RepPoints: Point Set Representation for Object Detection, Yang et. al, [[arXiv](https://arxiv.org/pdf/1904.11490.pdf)][[code](https://github.com/microsoft/RepPoints)]. Object detector supervised by target bounding boxes as all other detectors, additionally learns point-wise representation (without explicit supervision). 
- Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift, Ovadia et. al, NeurIPS 2019, [[arXiv ](https://arxiv.org/pdf/1906.02530.pdf)][[code](https://github.com/google-research/google-research/tree/master/uq_benchmark_2019)] Benchmarking various uncertainty estimation methods uner distributional shift.

#### 2018:
- Deep Reinforcement Learning that Matters, Henderson et. al, AAAI 2018, [[arXiv](https://arxiv.org/pdf/1709.06560.pdf)]. On reproduction of baselines in RL, they show that random seeds matters can greatly alter the results and many more, nice read.
- A Probabilistic U-Net for Segmentation of Ambiguous Images, Kohl etl al, DeepMind, NIPS 2019, [[arXiv](https://arxiv.org/pdf/1806.05034.pdf)] [[code](https://github.com/SimonKohl/probabilistic_unet)] Generative semantic segmentation network which allows to sample diverse and consistent segmentation variants. [[Follow-up](https://arxiv.org/pdf/1905.13077.pdf)] 
- The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, Zhang et. al, CVPR 2018, [[arXiv](https://arxiv.org/pdf/1801.03924.pdf)][[code](https://github.com/richzhang/PerceptualSimilarity)]
- Brute-Force Facial Landmark Analysis With InceA 140,000-Way Classifier, Li et. al, AAAI 2018, [[arXiv](https://arxiv.org/pdf/1802.01777.pdf)][[code](https://github.com/mtli/BFFL)]
- Noise2Noise: Learning Image Restoration without Clean Data, Lehtinen et. al, ICML 2018 [[arXiv](https://arxiv.org/pdf/1803.04189.pdf)][[code](https://github.com/NVlabs/noise2noise)]. State-of-the image denoising network is trained using only noisy data!
- The Unreasonable Effectiveness of Texture Transfer for Single Image Super-resolution, Gondal et. al, CVPR 2018 [[arXiv](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Gondal_The_Unreasonable_Effectiveness_of_Texture_Transfer_for_Single_Image_Super-resolution_ECCVW_2018_paper.pdf)][[code](https://github.com/waleedgondal/Texture-based-Super-Resolution-Network)]

#### 2017:
- Discovering Causal Signals in Images, Lopez-Paz et. al, CVPR 2017, [[arXiv](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lopez-Paz_Discovering_Causal_Signals_CVPR_2017_paper.pdf)]


### Other:

- Winner's Curse? On Pace, Progress, and Empirical Rigor, D. Sculley et. al, 2018 [[openReview](Winner's Curse? On Pace, Progress, and Empirical Rigor )]
- Relational inductive biases, deep learning, and graph networks, Battaglia et. al, 2018, [[arXiv](https://arxiv.org/pdf/1806.01261.pdf)]
- The Next Decade in AI: Four Steps Towards Robust Artificial Intelligence, Marcus 2020, [[arXiv](https://arxiv.org/pdf/2002.06177.pdf)]
- Building machines that learn and think like people, Lake et. al, 2016, [[arXiv](https://arxiv.org/pdf/1604.00289.pdf)]



