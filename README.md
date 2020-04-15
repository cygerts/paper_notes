List of interesting papers in Machine Learning, biased towards robustness, uncertainty estimation, understanding neural nets, computer vision.

### Intriguing properties of NN:

#### 2020:
- Overinterpretation reveals image classification model pathologies, Carter et. al, [[arXiv](https://arxiv.org/pdf/2003.08907.pdf)]. They remove 95% of the pixels from CIFAR-10 and ImageNet images without significantly changing accuracy of the classifier. What is more the remaining 5% pixels concentrate on the background and are nonsensical to humans, which looks like shortcut to solve the benchmark.
- What’s Hidden in a Randomly Weighted Neural Network?, Ramanujan et. al, [[arXiv](https://arxiv.org/pdf/1911.13299.pdf)]. Sufficiently big randomly initialized! neural networks contains a subnetwork that achieves competitive accuracy. "In short, we validate the unreasonable effectiveness of randomly weighted neural networks for image recognition."
- High-frequency Component Helps Explain the Generalization of Convolutional Neural Networks, Wang et. al, [[arXiv](https://arxiv.org/pdf/1905.13545.pdf)]

#### 2019:
- The lottery ticket hypothesis: finding sparse, trainable neural networks, Frankle & Carbin, ICLR 2019, [[arXiv](https://arxiv.org/pdf/1803.03635.pdf)][[code](https://github.com/google-research/lottery-ticket-hypothesis)]. For simple classification problems after network traning we can remove around 90% of neurons and still get very similar accuracy, the neurons that were not removed won "lottery ticket" - their randomly initialized weights made the traning particularly effective.
- Weight Agnostic Neural Networks, Gaiger & Ha, NeurIPS 2019, [[arXiv](https://arxiv.org/pdf/1906.04358.pdf)] [[DEMO](https://weightagnostic.github.io/)] Architecture search for randomly!! initialized neural networks which can perform reasonably well simple reinforcement learning and classifications tasks.
- Exploring Randomly Wired Neural Networks for Image Recognition, Xie et. al, [[arXiv](https://arxiv.org/pdf/1904.01569.pdf)][[code](https://github.com/seungwonpark/RandWireNN)] Neural architecure search for image recognition tasks starting from randomly wired architectures. Some bizzare looking architecture got really good results.
- Adversarial examples are not bugs, they are features, Ilyas et. al, NeurIPS 2019, [[arXiv](https://arxiv.org/pdf/1905.02175.pdf)][[BLOG](http://gradientscience.org/adv/)][[DISCUSSION](https://distill.pub/2019/advex-bugs-discussion/)] Such an amazing paper, they show that you can learn a classifier purely on adversarial examples which will generalize to TRUE test set! Anothe
- A fourier perspective on model robustness in computer vision, Dong et. al, NeurIPS 2019, [[arXiv](https://arxiv.org/pdf/1906.08988.pdf)]. They study how different augmentation techniques affects resulting model sensitivity to corruptions with different Fourier basis vectors, i.e. with regard no noise concentraed in high vs low frequency domain, very insightful.
- Benchmarking Neural Network Robustness to Common Corruptions and Perturbations, Hendrycks et. al, ICLR 2019, [[arXiv](https://arxiv.org/pdf/1903.12261.pdf)]. Introduces new benchmark for *common corruptions* (16 different types of noise, blur, adverse weather conditions) and show drastic reduction of accuracy. Also look at Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming by Michaelis et. al [[arXiv](https://arxiv.org/pdf/1907.07484.pdf)].
- Natural Adversarial Examples, D. Hendrycks et. al, [[arXiv](https://arxiv.org/pdf/1907.07174.pdf)] "Advesarial" images in the real world. Two another variants of ImageNet dataset: first is a set of natural images which current classifiers fail to classify, and the second for out of distribution detection.

#### 2018:
- ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness, Geirhos et. al, ICLR 2019, [[arXiv](https://arxiv.org/pdf/1811.12231.pdf)][[code](https://github.com/rgeirhos/texture-vs-shape)] They show that current classification algorithms are biased towards texture, using style transfer as data augmentation helps to partially mitigate the problem.
- Excessive invariance causes adversarial vulnerability, Jacobsen et. al, ICLR 2018, [[arXiv](https://arxiv.org/pdf/1811.00401.pdf)]. Another article on brittleness of neural networks. Authors create a "chating" variant of MNIST where the target label is encoded by adding artificial pixel to the image. Model trained on cheating version and tested on normal performed poorly, showing that neural nets learn non-robust features (when they help to minize the loss function). 
- Same-different problems strain convolutional neural networks, Ricci et. al,  [[arXiv](https://arxiv.org/pdf/1802.03390.pdf)]. They show one of the simplest examples of visual problems that current ml algorithms cannot solve.
- Deep Image Prior, Ulyanov et. al, CVPR 2018 [[arXiv](https://arxiv.org/pdf/1711.10925.pdf)][[code](https://github.com/DmitryUlyanov/deep-image-prior)] They use randomly-initialized neural network for inverse problems (denosing, super-resolution, image inpainting) to fit a single image! which results in competitive results.
- An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution, Liu et. al, NeurIPS 2018, [[arXiv](https://arxiv.org/abs/1807.03247)] [[BLOG](https://eng.uber.com/coordconv/)] 

#### 2015:
- Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images, Nguyen et. al, CVPR 2015, [[arXiv](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf)]

### CV / ML:

#### 2020:
- A critical analysis of self-supervision, or what can we learn from a single image, Asano et. al, ICLR 2020, [[arXiv](https://arxiv.org/pdf/1904.13132.pdf)] Another cool paper on learning from single sample. They show that using self-supervision from just 1 image! is enough to train first layers of neural net. Adding large scale data for deeper layers improve over 1 image baseline (but not as much as we thought) and adding even more data is unlikely to close the gap with strong supervision (with current self-supervised methods). 
- PointRend: Image Segmentation as Rendering, Kirillov et. al, [[arXiv](https://arxiv.org/pdf/1912.08193v2.pdf)]. New approach to image segmentation from randomly selected point-wise features, cool stuff.
- Self-training with Noisy Student improves ImageNet classification, Xie et. al, [[arXiv](https://arxiv.org/pdf/1911.04252.pdf)] On the importance of using noise during training.

#### 2019:
- Learning by Cheating, Chen et. al, CoRL 2019, [[arXiv](https://arxiv.org/pdf/1912.12294.pdf)][[code](https://github.com/dianchen96/LearningByCheating)]
- RepPoints: Point Set Representation for Object Detection, Yang et. al, [[arXiv](https://arxiv.org/pdf/1904.11490.pdf)][[code](https://github.com/microsoft/RepPoints)]. Object detector supervised by target bounding boxes as all other detectors, additionally learns point-wise representation (without explicit supervision). 
- Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift, Ovadia et. al, NeurIPS 2019, [[arXiv ](https://arxiv.org/pdf/1906.02530.pdf)][[code](https://github.com/google-research/google-research/tree/master/uq_benchmark_2019)] Benchmarking various uncertainty estimation methods uner distributional shift.

#### 2018:
- Deep Reinforcement Learning that Matters, Henderson et. al, AAAI 2018, [[arXiv](https://arxiv.org/pdf/1709.06560.pdf)]. On reproduction of baselines in RL, they show that random seeds matters can greatly alter the results and many more, nice read.
- A Probabilistic U-Net for Segmentation of Ambiguous Images, Kohl etl al, DeepMind, NIPS 2019, [[arXiv](https://arxiv.org/pdf/1806.05034.pdf)] [[code](https://github.com/SimonKohl/probabilistic_unet)] Generative semantic segmentation network which allows to sample diverse and consistent segmentation variants. [[Follow-up](https://arxiv.org/pdf/1905.13077.pdf)] 
- The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, Zhang et. al, CVPR 2018, [[arXiv](https://arxiv.org/pdf/1801.03924.pdf)][[code](https://github.com/richzhang/PerceptualSimilarity)]
- Brute-Force Facial Landmark Analysis With InceA 140,000-Way Classifier, Li et. al, AAAI 2018, [[arXiv](https://arxiv.org/pdf/1802.01777.pdf)][[code](https://github.com/mtli/BFFL)]
- Noise2Noise: Learning Image Restoration without Clean Data, Lehtinen et. al, ICML 2018 [[arXiv](https://arxiv.org/pdf/1803.04189.pdf)][[code](https://github.com/NVlabs/noise2noise)]. State-of-the image denoising network is trained using only noisy data (=without ever seing clean image)!
- Rethinking ImageNet Pre-Training, He et. al, ICCV 2019, [[arXiv](http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf)]. They show that ImageNet pretraining is not as crucial as we thought.
- The Unreasonable Effectiveness of Texture Transfer for Single Image Super-resolution, Gondal et. al, CVPR 2018 [[arXiv](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Gondal_The_Unreasonable_Effectiveness_of_Texture_Transfer_for_Single_Image_Super-resolution_ECCVW_2018_paper.pdf)][[code](https://github.com/waleedgondal/Texture-based-Super-Resolution-Network)]

#### 2017:
- Discovering Causal Signals in Images, Lopez-Paz et. al, CVPR 2017, [[arXiv](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lopez-Paz_Discovering_Causal_Signals_CVPR_2017_paper.pdf)]


### Other:
- How to do Research at the MIT AI Lab, 1988 [[LINK](https://dspace.mit.edu/bitstream/handle/1721.1/41487/AI_WP_316.pdf?sequence=4&isAllowed=y)]
- Adversarial Examples and the Deeper Riddle of Induction: The Need for a Theory of Artifacts in Deep Learning, Buckner, 2020. [[arXiv](https://arxiv.org/pdf/2003.11917.pdf)]
- Winner's Curse? On Pace, Progress, and Empirical Rigor, D. Sculley et. al, 2018 [[openReview](https://openreview.net/pdf?id=rJWF0Fywf)]
- Relational inductive biases, deep learning, and graph networks, Battaglia et. al, 2018, [[arXiv](https://arxiv.org/pdf/1806.01261.pdf)]
- The Next Decade in AI: Four Steps Towards Robust Artificial Intelligence, Marcus 2020, [[arXiv](https://arxiv.org/pdf/2002.06177.pdf)]
- Building machines that learn and think like people, Lake et. al, 2016, [[arXiv](https://arxiv.org/pdf/1604.00289.pdf)]



