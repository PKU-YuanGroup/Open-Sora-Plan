# NaVit-v1
## Model
We provide an open-source implementation of NaVit (https://arxiv.org/abs/2307.06304) and
its video version. 

For the image version of NaVit, we follow the original text,
allowing input of any resolution and embedding positions in three versions:
learnable, fixed, and Fourier.

For the video version of NaVit, we referred to variant-3 of Latte (https://github.com/Vchitect/Latte) and used alternating spatial-temporal self-attention to extract information from both temporal and spatial dimensions.
This requires the input video to have the same frame rate,
but still allows for videos of different resolutions.
![img](.\architecture.svg)
For the temporal dimension, we use fixed position embedding.
## Training
We provide a script for training NaVit. At present, there is no clear indication in the open-sora plan on which tasks to integrate NaViT and existing modules,
so we temporarily adopt the tasks from the original paper,
including image classification and language-image contrastive learning.
During the training process, we use the token dropout strategy proposed by the original author to improve model performance.
## Dataset
For image classification task, the training set should be organized as the form of "base/class-xxx/img-xxx.png",
and we assume that the resolution of images in each class is uniformly distributed.

For video classification task, the training set should be split into sub-videos with the same frame rate, and
 organized as the form of image set.

For language-image contrastive learning task, there is currently no clear restriction on the organization of the training set,
and we will adjust it in the future based on the specific pre-trained text model and dataset used.

JFT-4B and WebLI used in the original paper are not yet publicly available,
and we are looking forward to finding suitable alternative datasets.
## Structure 
├── README.md  
├── architecture.svg  
├── config  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── cfg_util.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── navit_default.yml  
├── dataset   
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── data_util.py  
├── model  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── clip.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── navit.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── navit_video.py  
├── scripts  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── train.py