# Food2K-TW101
All the accuracy (ACC) are the trained models do inference on un-seen validation data, not the training accuracy.

| Dataset | Model | Epochs | Optimizer | Top-1 ACC | Top-5 ACC | Pretrain | Augmentation type |
| :---------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| FOOD2K | PRENet-ResNet50 | N/A | `ongoing` | 83.03 | 97.21 | `ongoing` | `ongoing` |
| FOOD2K | Inception V4 | N/A | `ongoing` | 82.02 | 96.45 | `ongoing` | `ongoing` |
| FOOD2K | VGG16 | N/A | `ongoing` | 78.96 | 95.26 | `ongoing` | `ongoing` |

| Dataset | Model | Epochs | Optimizer | Top-1 ACC | Top-5 ACC | Pretrain | Augmentation type |
| :---------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| Taiwanese Food-101 | MobileFormer_508M | 100 | AdamW | 95.15 | 99.52 | imagenet-1k | Basic |
| Taiwanese Food-101 | PRENet-ResNet50 | 100 | SGD | 94.59 | 99.49 | food2K | Basic |
| Taiwanese Food-101 | PVTv2-B2-Linear | 100 | AdamW | 94.51 | 99.45 | imagenet-1k | Basic |
| Taiwanese Food-101 | ResNet50 | 100 | AdamW | 93.96 | 99.58 | imagenet-1k | Basic |
| Taiwanese Food-101 | Inception V4 | 100 | SGD | 92.14 | 99.01 | imagenet-1k | Basic |
| Taiwanese Food-101 | mobileViT v2 | 100 | AdamW | 89.98 | 98.46 | imagenet-1k | Basic |
| Taiwanese Food-101 | efficientNetv3 Large |100 | AdamW | 84.89 | 96.40 | imagenet-1k | Basic |
| Taiwanese Food-101 | efficient ViT_MIT | 100 | AdamW | 82.32 | 95.78 | imagenet-1k | Basic |
| Taiwanese Food-101 | RepViT_m2.3 | 100 | AdamW |  76.53 | 93.80 | imagenet-1k | Basic |
| Taiwanese Food-101 | RepViT_m0.9 | 100 | AdamW |  75.01 | 93.49 | imagenet-1k | Basic |
| Taiwanese Food-101 | VGG16 | N/A | `ongoing` |  67.65 | 89.33 | imagenet-1k | Basic |


| Dataset | Model | Epochs | Optimizer | Top-1 ACC | Top-5 ACC | Pretrain | Augmentation type |
| :---------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| FOOD2K-TW | PRENet-ResNet50 | N/A | `ongoing` | `ongoing` | `ongoing` | `ongoing` | `ongoing` |
| FOOD2K-TW | Inception V4 | 100 | SGD | 81.43 | 96.28 | imagenet-1k | Basic |
| FOOD2K-TW | VGG16 | N/A | `ongoing` | `ongoing` | `ongoing` | `ongoing` | `ongoing` |


## Do not trainable
| Dataset | Model | Epochs | Optimizer | Top-1 ACC | Top-5 ACC | Pretrain | Augmentation type |
| :---------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| Taiwanese Food-101  | convnextv2_huge.fcmae_ft_in1k | 15 | AdamW | 1.02 | 0.99 |  imagenet-1k | Normal |

## Augmentation type
| Type | Detail |
| :---------: | :--------: |
| Basic | RandomHorizontalFlip(p=0.5) <br> + RandomRotation(degrees=15) <br> + ColorJitter(brightness=0.126, saturation=0.5) <br> + Resize((550, 550)) <br> + RandomCrop(448)|


## To-Do List
| Dataset | Classes/Images | paper | Data Aviliable? |
| :---------: | :--------: | :--------: | :--------: |
| UEC Food256 | 256/25,088 | http://foodcam.mobi/taskcv2014.pdf | `ongoing` |
| ETH Food-101 | 101/101,000 | https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ | `ongoing` |
| Vireo Food-172 | 172/110,241 | https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?params=/context/sis_research/article/7501/&path_info=2964284.2964315.pdf | `ongoing` |
| Food524DB | 524/247,636 |  | `ongoing` |
| ChineseFoodNet | 208/192,000 | | `ongoing` |
| Sushi-50 | 50/3,963 | | `ongoing` |
| ISIA Food-500 | | | `ongoing` |



## To-Play List
https://arxiv.org/abs/2301.10936
