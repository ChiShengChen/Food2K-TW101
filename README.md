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
| Taiwanese Food-101 | ConvNeXtv2_nano | 100 | AdamW | 94.04 | 99.29 | imagenet-1k | Basic |
| Taiwanese Food-101 | ResNet50 | 100 | AdamW | 93.96 | 99.58 | imagenet-1k | Basic |
| Taiwanese Food-101 | ConvNeXtv2_femto | 100 | AdamW | 93.21 | 99.19 | imagenet-1k | Basic |
| Taiwanese Food-101 | MobileFormer_96M | 100 | AdamW | 93.01 | 99.25 | imagenet-1k | Basic |
| Taiwanese Food-101 | Inception V4 | 100 | SGD | 92.14 | 99.01 | imagenet-1k | Basic |
| Taiwanese Food-101 | mobilenetv3_large_100 | 100 | AdamW | 91.66 | 98.97 | imagenet-1k | Basic |
| Taiwanese Food-101 | mobileViT v2 | 100 | AdamW | 89.98 | 98.46 | imagenet-1k | Basic |
| Taiwanese Food-101 | efficientNetv3 Large |100 | AdamW | 84.89 | 96.40 | imagenet-1k | Basic |
| Taiwanese Food-101 | efficient ViT_MIT | 100 | AdamW | 82.32 | 95.78 | imagenet-1k | Basic |
| Taiwanese Food-101 | RepViT_m2.3 | 100 | AdamW |  76.53 | 93.80 | imagenet-1k | Basic |
| Taiwanese Food-101 | RepViT_m0.9 | 100 | AdamW |  75.01 | 93.49 | imagenet-1k | Basic |
| Taiwanese Food-101 | VGG16 | N/A | SGD |  67.65 | 89.33 | imagenet-1k | Basic |


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
| Dataset | Classes/Images | paper | dataset | Data Aviliable? |
| :---------: | :--------: | :--------: | :--------: | :--------: |
| Taiwanese-Food-101 | 101/20,200 | N/A (only master Thesis) | [dataset](https://drive.google.com/drive/folders/1wOdqes1KzEzp4DPG5jO8kHbFwcCmjzjh) | Yes |
| UEC Food256 | 256/25,088 | [paper](https://link.springer.com/chapter/10.1007/978-3-319-16199-0_1) | [dataset](http://foodcam.mobi/dataset256.html) | Yes |
| ETH Food-101 | 101/101,000 | [paper](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_29) | [dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) | Yes |
| Vireo Food-172 | 172/110,241 | [paper](https://dl.acm.org/doi/10.1145/2964284.2964315) | [dataset](https://fvl.fudan.edu.cn/dataset/vireofood172/list.htm) | Need to email |
| Food524DB | 524/247,636 | [paper](https://link.springer.com/chapter/10.1007/978-3-319-70742-6_41) | [dataset](http://www.ivl.disco.unimib.it/activities/food524db/) | Yes, but .mat format |
| CNFOOD-241 | 241/191,811 | N/A | [dataset](https://data.mendeley.com/datasets/fspyss5zbb/1) | Yes |
| ChineseFoodNet | 208/192,000 | [paper](https://arxiv.org/abs/1705.02743) | [dataset](https://sites.google.com/view/chinesefoodnet/) | [Link is dead, email didnot response](https://sites.google.com/view/chinesefoodnet/) |
| Sushi-50 | 50/3,963 | [paper](https://arxiv.org/abs/2207.03692) | [dataset](https://github.com/Jianing-Qiu/PARNet/tree/main/data) | Yes |
| ISIA Food-500 | 500/399,726 | [paper](https://arxiv.org/abs/2008.05655) | [dataset](http://123.57.42.89/FoodComputing-Dataset/ISIA-Food500.html) | Yes |
| Food2K | 2,000/1,036,564 | [paper](https://arxiv.org/abs/2103.16107) | [dataset](http://123.57.42.89/FoodProject.html) | Need to email. Yes |

[TaiwaneseFood101](https://github.com/106368015AlvinYang/Taiwanese-Food-101)
https://www.kaggle.com/datasets/kuantinglai/taiwanese-food-101/data

## To-Play List
https://arxiv.org/abs/2301.10936
https://www.kaggle.com/datasets/zachaluza/cnfood-241
https://universe.roboflow.com/search?q=class%3Adumpling
https://universe.roboflow.com/search?q=class%253Apork&p=1
https://ieeexplore.ieee.org/document/9214438
https://www.researchgate.net/publication/372133738_Deep_Learning_for_Food_Image_Recognition_and_Nutrition_Analysis_Towards_Chronic_Diseases_Monitoring_A_Systematic_Review
https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id=%22111NYPI0294006%22.&searchmode=basic
