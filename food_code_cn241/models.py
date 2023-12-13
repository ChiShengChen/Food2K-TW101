from timm.models import create_model
from torchvision import models
from torch import nn
import torch
import re

from mobileformer.mobile_former import mobile_former_508m, mobile_former_96m
from prenet.resnet import resnet50
from prenet.prenet import PRENet

# timm_models = ['convnextv2_nano.fcmae_ft_in1k', 'convnextv2_femto.fcmae_ft_in1k', 'mobilevitv2_100.cvnets_in1k', 'mobilevitv2_200.cvnets_in22k_ft_in1k_384',
# 'pvt_v2_b2_li.in1k', 'mobilenetv3_large_100', 'repvit_m0_9', 'repvit_m2_3']
# pytorch_models = ['resnet50']
# custom_models = ['mobile_former_508m', 'mobile_former_96m', 'prenet']

timm_models = []
pytorch_models = []
custom_models = ['prenet']

total_model_lists = timm_models + pytorch_models + custom_models

def get_model(model_name, use_pretrained, num_classes, weight_path=None, is_test=False):
    if model_name not in total_model_lists:
        print('Model is not included yet!')
        model = None
    elif model_name in timm_models:
        model = create_model(model_name, pretrained=use_pretrained, num_classes=num_classes)
        if is_test:
            model.load_state_dict(torch.load(weight_path))
    elif model_name in pytorch_models:
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=use_pretrained, progress=True)
            model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        if is_test:
            model.load_state_dict(torch.load(weight_path))
    elif model_name in custom_models:
        if model_name == 'mobile_former_508m':
            if is_test:
                model = mobile_former_508m()
                model.classifier.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.0, inplace=False),
                torch.nn.Linear(in_features=1920, out_features=num_classes, bias=True)
                )
                model.load_state_dict(torch.load(weight_path))
            elif use_pretrained:
                model = mobile_former_508m()
                model.load_state_dict(torch.load(weight_path)['state_dict'])
                model.classifier.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.0, inplace=False),
                torch.nn.Linear(in_features=1920, out_features=num_classes, bias=True)
                )
            else:
                model = mobile_former_508m()
                model.classifier.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.0, inplace=False),
                torch.nn.Linear(in_features=1920, out_features=num_classes, bias=True)
                )
            
        if model_name == 'mobile_former_96m':
            if is_test:
                model = mobile_former_96m()
                model.classifier.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.0, inplace=False),
                torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
                )
                model.load_state_dict(torch.load(weight_path))
            elif use_pretrained:
                model = mobile_former_96m()
                model.load_state_dict(torch.load(weight_path)['state_dict'])
                model.classifier.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.0, inplace=False),
                torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
                )
            else:
                model = mobile_former_96m()
                model.classifier.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.0, inplace=False),
                torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
                )
        if model_name == 'prenet':
            if is_test:
                model = resnet50(pretrained=False)
                model = PRENet(model, 512, classes_num=2000)
                state_dict = {}
                pretrained = torch.load('pretrained_models/mobile-former-508m.pth.tar')
                #print(pretrained.keys())
                for k, v in model.state_dict().items():
                    if k[9:] in pretrained.keys() and "fc" not in k:
                        state_dict[k] = pretrained[k[9:]]
                    elif "xx" in k and re.sub(r'xx[0-9]\.?',".", k[9:]) in pretrained.keys():
                        state_dict[k] = pretrained[re.sub(r'xx[0-9]\.?',".", k[9:])]
                    else:
                        state_dict[k] = v
                        # print(k)

                model.load_state_dict(state_dict)
                model.fc = nn.Linear(2048, num_classes)
                model.load_state_dict(torch.load(weight_path))
            elif use_pretrained:
                model = resnet50(pretrained=False)
                model = PRENet(model, 512, classes_num=2000)
                # model.fc = nn.Linear(2048, 1000)
                state_dict = {}
                pretrained = torch.load(weight_path)
                #print(pretrained.keys())
                for k, v in model.state_dict().items():
                    if k[9:] in pretrained.keys() and "fc" not in k:
                        state_dict[k] = pretrained[k[9:]]
                    elif "xx" in k and re.sub(r'xx[0-9]\.?',".", k[9:]) in pretrained.keys():
                        state_dict[k] = pretrained[re.sub(r'xx[0-9]\.?',".", k[9:])]
                    else:
                        state_dict[k] = v
                        # print(k)

                model.load_state_dict(state_dict)
                model.fc = nn.Linear(2048, num_classes)
            else:
                model = resnet50(pretrained=False)
                model = PRENet(model, 512, classes_num=num_classes)

    return model