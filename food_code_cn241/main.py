from models import get_model
from data_loader import load_data
import torch
from torch import nn
from torch.optim import lr_scheduler
from val_func import val, val_prenet
from train_func import train_general, train_prenet
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # args = parse_option()
    train_dataset, train_loader, test_dataset, test_loader = \
        load_data(image_path=args["image_path"], train_dir=args["train_path"], test_dir=args["test_path"], batch_size=args["batchsize"])
    # print('Data Preparation : Finished')
    criterion = torch.nn.CrossEntropyLoss()

    NUM_CATEGORIES = args["num_class"]  

    model = get_model(args["model_name"], use_pretrained=args["use_pretrained"], num_classes=NUM_CATEGORIES, weight_path=args["weight_path"], is_test=args["is_test"])
    
    # print(model)
    
    model.to(device)
    for param in model.parameters():
        param.requires_grad = True

    if args["model_name"] == 'prenet':
        ignored_params = list(map(id, model.features.parameters()))
        new_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer = torch.optim.SGD([
            {'params': model.features.parameters(), 'lr': args["learning_rate"]*0.1},
            {'params': new_params, 'lr': args["learning_rate"]}
        ],
            momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
   

    if args["is_test"]:
        if args["model_name"] == 'prenet':
            print('='*10+'testing'+'='*10)
            print(args["model_name"])
            print('='*20)
            val_loss, val_acc, val5_acc, val_acc_com, val5_acc_com = val_prenet(model, criterion, test_loader, device)
            print('Accuracy of the network on the val images: top1 = %.5f, top5 = %.5f, top1_combined = %.5f, top5_combined = %.5f, test_loss = %.6f\n' % (
                    val_acc, val5_acc, val_acc_com, val5_acc_com, val_loss))
            return
        else:
            print('='*10+'testing'+'='*10)
            print(args["model_name"])
            print('='*20)
            val_loss, val_acc, val5_acc = val(model, criterion, test_loader, device)
            print('val loss: {:.4f}, top1: {:.4f}, top5: {:.4f}'
            .format(val_loss, val_acc, val5_acc))
            return

    print('='*10+'training'+'='*10)
    print(args["model_name"])
    print('='*20)
    if args["model_name"] == 'prenet':
        train_prenet(model, train_loader, test_loader, criterion, optimizer, scheduler, args["epoch"], args["store_name"], device)
    else:
        train_general(model, train_loader, test_loader, criterion, optimizer, scheduler, args["epoch"], args["store_name"], device)

if __name__ == "__main__":
    import yaml
    
    def load_config(config_name):
        with open(os.path.join(CONFIG_PATH, config_name)) as file:
            config = yaml.safe_load(file)

        return config

    # folder to load config file
    CONFIG_PATH = "./configs/"
    config_files = os.listdir(CONFIG_PATH)
    # Function to load yaml configuration file
    for file in config_files:
        args = load_config(file)
        try:
            os.stat('outputs/' + args["store_name"])
        except:
            os.makedirs('outputs/' + args["store_name"])
        if not args["is_test"]:
            with open('outputs/' + args["store_name"]+'/config.yaml', 'w') as file:
                yaml.dump(args, file)
        main(args)