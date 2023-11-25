import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import timm
import PIL
from tqdm import tqdm

NUM_FINETUNE_CLASSES = 101

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txt_dir, image_path, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            # print("line: ", line)
            words = line.split(' ')
            words2 = line.split('/')
            # print("words2[1].strip()_img_name: ", words2[1].strip())
            # print("words2[0]_label :", words2[0])
            imgs.append(((words2[1].strip()), words2[0]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader
        self.image_path = image_path

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        # label = list(map(int, label))
        # print label
        # print type(label)
        #img = self.loader('/home/vipl/llh/food101_finetuning/food101_vgg/origal_data/images/'+img_name.replace("\\","/"))
        label_str = str(label + "/")
        img = self.loader(self.image_path + label_str + img_name)
        # print("self.image_path: ", self.image_path)
        # print img
        if self.transform is not None:
            img = self.transform(img)
            # print img.size()
            # label =torch.Tensor(label)
            # print label.size()
        return img, label
        # if the label is the single-label it can be the int
        # if the multilabel can be the list to torch.tensor

def load_data(image_path, train_dir, test_dir, batch_size):
    normalize = transforms.Normalize(mean=[0.5457954, 0.44430383, 0.34424934],
                                     std=[0.23273608, 0.24383051, 0.24237761])
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # default value is 0.5
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.126, saturation=0.5),
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448),
        transforms.ToTensor(),
        normalize
    ])

    # transforms of test dataset
    test_transforms = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        normalize
    ])
    train_dataset = MyDataset(txt_dir=train_dir, image_path=image_path, transform=train_transforms)
    test_dataset = MyDataset(txt_dir=test_dir, image_path=image_path, transform=test_transforms)
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size//2,  shuffle=False, num_workers=0)
    return train_dataset, train_loader, test_dataset, test_loader



# Load the pre-trained ResNet-18 model
# model = models.resnet18(pretrained=True)
# model = timm.create_model('inception_v4', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
model = timm.create_model('efficientvit_b3.r288_in1k', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
# print(model)

# Freeze all the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the last layer of the model
num_classes = NUM_FINETUNE_CLASSES # replace with the number of classes in your dataset
# model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the train and validation datasets
# train_dataset = ImageFolder('custom_dataset/train', transform=transform)
# val_dataset = ImageFolder('custom_dataset/val', transform=transform)

image_path = "/home/meow/my_data_disk_5T/food_classification/TaiwaneseFood101/Taiwanese Food 101/images/"
train_dir = "/home/meow/my_data_disk_5T/food_classification/TaiwaneseFood101/Taiwanese Food 101/meta/train.txt"
test_dir = "/home/meow/my_data_disk_5T/food_classification/TaiwaneseFood101/Taiwanese Food 101/meta/test.txt"
batch_size = 16

train_dataset, train_loader, val_dataset, val_loader = load_data(image_path, train_dir, test_dir, batch_size)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-5)

lr = 1e-3
# Create data loaders for the train and validation datasets
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    label_to_int = {
        'bawan' : 0,
        'beef_noodles' : 1,
        'beef_soup' : 2,
        'bitter_melon_with_salted_eggs' : 3,
        'braised_napa_cabbage' : 4,
        'braised_pork_over_rice' : 5,
        'brown_sugar_cake' : 6,
        'bubble_tea' : 7,
        'caozaiguo' : 8,
        'chicken_mushroom_soup' : 9,
        'chinese_pickled_cucumber' : 10,
        'coffin_toast' : 11,
        'cold_noodles' : 12,
        'crab_migao' : 13,
        'deep-fried_chicken_cutlets' : 14,
        'deep_fried_pork_rib_and_radish_soup' : 15,
        'dried_shredded_squid' : 16,
        'egg_pancake_roll' : 17,
        'eight_treasure_shaved_ice' : 18,
        'fish_head_casserole' : 19,
        'fried-spanish_mackerel_thick_soup' : 20,
        'fried_eel_noodles' : 21,
        'fried_instant_noodles' : 22,
        'fried_rice_noodles' : 23,
        'ginger_duck_stew' : 24,
        'grilled_corn' : 25,
        'grilled_taiwanese_sausage' : 26,
        'hakka_stir-fried' : 27, 
        'hot_sour_soup' : 28,
        'hung_rui_chen_sandwich' : 29,
        'intestine_and_oyster_vermicelli' : 30,
        'iron_egg' : 31,
        'jelly_of_gravey_and_chicken_feet_skin' : 32,
        'jerky' : 33,
        'kung-pao_chicken' : 34,
        'luwei' : 35,
        'mango_shaved_ice' : 36,
        'meat_dumpling_in_chili_oil' : 37,
        'milkfish_belly_congee' : 38,
        'mochi' : 39,
        'mung_bean_smoothie_milk' : 40,
        'mutton_fried_noodles' : 41,
        'mutton_hot_pot' : 42, 
        'nabeyaki_egg_noodles' : 43,
        'night_market_steak' : 44,
        'nougat' : 45,
        'oyster_fritter' : 46, 
        'oyster_omelet' : 47,
        'papaya_milk' : 48,
        'peanut_brittle' : 49,
        'pepper_pork_bun' : 50,
        'pig\'s_blood_soup' : 51,
        'pineapple_cake' : 52,
        'pork_intestines_fire_pot' : 53,
        'potsticker' : 54, 
        'preserved_egg_tofu' : 55,
        'rice_dumpling' : 56,
        'rice_noodles_with_squid' : 57,
        'rice_with_soy-stewed_pork' : 58, 
        'roasted_sweet_potato' : 59,
        'sailfish_stick' : 60,
        'salty_fried_chicken_nuggets' : 61,
        'sanxia_golden_croissants' : 62,
        'saute_spring_onion_with_beef' : 63,
        'scallion_pancake' : 64,
        'scrambled_eggs_with_shrimp' : 65,
        'scrambled_eggs_with_tomatoes' : 66,
        'seafood_congee' :67,
        'sesame_oil_chicken_soup' : 68,
        'shrimp_rice' : 69,
        'sishen_soup' : 70,
        'sliced_pork_bun' : 71,
        'spicy_duck_blood' : 72,
        'steam-fried_bun' : 73,
        'steamed_cod_fish_with_crispy_bean' : 74,
        'steamed_taro_cake' : 75,
        'stewed_pig\'s_knuckles' : 76,
        'stinky_tofu' : 77,
        'stir-fried_calamari_broth' : 78,
        'stir-fried_duck_meat_broth' : 79,
        'stir-fried_loofah_with_clam' : 80,
        'stir-fried_pork_intestine_with_ginger' : 81,
        'stir_fried_clams_with_basil' : 82,
        'sugar_coated_sweet_potato' : 83,
        'sun_cake' : 84,
        'sweet_and_sour_pork_ribs' : 85,
        'sweet_potato_ball' : 86,
        'taiwanese_burrito' : 87,
        'taiwanese_pork_ball_soup' : 88,
        'taiwanese_sausage_in_rice_bun' : 89,
        'tanghulu' : 90,
        'tangyuan' : 91,
        'taro_ball' : 92,
        'three-cup_chicken' : 93,
        'tube-shaped_migao' : 94,
        'turkey_rice' : 95,
        'turnip_cake' : 96,
        'twist_roll' : 97,
        'wheel_pie' : 98,
        'xiaolongbao' : 99,
        'yolk_pastry' : 100
    }    


    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Create a tqdm progress bar for the training data loader
        train_loader = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        # Iterate over the batches of the train loader
        for inputs, labels in train_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            # print("inputs: ", inputs)
            # print("labels: ", labels)

            # labels should not be string
            labels = [label_to_int[label] for label in labels]
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            #    labels = labels.to(device)
          
            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Calculate the train loss and accuracy
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        # Set the model to evaluation mode
        model.eval()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the batches of the validation loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move the inputs and labels to the device
                inputs = inputs.to(device)
                # labels = labels.to(device)
                labels = [label_to_int[label] for label in labels]
                labels = torch.tensor(labels, dtype=torch.long).to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # Calculate the validation loss and accuracy
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    
    # Save the final trained model
    model_name = 'effientvit_mit.pth'
    torch.save(model.state_dict(), '/home/meow/my_data_disk_5T/food_classification/TaiwaneseFood101/Taiwanese Food 101/'+model_name)
    print("Model have saved in /home/meow/my_data_disk_5T/food_classification/TaiwaneseFood101/Taiwanese Food 101/"+model_name)


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Fine-tune the last layer for a few epochs
# Learning rate scheduler; adjusts the learning rate during training
# lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
#                                                    max_lr=lr, 
#                                                    total_steps=num_epochs*len(train_dataloader))
# metric = MulticlassAccuracy()
# optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
# train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

# Unfreeze all the layers and fine-tune the entire network for a few more epochs
for param in model.parameters():
    param.requires_grad = True
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #val=0.08
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)
