import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import timm
from PIL import Image
import PIL

NUM_FINETUNE_CLASSES = 101

# Define the path to the validation dataset and trained model weight
image_path = "/home/meow/my_data_disk_5T/food_classification/TaiwaneseFood101/Taiwanese Food 101/images/"
val_txt_path = "/home/meow/my_data_disk_5T/food_classification/TaiwaneseFood101/Taiwanese Food 101/meta/validation.txt"
model_weight_path = "/home/meow/my_data_disk_5T/food_classification/TaiwaneseFood101/Taiwanese Food 101/final_model_inceptionv4.pth"

# Load the trained model
model = timm.create_model('inception_v4', pretrained=False, num_classes=NUM_FINETUNE_CLASSES)
model.load_state_dict(torch.load(model_weight_path))
model.eval()

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the transformations for validation data
normalize = transforms.Normalize(mean=[0.5457954, 0.44430383, 0.34424934],
                                 std=[0.23273608, 0.24383051, 0.24237761])
val_transforms = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    normalize
])

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')

# Create a custom dataset for validation
class MyValidationDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path, image_path, transform=None, loader=My_loader):
        data_txt = open(txt_path, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(' ')
            words2 = line.split('/')
            # imgs.append((words[0], words[1]))
            imgs.append(((words2[1].strip()), words2[0]))
        self.imgs = imgs
        self.transform = transform
        self.loader = My_loader
        self.image_path = image_path

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        # img = Image.open(img_name).convert('RGB')
        label_str = str(label + "/")
        img = self.loader(self.image_path + label_str + img_name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# Create a DataLoader for validation
val_dataset = MyValidationDataset(txt_path=val_txt_path, image_path=image_path, transform=val_transforms)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Function to calculate top-k accuracy
def top_5_accuracy(outputs, targets, topk=(5,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        # _, pred = outputs.topk(maxk, 1, True, True)
        values, indices = torch.topk(outputs, maxk)
        print("values: ", values)
        print("values.size: ", values.size())
        print("indices: ", indices)
        print("indices.size: ", indices.size())
        # pred = pred.t()
        # correct = pred.eq(targets.view(1, -1).expand_as(pred))
        y = torch.reshape(targets, [-1, 1])
        correct = (y == indices) * 1. 
        print("correct: ", correct)
        print("correct.size: ", correct.size())

        res = []
        for k in topk:
            # correct_k = correct_k_sel[:batch_size][:,0].reshape(-1).float().sum(0, keepdim=True)
            correct_k = correct.sum(dim=0, keepdim=True).sum(dim=1, keepdim=True)
            # print("correct[0,:]: ", correct[0,:])
            print("correct[:k][:,0]: ", correct[:batch_size][:,0])
            print("correct[:k].reshape(-1): ", correct[:batch_size][:,0].reshape(-1))
            print("correct_k: ", correct_k)
            res.append(correct_k)
            print("res: ", res)
        return res


def top_1_accuracy(outputs, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        # _, pred = outputs.topk(maxk, 1, True, True)
        values, indices = torch.topk(outputs, maxk)
        # print("values: ", values)
        # print("values.size: ", values.size())
        # print("indices: ", indices)
        # print("indices.size: ", indices.size())
        # pred = pred.t()
        # correct = pred.eq(targets.view(1, -1).expand_as(pred))
        y = torch.reshape(targets, [-1, 1])
        correct = (y == indices) * 1. 
        # print("correct: ", correct)
        # print("correct.size: ", correct.size())

        res = []
        for k in topk:
            correct_k = correct[:batch_size][:,0].reshape(-1).float().sum(0, keepdim=True)
            # print("correct[:k][:,0]: ", correct[:batch_size][:,0])
            # print("correct[:k].reshape(-1): ", correct[:batch_size][:,0].reshape(-1))
            # print("correct_k: ", correct_k)
            res.append(correct_k)
            # print("res: ", res)
        return res

# Inference on the validation dataset
model.eval()
top1_acc = 0.0
top5_acc = 0.0
total_samples = 0

with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc='Validation', unit='batch'):
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
        
        inputs = inputs.to(device)

        # labels should not be string
        labels = [label_to_int[label] for label in labels]
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # Forward pass
        outputs = model(inputs)
        total_samples += labels.size(0)

        # Calculate top-1 and top-5 accuracy
        # acc1, acc5 = top_k_accuracy(outputs, labels, topk=(1, 5))
        acc1 = top_1_accuracy(outputs, labels, topk=(1,))
        acc5 = top_5_accuracy(outputs, labels, topk=(5,))
        print("acc1: ", acc1)
        print("acc1[0].item(): ", acc1[0].item())
        print("acc5: ", acc5)
        print("acc5[0].item(): ", acc5[0].item())
        # print("labels: ", labels)
        top1_acc += acc1[0].item()
        top5_acc += acc5[0].item()

print("top1_acc: ", top1_acc)
print("top5_acc: ", top5_acc)
# Calculate final top-1 and top-5 accuracy
final_top1_acc = top1_acc / total_samples
final_top5_acc = top5_acc / total_samples

final_top1_acc *= 100
final_top5_acc *= 100

# Print the results
print(f'Top-1 Accuracy: {final_top1_acc:.2f}%')
print(f'Top-5 Accuracy: {final_top5_acc:.2f}%')
