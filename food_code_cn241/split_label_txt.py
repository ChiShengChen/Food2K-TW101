import random

def split_data(txt_file, train_ratio=0.7):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # 計算拆分的索引
    total_samples = len(lines)
    train_size = int(total_samples * train_ratio)

    # 隨機打亂順序
    random.shuffle(lines)

    # 分割成訓練和測試資料
    train_data = lines[:train_size]
    test_data = lines[train_size:]

    return train_data, test_data

def write_to_txt(data, output_txt):
    with open(output_txt, 'w') as file:
        file.writelines(data)

# 輸入的 txt 檔案名稱
input_txt = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/train600x600_raw.txt'

# 訓練和測試資料的比例
train_ratio = 0.7

# 拆分資料
train_data, test_data = split_data(input_txt, train_ratio)

# 輸出的 txt 檔案名稱
train_output_txt = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/train.txt'
test_output_txt = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/test.txt'

# 將資料寫入 txt
write_to_txt(train_data, train_output_txt)
write_to_txt(test_data, test_output_txt)
