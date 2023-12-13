import os

def list_files(folder):
    file_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_paths.append(os.path.join(root.split('/')[-1], file))
    return file_paths

def write_to_txt(file_paths, output_txt):
    with open(output_txt, 'w') as file:
        for path in file_paths:
            file.write(f"{path}\n")

# 資料夾路徑
folder_path = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/train600x600'

# 取得所有檔案路徑
files = list_files(folder_path)

# 輸出的 txt 檔案名稱
output_txt = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/train600x600_raw.txt'

# 將檔案路徑寫入 txt
write_to_txt(files, output_txt)
