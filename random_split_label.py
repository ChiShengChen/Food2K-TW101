import random

# input label format example:
'''
000/000668.jpg
000/000669.jpg
001/000000.jpg
001/000001.jpg
001/000002.jpg
'''
# output label format example:
'''
000/000668.jpg 0
000/000669.jpg 0
001/000000.jpg 1
001/000001.jpg 1
001/000002.jpg 1
'''
# and random split by catgory

def split_data_by_cat(txt_file, train_ratio, cat_num):

    with open(txt_file, 'r') as file:
        lines = file.readlines()
    
    # relable the format
    new_label_list = []
    for line in lines:
        line = line.strip()
        words = line.split('/')
        new_label_list.append(line + " " + str(int(words[0])) + '\n')
    
    cat_lists = [[] for _ in range(cat_num)]
    for n_line in new_label_list:
        n_line = n_line.strip()
        n_word = n_line.split(' ')
        cat_lists[int(n_word[-1])].append(n_line)
    
    out_train_data = []
    out_test_data = []
    for cat_fn in cat_lists:
        # print(cat_fn)
        total_samples = len(cat_fn)
        train_size = int(total_samples * train_ratio)
        random.shuffle(cat_fn)
        out_train_data.append('\n'.join(cat_fn[:train_size]) + '\n')
        out_test_data.append('\n'.join(cat_fn[train_size:]) + '\n')

    return out_train_data, out_test_data

def write_to_txt(data, output_txt):
    with open(output_txt, 'w') as file:
        file.writelines(data)


input_txt = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/train600x600_raw.txt'

train_ratio = 0.7
class_num = 241

train_data, test_data = split_data_by_cat(input_txt, train_ratio, class_num)

train_output_txt = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/train_n.txt'
test_output_txt = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/test_n.txt'

write_to_txt(train_data, train_output_txt)
write_to_txt(test_data, test_output_txt)
