with open('/home/meow/Food2KTW_net/label_output.txt', 'w') as file:
    for i in range(2000):
        file.write(f"'{i}': {i},\n")