
file1_content = []
file2_content = []

file_name = input("enter file name you want to split\n")

with open(file_name) as input_file:
    file1_content += [next(input_file) for _ in range(15000)]
    file2_content += [next(input_file) for _ in range(5000)]

train_set = ''.join(file1_content)
test_set = ''.join(file2_content)

with open("train_set_" + file_name, "w") as train_file:
    train_file.write(train_set)

with open("test_set_" + file_name, "w") as test_file:
    test_file.write(test_set)
