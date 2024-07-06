# read the content of .txt file
with open('test_txt_1/0.txt', 'r') as file:
    lines = file.readlines()

# deal with each line in the .txt
processed_lines = []
for line in lines:
    # remove the lines starting with #
    if not line.startswith('#'):
        # change , to ,
        processed_line = line.replace(', ', ',')
        processed_lines.append(processed_line)

# write the processed files into new file
with open('test_txt_1/0.txt', 'w') as file:
    file.writelines(processed_lines)
