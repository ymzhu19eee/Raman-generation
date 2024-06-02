# 读取文本文件内容
with open('test_txt_1/0.txt', 'r') as file:
    lines = file.readlines()

# 处理每一行文本
processed_lines = []
for line in lines:
    # 去掉前面带有 # 开头的行
    if not line.startswith('#'):
        # 将 , 替换为 ,
        processed_line = line.replace(', ', ',')
        processed_lines.append(processed_line)

# 将处理后的文本写入新文件
with open('test_txt_1/0.txt', 'w') as file:
    file.writelines(processed_lines)