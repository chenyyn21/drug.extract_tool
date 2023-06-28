from matplotlib import pyplot as plt
datafile = 'new_labels_0_10'
file_AUCs = 'data/result/skip/skip_1GATNet(DrugA_DrugB)' + '1' + '--AUCs--' + datafile + '.txt'
AUC_map = 'data/result/picture/skip_1GATNet(DrugA_DrugB)' + '1' + '--AUCs--' + datafile + '.png'
with open(file_AUCs, 'r') as file:
    # 读取文件的每一行
    lines = file.readlines()

# 提取所需列的数据
column_data1 = []
column_data2 = []
for line in lines:
    # 根据文件的分隔符（例如空格、逗号等）拆分每一行的数据
    values = line.split('\t')  # 如果文件中的列使用制表符分隔，可以使用'\t'作为分隔符
    # 假设所需的列是第二列（索引为1），将该列的数据添加到列表中
    column_data1.append(values[0])
    column_data2.append(values[1][:6])

# 绘制曲线图
plt.plot(column_data1, column_data2)

plt.xlabel('Epoch')
plt.ylabel('AUC_dev')
plt.title('AUC')
plt.savefig(AUC_map)
# 显示图形
plt.show()


