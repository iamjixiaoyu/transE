import os
import codecs


# 输入参数：文件路径
# 输出参数：name2id词典
def load_dict(path):
    # 判断文件是否存在
    if not os.path.exists(path):
        print('文件路径错误！请确认路径是否存在')
        return None
    # 定义名称向数字转换的字典
    name2id_dict = {}
    # 按行读取文件内容，并根据文件内容向字典中添加数据
    for line in codecs.open(path, 'r').readlines():
        # 每一行的文本结构为：name\tid
        splited_line = line.strip().split('\t')
        # 判断分割结果的数量是否为2，如果为2则添加数据，如果不为2则说明有错误，跳过该行
        if len(splited_line) != 2:
            continue
        else:
            name = str(splited_line[0])
            id = int(splited_line[1])
            name2id_dict[name] = id
    # 返回字典
    return name2id_dict


# 输入参数：
#   (1)文件路径；
#   (2)实体name2id字典；
#   (3)关系name2id字典
# 输出参数：
#   三元组列表
def load_data(data_path, entity2id_dict, relation2id_dict):
    # 判断文件夹是否存在
    if not os.path.exists(data_path):
        print('文件夹路径错误！请确认路径是否存在')
        return None

    # 定义名称型的三元组列表
    triple_list = []
    # 按行读取文件内容，并根据文件内容向列表中添加数据
    for line in codecs.open(data_path, 'r').readlines():
        # 每一行的文本结构为：head_name\ttail_name\trelation_name
        splited_line = line.strip().split('\t')
        # 判断分割结果的数量是否为3，如果为3则添加数据，如果不为3则说明有错误，跳过该行
        if len(splited_line) != 3:
            continue
        else:
            head_name, tail_name, relation_name = [str(name) for name in splited_line]
            head_id = entity2id_dict[head_name]
            tail_id = entity2id_dict[tail_name]
            relation_id = relation2id_dict[relation_name]
            triple_list.append([head_id, tail_id, relation_id])
    # 返回名称型的三元组列表
    return triple_list


# 输入参数：数据集文件夹
# 输出参数：
#   (1)训练集三元组列表
#   (2)验证集三元组列表
#   (3)测试集三元组列表
#   (4)实体name2id字典
#   (5)关系name2id字典
def load_all_data(dirname):
    # 获取数据集文件夹的绝对路径
    dir_path = os.path.abspath(dirname)
    # 训练集文件路径
    train_data_path = os.path.join(dir_path, 'train.txt')
    # 验证集文件路径
    valid_data_path = os.path.join(dir_path, 'valid.txt')
    # 测试集文件路径
    test_data_path = os.path.join(dir_path, 'test.txt')
    # 实体name2id字典文件路径
    entity2id_dict_path = os.path.join(dir_path, 'entity2id.txt')
    # 关系name2id字典文件路径
    relation2id_dict_path = os.path.join(dir_path, 'relation2id.txt')
    # 获取实体name2id字典
    entity2id_dict = load_dict(entity2id_dict_path)
    print('实体字典加载完成：', len(entity2id_dict))
    # 获取关系name2id字典
    relation2id_dict = load_dict(relation2id_dict_path)
    print('关系字典加载完成：', len(relation2id_dict))
    # 获取训练集三元组列表
    train_list = load_data(train_data_path, entity2id_dict, relation2id_dict)
    print('训练集加载完成：', len(train_list))
    # 获取验证集三元组列表
    valid_list = load_data(valid_data_path, entity2id_dict, relation2id_dict)
    print('验证集加载完成：', len(valid_list))
    # 获取测试集三元组列表
    test_list = load_data(test_data_path, entity2id_dict, relation2id_dict)
    print('测试集加载完成：', len(test_list))
    # 返回全部三元组列表与字典
    return train_list, valid_list, test_list, entity2id_dict, relation2id_dict























































