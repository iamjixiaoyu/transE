import os
import codecs


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


def load_data(dirname, dataname = 'train'):
    # 判断文件夹是否存在
    if not os.path.exists(dirname):
        print('文件夹路径错误！请确认路径是否存在')
        return None
    dir_path = os.path.abspath(dirname)
    entity2id_dict_path = os.path.join(dir_path, 'entity2id.txt')
    relation2id_dict_path = os.path.join(dir_path, 'relation2id.txt')
    data_path = os.path.join(dir_path, dataname + '.txt')
    entity2id_dict = load_dict(entity2id_dict_path)
    relation2id_dict = load_dict(relation2id_dict_path)

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
    # 打印三元组数量信息
    print(f'数据集名称：{dataname}')
    print(f'实体数量：{len(entity2id_dict)}')
    print(f'关系数量：{len(relation2id_dict)}')
    print(f'三元组数量：{len(triple_list)}')
    # 返回名称型的三元组列表
    return triple_list


def load_data_all(dirname):
    return None





















































