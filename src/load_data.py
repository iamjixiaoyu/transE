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
#   (1)数据集顶层文件夹
#   (2)模式mode：'train', 'valid', 'test'
# 输出参数：
#   (1)实体集
#   (2)关系集
#   (3)三元组列表
def load_data(data_dir, mode='train'):
    # 判断文件夹是否存在
    if not os.path.exists(data_dir):
        print('文件夹路径错误！请确认路径是否存在')
        return None
    # 获取实体和关系name2id字典文件路径
    entity2id_path = os.path.join(os.path.abspath(data_dir), 'entity2id.txt')
    relation2id_path = os.path.join(os.path.abspath(data_dir), 'relation2id.txt')
    # 生成实体和关系name2id字典
    entity2id_dict = load_dict(entity2id_path)
    relation2id_dict = load_dict(relation2id_path)
    # 定义实体集和关系集
    entity_set, relation_set = set(), set()
    # 定义名称型的三元组列表
    triple_list = []
    # 获取训练、验证、测试集文件路径
    data_path = os.path.join(os.path.abspath(data_dir), mode + '.txt')
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
            entity_set.add(head_id)
            entity_set.add(tail_id)
            relation_set.add(relation_id)
            triple_list.append([head_id, tail_id, relation_id])
    # 返回数字型的实体集、关系集和三元组列表
    return entity_set, relation_set, triple_list


























































