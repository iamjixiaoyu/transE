import numpy as np
import copy
import time
import codecs


class TransE:
    # 构造函数
    # 输入参数：
    #   (1)实体集
    #   (2)关系集
    #   (3)三元组列表
    #   (4)嵌入向量长度
    #   (5)学习率
    #   (6)边界值
    #   (7)范数
    def __init__(self, entity_set,
                 rel_set,
                 triple_list,
                 embed_dim=50,
                 lr=0.01,
                 margin=1.0,
                 norm=1):
        self.embed_dim = embed_dim
        self.lr = lr
        self.margin = margin
        self.norm = norm
        self.entities = entity_set
        self.rels = rel_set
        self.triple_list = triple_list
        self.loss = 0.0

    # 初始化向量
    def init_embeddings(self):
        # 定义id2embedding字典
        entity_dict = {}
        rel_dict = {}
        # 为每个实体初始化一个随机向量，向量长度为self.embed_dim
        for entity in self.entities:
            # 使用均匀分布，np.random.uniform(low, high, size)生成大小为size的[low, high)区间均匀分布的向量
            e_emb = np.random.uniform(-6 / np.sqrt(self.embed_dim),
                                      6 / np.sqrt(self.embed_dim),
                                      self.embed_dim)
            # 对向量进行L2范数归一化，保证其L2范数为1
            e_emb = e_emb / np.linalg.norm(e_emb, ord=2)
            # 加入id2embedding字典
            entity_dict[entity] = e_emb
        # 使用相同方法为每个关系初始化一个随机向量
        for rel in self.rels:
            r_emb = np.random.uniform(-6 / np.sqrt(self.embed_dim),
                                      6 / np.sqrt(self.embed_dim),
                                      self.embed_dim)
            r_emb = r_emb / np.linalg.norm(r_emb, ord=self.norm)
            rel_dict[rel] = r_emb
        # 用字典替换实体集和关系集：{id: embedding}
        self.entities = entity_dict
        self.rels = rel_dict

    # 负样本生成
    # 随机替换正样本三元组的头实体或尾实体来生成负样本
    # 输入参数：正样本三元组
    # 输出参数：负样本三元组
    def negative_sample(self, p_triple):
        # 进行深拷贝，避免赋值时更改正样本三元组
        n_triple = copy.deepcopy(p_triple)
        # 生成[0, 1)之间的随机种子
        seed = np.random.random()
        # 若种子大于0.5则替换头实体，否则替换尾实体
        if seed > 0.5:
            # 只要负样本还属于已知的正样本集合，则一直随机生成替换的头实体
            while n_triple in self.triple_list:
                rand_head = np.random.choice(len(self.entities))
                n_triple[0] = rand_head
        else:
            # 与替换头实体原理相同
            while n_triple in self.triple_list:
                rand_tail = np.random.choice(len(self.entities))
                n_triple[1] = rand_tail
        # 返回负样本三元组
        return n_triple

    # 评分函数
    def score(self, h_emb, t_emb, r_emb):
        if self.norm == 2:
            return np.sum(np.square(h_emb + r_emb - t_emb))
        else:
            return np.sum(np.fabs(h_emb + r_emb - t_emb))

    # 误差函数
    # 以正负三元组对儿的评分为基础
    # 理论上，正样本评分越高同时负样本评分越低，误差越小，但需要有边界[0, margin]约束
    # 即正样本评分与误差反相关，负样本评分与误差正相关
    def hinge_loss(self, p_score, n_score):
        return max(0, p_score - n_score + self.margin)

    # 更新向量
    # 以批量的正负三元组样本对儿为单位进行更新
    def update_embeddings(self, batch_pn_triple_pair):
        # 对更新之前的向量进行深拷贝，避免一边计算一边更新
        entity_embs_copy = copy.deepcopy(self.entities)
        rel_embs_copy = copy.deepcopy(self.rels)
        # 遍历batch，每个正负三元组样本对儿更新一次
        for p_triple, n_triple in batch_pn_triple_pair:
            # 取更新之前的向量进行计算
            # 正样本
            p_h_emb = self.entities[p_triple[0]]
            p_t_emb = self.entities[p_triple[1]]
            p_r_emb = self.rels[p_triple[2]]
            # 负样本
            n_h_emb = self.entities[n_triple[0]]
            n_t_emb = self.entities[n_triple[1]]
            n_r_emb = self.rels[n_triple[2]]

            # 使用更新之前的向量计算评分
            p_score = self.score(p_h_emb, p_t_emb, p_r_emb)
            n_score = self.score(n_h_emb, n_t_emb, n_r_emb)
            # 根据评分计算当前正负样本对儿的误差
            cur_loss = self.hinge_loss(p_score, n_score)
            # 根据hinge函数可知，当前误差大于等于0，等于0不能求导
            if cur_loss > 0:
                # 将当前误差加入总误差
                self.loss += cur_loss
                # 计算梯度，包含两层函数：评分函数和hinge函数，hinge函数求导为正负1，因此主要是对评分函数
                # norm = 2时
                p_grad = 2 * (p_h_emb + p_r_emb - p_t_emb)
                n_grad = 2 * (n_h_emb + n_r_emb - n_t_emb)
                # norm = 1时，在L2范数求导的基础上逐个元素判断正负，为正赋值1，为负则赋值-1
                if self.norm == 1:
                    for i in np.arange(len(p_grad)):
                        if p_grad[i] > 0:
                            p_grad[i] = 1
                        else:
                            p_grad[i] = -1
                    for i in np.arange(len(n_grad)):
                        if n_grad[i] > 0:
                            n_grad[i] = 1
                        else:
                            n_grad[i] = -1
                # 梯度下降
                # 个人觉得原始版本的代码中此处写的复杂难以理解，因此改为以下方式
                # 无须判断替换的是头实体还是尾实体，根据正样本和负样本各更新一次即可
                # 基于正样本三元组的梯度对拷贝向量进行更新
                entity_embs_copy[p_triple[0]] -= self.lr * p_grad
                entity_embs_copy[p_triple[1]] -= (-1) * self.lr * p_grad
                rel_embs_copy[p_triple[2]] -= self.lr * p_grad
                # 基于负样本三元组的梯度对拷贝向量进行更新
                entity_embs_copy[n_triple[0]] -= (-1) * self.lr * n_grad
                entity_embs_copy[n_triple[1]] -= self.lr * n_grad
                rel_embs_copy[n_triple[2]] -= (-1) * self.lr * n_grad
        # 更新完一个batch后对所有向量再进行一次归一化
        for key in entity_embs_copy.keys():
            entity_embs_copy[key] /= np.linalg.norm(entity_embs_copy[key], ord=self.norm)
        for key in rel_embs_copy.keys():
            rel_embs_copy[key] /= np.linalg.norm(rel_embs_copy[key], ord=self.norm)
        # 实现更新
        self.entities = entity_embs_copy
        self.rels = rel_embs_copy

    # 训练函数
    def train(self, epochs):
        # 将所有三元组训练集划分为400个batch
        nbatches = 400
        # batchsize通过计算得出
        batch_size = len(self.triple_list) // nbatches
        # 打印batchsize
        print('batch size:', batch_size)
        for epoch in np.arange(epochs):
            start_time = time.time()
            # 每个epoch将loss归零
            self.loss = 0.0
            for batch in np.arange(nbatches):
                batch_pn_triple_pair = []
                # 从0~N-1中随机抽取出batch_size个整数
                random_index = np.random.choice(len(self.triple_list), size=batch_size, replace=False)
                # 以随机整数为下标重构batch_size大小的正样本batch
                p_batch = [self.triple_list[i] for i in random_index]
                # 对每个正样本三元组进行负采样，构成批量的正负样本三元组对儿
                for p_triple in p_batch:
                    n_triple = self.negative_sample(p_triple)
                    batch_pn_triple_pair.append([p_triple, n_triple])
                # 根据当前的批量正负样本三元组对儿更新向量
                self.update_embeddings(batch_pn_triple_pair)
                # 计算平均误差
                self.loss = self.loss / (batch + 1)
            end_time = time.time()
            # 每个epoch结束后打印信息
            print('epoch:', epoch, 'cost time:', round(end_time - start_time, 3))
            print('loss:', self.loss)

            # 保存当前epoch的误差
            with codecs.open('../res/epoch_loss', 'a') as loss_f:
                loss_f.write('epoch: %d\tloss: %s\n' % (epoch, self.loss))
            # 每10个epoch保存一次实体和关系的嵌入结果
            if epoch % 10 == 0:
                with codecs.open('../res/e_embs_per_10_epoch', 'w') as e_emb_f:
                    for e in self.entities.keys():
                        e_emb_f.write(str(e) + '\t' + str(list(self.entities[e])) + '\n')
                with codecs.open('../res/r_embs_per_10_epoch', 'w') as r_emb_f:
                    for r in self.rels.keys():
                        r_emb_f.write(str(r) + '\t' + str(list(self.rels[r])) + '\n')
        # 全部训练结束后保存最终嵌入结果
        print('写入文件…………')
        with codecs.open('../res/entity_dim' + str(self.embed_dim) + '_batch400', 'w') as f1:
            for e in self.entities.keys():
                f1.write(str(e) + '\t' + str(list(self.entities[e])) + '\n')
        with codecs.open('../res/relation_dim' + str(self.embed_dim) + '_batch400', 'w') as f2:
            for r in self.rels.keys():
                f2.write(str(r) + '\t' + str(list(self.rels[r])) + '\n')
        print('写入完成！')






































































