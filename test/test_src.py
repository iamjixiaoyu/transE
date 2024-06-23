from src.load_data import load_data
from src.transE_simple import TransE

if __name__ == '__main__':
    data_dir = '../data/FB15k'
    entity_set, relation_set, train_triple_list = load_data(data_dir, 'train')
    embed_dim = 50
    lr = 0.01
    margin = 1.0
    norm = 1
    epochs = 1
    model = TransE(entity_set, relation_set, train_triple_list, embed_dim, lr, margin, norm)
    model.init_embeddings()
    model.train(epochs)
