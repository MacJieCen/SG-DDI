import numpy as np
from scipy import linalg
from torch_geometric.loader import DataLoader
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import warnings
from time import time

import pandas as pd
import torch

from configs import get_cfg_defaults
from dataloader import DTIDataset
from models import DeeperGCN
from trainer import Trainer
from utils import set_seed, mkdir
from transformers import AutoModel, AutoTokenizer

model_path = "biobert-v1.1"

model = AutoModel.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)

embedding_size = 256
drug_num = 1706

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DrugBAN for DDI prediction")
parser.add_argument('--cfg', default="configs/SG-DDI.yaml", help="path to config file", type=str)
parser.add_argument('--data', default="D1", type=str, metavar='TASK',
                    help='dataset')
# parser.add_argument('--split', default='random', type=str, metavar='S', help="split task",
#                     choices=['random', 'cold', 'cluster'])
args = parser.parse_args()


def compute_kernel_bias(vecs, n_components=embedding_size):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


def description_process(description_dataset):
    encoded_inputs = tokenizer(description_dataset,
                               return_tensors="pt",
                               padding="max_length",
                               truncation=True,
                               max_length=32)
    output = model(**encoded_inputs)
    text_embeddings = output[0]
    tmp_embeddings = np.zeros((len(text_embeddings[0]), len(text_embeddings[0][0])), dtype = float)
    text_embeddings = text_embeddings.detach().numpy()
    void_text = 'not_have'

    indices = [index for index, value in enumerate(description_dataset) if value == void_text]
    for index in indices:
        text_embeddings[index] = tmp_embeddings

    return text_embeddings

def label_process(label_dataset):
    encoded_inputs = tokenizer(label_dataset,
                               return_tensors="pt",
                               padding="max_length",
                               truncation=True,
                               max_length=128)
    output = model(**encoded_inputs)
    text_embeddings = output[0][:, 0, :]

    text_embeddings = text_embeddings.detach().numpy()

    v_data = np.array(text_embeddings)
    kernel, bias = compute_kernel_bias(v_data, embedding_size)
    text_embeddings = transform_and_normalize(v_data, kernel=kernel, bias=bias)
    return text_embeddings

def label_sentence_process(label_dataset):
    encoded_inputs = tokenizer(label_dataset,
                               return_tensors="pt",
                               padding="max_length",
                               truncation=True,
                               max_length=32)
    output = model(**encoded_inputs)
    text_embeddings = output[0][:, 0, :]

    text_embeddings = text_embeddings.detach().numpy()

    v_data = np.array(text_embeddings)
    kernel, bias = compute_kernel_bias(v_data, 32)
    text_embeddings = transform_and_normalize(v_data, kernel=kernel, bias=bias)
    return text_embeddings

def label_explanation_process(label_dataset):
    encoded_inputs = tokenizer(label_dataset,
                               return_tensors="pt",
                               padding="max_length",
                               truncation=True,
                               max_length=64)
    output = model(**encoded_inputs)
    text_embeddings = output[0][:, 0, :]

    text_embeddings = text_embeddings.detach().numpy()


    v_data = np.array(text_embeddings)
    kernel, bias = compute_kernel_bias(v_data, 64)
    text_embeddings = transform_and_normalize(v_data, kernel=kernel, bias=bias)
    return text_embeddings

def process_smiles(df):
    fc = partial(smiles_to_bigraph, add_self_loop=True)
    max_drug_nodes = 460
    drug_smiles = []
    drug_edges = []
    for i in range(len(df['Smiles'])):
        v_d = df['Smiles'][i]
        v_d = fc(smiles=v_d, node_featurizer=CanonicalAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer(self_loop=True))
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        src, dst = v_d.edges()
        v_d_edge_index = torch.stack([torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long)], dim=0)
        v_d = v_d.ndata['h']
        drug_smiles.append(v_d)
        drug_edges.append(v_d_edge_index)

    return drug_smiles, drug_edges


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'

    train_path = os.path.join(dataFolder, "train.csv")
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    label_path = os.path.join(dataFolder, "label.csv")
    bio_path = os.path.join(dataFolder, "Bio_info.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_label = pd.read_csv(label_path)
    df_bio = pd.read_csv(bio_path, encoding='ISO-8859-1')
    b = 0.0005
    label_num = df_train['y'].value_counts().sort_index().to_numpy()
    mean_label_num = np.mean(label_num)
    label_coefficient = np.tanh(b * (label_num - mean_label_num))

    description_dataset = df_bio['Description'].to_numpy()
    description_text = []
    for i in range(len(description_dataset)):
        description_text.append(description_dataset[i])
    description_text = description_process(description_text)
    description_text = torch.tensor(description_text)
    smiles_graph, edge_graph = process_smiles(df_bio)

    train_dataset = DTIDataset(df_train.index.values, df_train, description_text, smiles_graph, edge_graph)
    val_dataset = DTIDataset(df_val.index.values, df_val, description_text, smiles_graph, edge_graph)
    test_dataset = DTIDataset(df_test.index.values, df_test, description_text, smiles_graph, edge_graph)

    label_dataset = df_label['interaction'].to_numpy()
    label_sentence = df_label['sentence'].to_numpy()
    label_explanation = df_label['explanation'].to_numpy()

    label_text = []

    for i in range(len(label_dataset)):
        label_text.append(label_dataset[i])
    label_text = label_process(label_text)
    label_text = torch.tensor(label_text)

    label_text1 = []
    for i in range(len(label_sentence)):
        label_text1.append(label_sentence[i])
    label_text1 = label_sentence_process(label_text1)
    label_text1 = torch.tensor(label_text1)

    label_text2 = []
    for i in range(len(label_explanation)):
        label_text2.append(label_explanation[i])
    label_text2 = label_explanation_process(label_text2)
    label_text2 = torch.tensor(label_text2)

    label_text = torch.cat((label_text, label_text1, label_text2), 1)



    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'pin_memory': False}

    training_generator = DataLoader(train_dataset, **params)
        # params['shuffle'] = False
        # params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)



    # device_ids = [0, 1]
    model = DeeperGCN(**cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, label_coefficient, label_text, training_generator, val_generator,
                      test_generator, discriminator=None, experiment=experiment, **cfg)

    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
