import math
from dgl.nn.pytorch import MaxPooling
from dgllife.model.gnn import GCN
from torch.nn import Linear, ReLU, LayerNorm
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.nn import DeepGCNLayer, GENConv, SAGPooling, global_max_pool, global_mean_pool
from ban import BANLayer
import hsic
import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_cross_entropy(pred_output, labels):
    labels = torch.squeeze(labels, 1)
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def contrastive_loss(image_embeddings, text_embeddings, labels, label_coefficient, alpha, temperature=0.07):

    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)

    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature

    coefficient = alpha * label_coefficient + 1
    coefficient = coefficient.to(torch.float)

    for i in range(len(logits)):
        logits[i, labels[i]] *= coefficient[labels[i]]
    loss = F.cross_entropy(logits, labels)

    return loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

class DeeperGCN(torch.nn.Module):
    def __init__(self, **config):

        super(DeeperGCN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]

        self.node_encoder = Linear(drug_in_feats, mlp_hidden_dim)
        self.readout = SAGPooling(1 * 512, min_score=-1)
        self.layers = torch.nn.ModuleList()
        for i in range(1, 12 + 1):
            conv = GENConv(mlp_hidden_dim, mlp_hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(mlp_hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.bcn = weight_norm(
            BANLayer(v_dim=512, q_dim=512, h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.text_bcn = weight_norm(
            BANLayer(v_dim=768, q_dim=768, h_dim=128, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(512, 256, 128, binary=out_binary)
        self.rel_emb = nn.Embedding(62, 64)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        self.fc = nn.Linear(512, 352)
        self.fc_1 = nn.Linear(320, 352)
        self.fc_2 = nn.Linear(128, 352)
        self.fc_4 = nn.Linear(192, 352)
        self.smile_fc = MLPDecoder(3412, 1024, 256, binary=64)

    def forward(self, graph, label, label_coefficient, alpha, v_d_target, v_d_smile, v_d_description, v_p_target, v_p_smile, v_p_description, label_text):

        h_data, t_data = graph
        label = label.long()
        rels = self.rel_emb(label).squeeze()

        drug_smile = torch.cat((v_d_smile, v_p_smile), dim=1)
        drug_smile = self.smile_fc(drug_smile.float())

        bio_description = self.text_bcn(v_d_description, v_p_description)


        v_d = self.node_encoder(h_data.x)
        v_d = self.layers[0].conv(v_d, h_data.edge_index)

        v_p = self.node_encoder(t_data.x)
        v_p = self.layers[0].conv(v_p, t_data.edge_index)

        for layer in self.layers[1:]:
            v_d = layer(v_d, h_data.edge_index)
        v_d = self.layers[0].act(self.layers[0].norm(v_d))
        v_d = F.dropout(v_d, p=0.1, training=self.training)

        for layer in self.layers[1:]:
            v_p = layer(v_p, t_data.edge_index)
        v_p = self.layers[0].act(self.layers[0].norm(v_p))
        v_p = F.dropout(v_p, p=0.1, training=self.training)

        label_text = label_text.float()

        v_d = v_d.reshape(len(h_data.ptr)-1, 460, 512)
        v_p = v_p.reshape(len(h_data.ptr)-1, 460, 512)

        bio = self.bcn(v_d, v_p)
        f = torch.cat((bio, bio_description, drug_smile, rels), dim=1)
        f_1 = torch.cat((bio, rels), dim=1)
        f_2 = torch.cat((drug_smile, rels), dim=1)
        f_4 = torch.cat((bio_description, rels), dim=1)

        label = label.squeeze()

        loss_bio1 = contrastive_loss(self.fc_1(f_1), label_text, label, label_coefficient, alpha)
        loss_bio2 = contrastive_loss(self.fc_2(f_2), label_text, label, label_coefficient, alpha)
        loss_bio4 = contrastive_loss(self.fc_4(f_4), label_text, label, label_coefficient, alpha)
        loss_sep = (loss_bio1 + loss_bio2 + loss_bio4) / 3
        loss_bios = contrastive_loss(self.fc(f), label_text, label, label_coefficient, alpha)
        loss_con = (loss_sep + loss_bios) / 2

        loss_hsic = hsic.HSIC(bio, drug_smile)
        f = self.mlp_classifier(f)
        loss_2 = loss_hsic + loss_con

        return f, loss_2

    def pool(self, h, batch):
        h_fp32 = h.float()
        h_max = global_max_pool(h_fp32, batch)
        h_mean = global_mean_pool(h_fp32, batch)
        h = torch.cat([h_max, h_mean], dim=-1).type_as(h)
        h = self.graph_pred_linear(h)
        return h

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
