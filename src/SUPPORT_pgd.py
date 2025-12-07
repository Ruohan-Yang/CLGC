import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, JumpingKnowledge
from sklearn import metrics
from tqdm import tqdm

class GCN(nn.Module):
    def __init__(self, feature_dims, out_dims, hidden_dims, num_layers=2):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(feature_dims, hidden_dims))
        self.bns.append(nn.BatchNorm1d(hidden_dims))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dims, hidden_dims))
            self.bns.append(nn.BatchNorm1d(hidden_dims))

        self.convs.append(GCNConv(hidden_dims, out_dims))

        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.relu(x)
        x = self.convs[-1](x, edge_index)
        return x


class JK_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, mode='cat'):
        super(JK_GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.jump = JumpingKnowledge(mode)

        if mode == 'cat':
            self.lin = nn.Linear(num_layers * hidden_channels, out_channels)
        else:
            self.lin = nn.Linear(hidden_channels, out_channels)

        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        xs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.relu(x)
            xs.append(x)

        x = self.jump(xs)
        x = self.lin(x)
        return x


class Model_Net(nn.Module):
    def __init__(self, embedding_dim, layer_number, gcn_data, gcn_type, gcn_layer):
        super(Model_Net, self).__init__()

        self.layer_number = layer_number
        self.node_dim = embedding_dim
        self.edge_dim = embedding_dim * 2
        self.gcn_data = gcn_data

        if gcn_type == 'JK_GCN':
            for i in range(self.layer_number):
                gcn = JK_GCN(in_channels=1, hidden_channels=64, out_channels=self.node_dim, num_layers=gcn_layer, mode='cat')
                setattr(self, 'gcn%i' % i, gcn)
        else:
            for i in range(self.layer_number):
                gcn = GCN(feature_dims=1, out_dims=self.node_dim, hidden_dims=64, num_layers=gcn_layer)
                setattr(self, 'gcn%i' % i, gcn)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.edge_dim, out_channels=self.edge_dim, kernel_size=7, padding=3),
            nn.ReLU())

        self.layer_classifier = nn.Linear(self.edge_dim, self.layer_number)
        self.link_classifier = nn.Linear(self.edge_dim, 2)



    def forward(self, now_layer, leftnode, rightnode):

        for i in range(self.layer_number):
            layer_embed = eval('self.gcn'+str(i))(self.gcn_data[i]).cuda()
            setattr(self, 'layer%i' % i, layer_embed)

        layer_names = ['self.layer'+str(i) for i in now_layer.cpu().numpy().tolist()]
        layer_specific = torch.Tensor().cuda()
        for (l, i, j) in zip(layer_names, leftnode, rightnode):
            temp = torch.cat((eval(l)[i], eval(l)[j]), dim=0).cuda()
            temp = torch.unsqueeze(temp, dim=0)
            layer_specific = torch.cat((layer_specific, temp), dim=0)

        # shared = self.cnn(layer_specific.permute(1, 0)).permute(1, 0)

        # discriminant_outs = self.layer_classifier(shared)

        prediction_outs = self.link_classifier(layer_specific)

        return prediction_outs #, discriminant_outs

    def metrics_eval(self, eval_data, mode='eval'):
        scores = []
        labels = []
        preds = []
        iter_data = tqdm(eval_data, desc="Predicting...") if mode == 'test' else eval_data
        for data in iter_data:
            network_labels, left_nodes, right_nodes, link_labels = data
            with torch.no_grad():
                network_labels = Variable(network_labels).cuda()
                left_nodes = Variable(left_nodes).cuda()
                right_nodes = Variable(right_nodes).cuda()
                link_labels = Variable(link_labels).cuda()
            # output, _ = self.forward(network_labels, left_nodes, right_nodes)
            output = self.forward(network_labels, left_nodes, right_nodes)
            output = F.softmax(output, dim=1)
            _, argmax = torch.max(output, 1)
            scores += list(output[:, 1].cpu().detach().numpy())
            labels += list(link_labels.cpu().detach().numpy())
            preds += list(argmax.cpu().detach().numpy())

        acc = metrics.accuracy_score(labels, preds)
        bal_acc = metrics.balanced_accuracy_score(labels, preds)
        pre = metrics.precision_score(labels, preds, average='weighted')
        f1 = metrics.f1_score(labels, preds, average='weighted')
        auc = metrics.roc_auc_score(labels, scores, average=None)

        ap = metrics.average_precision_score(labels, scores)
        precision, recall, _ = metrics.precision_recall_curve(labels, scores)
        aupr = metrics.auc(recall, precision)

        eval_results = {
            'acc': acc,
            'bal_acc': bal_acc,
            'precision': pre,
            'f1': f1,
            'auc': auc,
            'ap': ap,
            'aupr': aupr
        }
        return eval_results

