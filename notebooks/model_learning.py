# pip install pandas, numpy, tqdm, networkx
# pip intsall torch, torch-geometric
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import networkx as nx
import pickle
from tqdm import tqdm, trange
from torch_geometric.utils import from_networkx


class GATRegression(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GATRegression, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x
    
def create_dataset(g: nx.Graph, graph, node_labels, target_function):
    x = torch.tensor(node_labels, dtype=torch.float)
    edge_index = graph.edge_index
    y = torch.tensor(list(target_function(g).values()), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

def train(epochs, data, model, optimizer, loss_fn, train_mask, test_mask):
    loss_items = []
    for epoch in trange(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index).squeeze()
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        loss_items.append(loss.item())
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index).squeeze()
        test_loss = loss_fn(predictions[test_mask], data.y[test_mask])
        return test_loss.item(), loss_items

class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, heads=1)
        
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class TemporalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GraphTemporalPipeline(nn.Module):
    def __init__(self, gat_encoder, temporal_model):
        super().__init__()
        self.gat_encoder = gat_encoder
        self.temporal_model = temporal_model
        
    def forward(self, graphs):
        embeddings = []
        for graph in graphs:
            emb = self.gat_encoder(graph.x, graph.edge_index)
            emb = emb.mean(dim=0)
            embeddings.append(emb)
        
        # [num_years, num_nodes, embedding_dim]
        temporal_input = torch.stack(embeddings, dim=0).unsqueeze(0)
        
        return self.temporal_model(temporal_input)

def load_graph(year, graph, encodings, target_func):
    filtered_nodes = [n for n, attr in graph.nodes(data=True) if 0 < attr.get('year') <= year]
    filtered_edges = [(u, v) for u,v in graph.edges if u in filtered_nodes and v in filtered_nodes]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(graph.nodes)
    subgraph.add_edges_from(filtered_edges)
    # subgraph = graph.subgraph(filtered_nodes)
    pyg_data = from_networkx(subgraph)
    x = torch.tensor(np.array(encodings), dtype=torch.float)
    edge_indices = pyg_data.edge_index
    y = torch.tensor(list(target_func(subgraph).values()), dtype=torch.float)
    return Data(x=x, edge_index=edge_indices, y=y)

def train_lstm(epochs, seqs, model, optimizer, criterion):
    loss_items = []
    for _ in trange(epochs):
        for sequence, target in seqs:
            optimizer.zero_grad()
            preds = model(sequence)
            loss = criterion(preds, target.y.unsqueeze(0))
            loss.backward()
            optimizer.step()
            loss_items.append(loss.item())
    return loss_items

if __name__ == "__main__":
    graph = pickle.load(open("../data/graph_medium.pickle", "rb"))
    encoded_data = np.genfromtxt("labels2.csv", delimiter=',')
    encoded_data_no_year = np.genfromtxt("labels_no_year2.csv", delimiter=',')
    years = np.genfromtxt("years2.csv", delimiter=',')
    
    assert encoded_data.shape[0] == len(years)
    
    node_list = list(graph.nodes())
    node_mapping = {node: idx for idx, node in enumerate(node_list)}
    graph = nx.relabel_nodes(graph, node_mapping)
    years_dict = dict(enumerate(years))
    nx.set_node_attributes(graph, years_dict, name="year")
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Если запускается на gpu
    device = torch.device('cpu') # Если не запускается на gpu
    

    # Первая модель
    pyg_data = from_networkx(graph.to_undirected())
    
    dataset_pr = create_dataset(graph, pyg_data, encoded_data, nx.pagerank).to(device)
    dataset_harmonic = create_dataset(graph, pyg_data, encoded_data, nx.harmonic_centrality).to(device)

    test_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('year') == 2016]
    train_nodes = list(set(list(range(len(graph.nodes)))).difference(set(test_nodes)))
  
    train_mask = torch.zeros(len(graph.nodes), dtype=torch.bool).to(device)
    train_mask[train_nodes] = True
    test_mask = torch.zeros(len(graph.nodes), dtype=torch.bool).to(device)
    test_mask[test_nodes] = True

    print("Model #1, Pagerank")
    model = GATRegression(in_channels=1538, hidden_channels=2048, out_channels=1).to(device) # Если не запустится, hidden_channels -> 2048
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Если будет быстро, но плохой результат, то можно lr=0.001
    loss_fn = torch.nn.MSELoss()
    val, loss_items = train(1, dataset_pr, model, optimizer, loss_fn, train_mask, test_mask) # Первое число - количество эпох. Можно увеличить в сотнях, если будет летать
    std_pr = torch.std(dataset_pr.y[test_mask])
    with open("results.txt", "w") as f:
        f.write(f"Last 5 loss items:\n")
        for item in loss_items[-5:]:
            f.write(f"{item}\n")
        f.write(f"Test Loss = {val}\n")
        f.write(f"Standard deviation = {std_pr}\n")
        f.write("-"*30 + "\n")
    print("Done!")

    print("Model #1, Harmonic")
    model = GATRegression(in_channels=1538, hidden_channels=2048, out_channels=1).to(device) # Если не запустится, hidden_channels -> 2048
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Как выше
    loss_fn = torch.nn.MSELoss()
    val, loss_items = train(1, dataset_harmonic, model, optimizer, loss_fn, train_mask, test_mask)
    std_harmonic = torch.std(dataset_harmonic.y[test_mask])
    with open("results.txt", "a") as f:
        f.write(f"Last 5 loss items:\n")
        for item in loss_items[-5:]:
            f.write(f"{item} ")
        f.write(f"Test Loss = {val}\n")
        f.write(f"Standard deviation = {std_harmonic}\n")
        f.write("-"*30 + "\n")
    print("Done!")
        
    # Вторая модель. Может имеет смысл закоментить первую модель, чтобы их запускать отдельно, чтобы память 
    num_nodes = encoded_data_no_year.shape[0]
    num_features = encoded_data_no_year.shape[1]
    hidden_dim = 2048
    num_heads = 1
    years_range = list(range(2001, 2015))
    
    graphs = [load_graph(year, graph, encoded_data_no_year, nx.pagerank).to(device) for year in years_range]
    sequences = []
    for i in range(len(graphs) - 1):
        sequence = graphs[:i+1]
        target = graphs[i+1]
        sequences.append((sequence, target))
    
    graphs_harmonic = [load_graph(year, graph, encoded_data_no_year, nx.harmonic_centrality).to(device) for year in years_range]
    sequences_harmonic = []
    for i in range(len(graphs_harmonic) - 1):
        sequence = graphs_harmonic[:i+1]
        target = graphs_harmonic[i+1]
        sequences_harmonic.append((sequence, target))

    print("Model #2, PageRank")
    gat = GATEncoder(num_features, hidden_dim, num_heads).to(device)
    temporal = TemporalModel(hidden_dim, hidden_dim, num_nodes).to(device)
    pipeline = GraphTemporalPipeline(gat, temporal).to(device)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    loss_items = train_lstm(60, sequences, pipeline, optimizer, criterion) # Аналогично первое число - эпохи

    graphs.append(load_graph(2016, graph, encoded_data_no_year, nx.pagerank).to(device))
    sequences.append([graphs[:-1], graphs[-1]])
    pipeline.eval()
    with torch.no_grad():
        preds = pipeline(sequences[-1][0])
        loss = criterion(preds, sequences[-1][1].y.unsqueeze(0))
    std_pr2 = torch.std(sequences[-1][1].y)
    with open("results.txt", "a") as f:
        f.write(f"Last 5 loss items:\n")
        for item in loss_items[-5:]:
            f.write(f"{item}\n")
        f.write(f"Test Loss = {val}\n")
        f.write(f"Standard deviation = {std_pr2}\n")
        f.write("-"*30 + "\n")
    print("Done!")

    print("Model #2, Harmonic")
    gat = GATEncoder(num_features, hidden_dim, num_heads).to(device)
    temporal = TemporalModel(hidden_dim, hidden_dim, num_nodes).to(device)
    pipeline = GraphTemporalPipeline(gat, temporal).to(device)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    loss_items = train_lstm(200, sequences_harmonic, pipeline, optimizer, criterion)

    graphs_harmonic.append(load_graph(2016, graph, encoded_data_no_year, nx.harmonic_centrality).to(device))
    sequences_harmonic.append([graphs_harmonic[:-1], graphs_harmonic[-1]])
    pipeline.eval()
    with torch.no_grad():
        preds = pipeline(sequences_harmonic[-1][0])
        loss = criterion(preds, sequences_harmonic[-1][1].y.unsqueeze(0))
        print(f"Loss on test = {loss.item()}")
    std_harmonic2 = torch.std(sequences_harmonic[-1][1].y)
    with open("results.txt", "a") as f:
        f.write(f"Last 5 loss items:\n")
        for item in loss_items[-5:]:
            f.write(f"{item}\n")
        f.write(f"Test Loss = {val}\n")
        f.write(f"Standard deviation = {std_harmonic2}\n")
        f.write("-"*30 + "\n")
    print("Done!")
