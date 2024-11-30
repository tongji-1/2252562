import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GlobalAttention
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np

# 1. 基础GNN模型
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, 
                 dropout=0.3, model_type='GCN'):
        super(GNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        
        # 初始化层列表
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层
        if model_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=8, concat=True))
        elif model_type == 'SAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            if model_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif model_type == 'GAT':
                self.convs.append(GATConv(hidden_dim * 8, hidden_dim, heads=8, concat=True))
            elif model_type == 'SAGE':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 最后一层
        if model_type == 'GCN':
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(hidden_dim * 8, output_dim, heads=1, concat=False))
        elif model_type == 'SAGE':
            self.convs.append(SAGEConv(hidden_dim, output_dim))
            
        # 全局注意力池化
        self.global_attention = GlobalAttention(
            nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        # MLP用于最终预测
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 图卷积层
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层卷积
        x = self.convs[-1](x, edge_index)
        
        # 全局池化
        x = self.global_attention(x, batch)
        
        # MLP预测
        out = self.mlp(x)
        
        return out

# 2. 自定义数据集类
class CustomGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

# 3. 图数据预处理函数
def preprocess_graph_data(nodes, edges, features, labels):
    """
    将原始数据转换为PyTorch Geometric的Data对象
    """
    data_list = []
    
    for i in range(len(nodes)):
        x = torch.FloatTensor(features[i])
        edge_index = torch.LongTensor(edges[i]).t().contiguous()
        y = torch.FloatTensor([labels[i]])
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list

# 4. 训练函数
def train_gnn(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        out = model(batch)
        loss = criterion(out, batch.y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(train_loader.dataset)

# 5. 验证函数
def validate_gnn(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(val_loader.dataset)

# 6. 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型参数
    input_dim = 64  # 节点特征维度
    hidden_dim = 128
    output_dim = 1  # 预测目标维度
    num_layers = 3
    dropout = 0.3
    model_type = 'GAT'  # 可选 'GCN', 'GAT', 'SAGE'
    
    # 初始化模型
    model = GNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        model_type=model_type
    ).to(device)
    
    # 示例：创建随机数据
    num_graphs = 100
    num_nodes = 20
    num_edges = 40
    
    # 生成随机图数据
    graphs = []
    labels = []
    
    for i in range(num_graphs):
        # 随机节点特征
        features = torch.randn(num_nodes, input_dim)
        
        # 随机边
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # 随机标签
        label = torch.randn(1)
        
        # 创建图数据对象
        data = Data(x=features, edge_index=edge_index, y=label)
        graphs.append(data)
        labels.append(label)
    
    # 划分数据集
    train_size = int(0.8 * num_graphs)
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    # 创建数据加载器
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练循环
    num_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_gnn(model, train_loader, optimizer, criterion, device)
        val_loss = validate_gnn(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gnn_model.pth')
    
    print('Training completed!')

# 7. 预测函数
def predict_graph(model, graph, device):
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        output = model(graph)
        return output.cpu().numpy()

if __name__ == '__main__':
    main()
