import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeRepresentationExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, num_relations):
        super(EdgeRepresentationExtractor, self).__init__()
        self.rgcn_layer = RGCNLayer(input_dim, output_dim, num_relations)
    
    def forward(self, graph, node_features):
        edge_representations = self.rgcn_layer(graph, node_features)
        
        edge_representations = [torch.mm(hi_lplus1, hj_lplus1.t()) for hi_lplus1, hj_lplus1 in edge_representations]
        
        return edge_representations


class RGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_relations):
        super(RGCNLayer, self).__init__()
        self.W_r = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_relations)])
        self.W0 = nn.Linear(input_dim, output_dim)
    
    def forward(self, graph, node_features):
        # 获取节点的邻居节点
        neighbors = graph.get_neighbors()  # 这里需要根据您的数据结构获取邻居节点
        
        updated_node_features = []
        for i, node_feature in enumerate(node_features):
            neighbor_sum = 0
            for relation in range(num_relations):
                relation_sum = 0
                for neighbor in neighbors[i]:
                    # 计算公式中的每个部分
                    relation_sum += self.W_r[relation](node_feature)  # W_r[relation] 对应公式中的 W^r_l
                neighbor_sum += relation_sum
        
            # 计算节点的新表示
            new_node_feature = F.relu(neighbor_sum + self.W0(node_feature))
            updated_node_features.append(new_node_feature)
        
        return updated_node_features

# 类型门
class TypeGate(nn.Module):
    def __init__(self, type_dim, edge_dim):
        super(TypeGate, self).__init__()
        self.fc = nn.Linear(type_dim + edge_dim, 1)
    
    def forward(self, type_representation, edge_representation, phi):
        combined_representations = torch.cat((type_representation, edge_representation), dim=1)
        delta_i = torch.sigmoid(self.fc(combined_representations))
        return delta_i


# 混合专家鉴别器
class MixedExpertDiscriminator(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(MixedExpertDiscriminator, self).__init__()
        self.expert_networks = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_experts)])
    
    def forward(self, edge_representation, delta):
        expert_outputs = [expert(edge_representation) for expert in self.expert_networks]
        weighted_expert_outputs = [delta[i] * expert_outputs[i] for i in range(len(self.expert_networks))]
        final_representation = sum(weighted_expert_outputs)
        return final_representation


# 完整的 MEDTI 模型
class MEDTI(nn.Module):
    def __init__(self, input_dim, num_relations, type_dim, num_experts):
        super(MEDTI, self).__init()
        self.edge_extractor = EdgeRepresentationExtractor(input_dim, input_dim, num_relations)
        self.type_gate = TypeGate(type_dim, input_dim)
        self.expert_discriminator = MixedExpertDiscriminator(input_dim, num_experts)
    
    def forward(self, graph, node_features, type_representation):
        edge_representation = self.edge_extractor(graph, node_features)
        delta = self.type_gate(type_representation, edge_representation)
        final_representation = self.expert_discriminator(edge_representation)
        return final_representation

# 创建 MEDTI 模型
input_dim =  128
num_relations =  2
type_dim =  128
num_experts =  2
model = MEDTI(input_dim, num_relations, type_dim, num_experts)


# 准备数据
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# 2. 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 在验证集上评估模型
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            outputs = model(inputs)
            val_loss += criterion(outputs, labels)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%')

print('Training completed.')