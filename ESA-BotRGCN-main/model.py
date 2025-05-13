import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    RGCNConv, FastRGCNConv, GCNConv, GATConv, RGATConv
)



class ESAFastBotRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1, embedding_dimension=128, dropout=0.3):
        super(ESAFastBotRGCN, self).__init__()
        self.dropout = dropout

        # 将每一类特征线性压缩为 embedding_dimension/5 维度，共五类（des, tweet, num_prop, cat_prop, new_feature）
        per_feature_dim = embedding_dimension // 5

        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, per_feature_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 28),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, per_feature_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, per_feature_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(
            nn.Linear(new_feature_size, per_feature_dim),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.rgcn1 = FastRGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        self.rgcn2 = FastRGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)

        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)

        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)

        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x






class ESABotRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=2, dropout=0.3):
        super(ESABotRGCN, self).__init__()
        self.dropout = dropout

        # 👉 各自压缩维度按你的需求设置
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, 25),  # 32维度
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 28),  # 52维度
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, 25),  # 12维度
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, 25),  # 20维度
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(  # 新增的特征 12维
            nn.Linear(new_feature_size, 25),  # 12维度
            nn.LeakyReLU()
        )
        
        # 👇 总维度改成 128（32 + 52 + 12 + 20 + 12）
        self.linear_relu_input = nn.Sequential(
            nn.Linear(128, 128),  # 重新调整总输入维度为128
            nn.LeakyReLU()
        )

        self.rgcn = RGCNConv(128, 128, num_relations=2)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(128, 2)

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)  # 处理新特征

        # 拼接时加入新特征，注意新的特征维度
        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


class ESACustomBotRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1,
                 des_dim=28, tweet_dim=36, num_dim=12, cat_dim=40, new_dim=12, dropout=0.3):
        super(ESACustomBotRGCN, self).__init__()
        self.dropout = dropout

        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, des_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, tweet_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, num_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, cat_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(
            nn.Linear(new_feature_size, new_dim),
            nn.LeakyReLU()
        )

        total_dim = des_dim + tweet_dim + num_dim + cat_dim + new_dim
        self.linear_relu_input = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.LeakyReLU()
        )

        self.rgcn1 = RGCNConv(total_dim, total_dim, num_relations=2)
        self.rgcn2 = RGCNConv(total_dim, total_dim, num_relations=2)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(total_dim, 2)

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)

        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x







class ESAFeatureAttention(nn.Module):
    def __init__(self, num_features):
        super(ESAFeatureAttention, self).__init__()
        self.attn_weights = nn.Parameter(torch.Tensor(num_features, 1))
        nn.init.xavier_uniform_(self.attn_weights)

    # def forward(self, features):
    #     # features: list of [batch, dim] tensors
    #     stacked = torch.stack(features, dim=0)  # [num_features, batch, dim]
    #     weights = torch.softmax(self.attn_weights, dim=0)  # [num_features, 1]
    #     weighted = weights * stacked
    #     fused = torch.sum(weighted, dim=0)  # [batch, dim]
    #     return fused
    def forward(self, features):
        # features: list of [batch_size, dim] tensors
        stacked = torch.stack(features, dim=0)  # [num_features, batch_size, dim]
        weights = torch.softmax(self.attn_weights, dim=0).unsqueeze(1)  # [num_features, 1, 1]
        weighted = weights * stacked  # ✅ 广播成功
        fused = torch.sum(weighted, dim=0)  # [batch_size, dim]
        return fused

    def get_attention_weights(self):
        return torch.softmax(self.attn_weights.data, dim=0).squeeze()


class ESABotRGCNWithAttention(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=3, new_feature_size=1,
                 common_dim=64, embedding_dimension=128, dropout=0.3):
        super(ESABotRGCNWithAttention, self).__init__()
        self.dropout = dropout

        # 特征编码（先压缩成统一维度 common_dim）
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, common_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, common_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, common_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, common_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(
            nn.Linear(new_feature_size, common_dim),
            nn.LeakyReLU()
        )

        # 注意力融合层
        self.feature_attention = ESAFeatureAttention(num_features=5)

        # 图神经网络部分
        self.linear_relu_input = nn.Sequential(
            nn.Linear(common_dim, embedding_dimension),
            nn.LeakyReLU()
        )
        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        # 各类特征压缩
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)

        # 注意力融合
        fused = self.feature_attention([d, t, n, c, nf])  # [batch, common_dim]

        # GNN 后续处理
        x = self.linear_relu_input(fused)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x

    def print_attention_weights(self):
        weights = self.feature_attention.get_attention_weights()
        print("\n🧠 Attention Weights (Feature Importance):")
        for name, w in zip(["Description", "Tweets", "NumProp", "CatProp", "NewFeature"], weights):
            print(f"  {name}: {w.item():.4f}")




class ESABotNoGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1, dropout=0.3):
        super(ESABotNoGCN, self).__init__()
        self.dropout = dropout

        # 👉 各自压缩维度按你的需求设置
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, 28),  # 32维度
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 36),  # 52维度
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, 12),  # 12维度
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, 40),  # 20维度
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(  # 新增的特征 12维
            nn.Linear(new_feature_size, 12),  # 12维度
            nn.LeakyReLU()
        )
        
        # 👇 总维度改成 128（32 + 52 + 12 + 20 + 12）
        self.linear_relu_input = nn.Sequential(
            nn.Linear(128, 128),  # 重新调整总输入维度为128
            nn.LeakyReLU()
        )

        # noGCN 部分：我们用一个全连接层来代替RGCN
        self.fc1 = nn.Linear(128, 128)  # 用全连接层代替图卷积层
        self.fc2 = nn.Linear(128, 128)  # 第二个全连接层
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(128, 2)

    def forward(self, des, tweet, num_prop, cat_prop, new_feature):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)  # 处理新特征

        # 拼接时加入新特征，注意新的特征维度
        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)

        # 用全连接层替代RGCN层
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x









import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # 引入 GATConv 层
class ESABotGAT(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1, dropout=0.3, num_heads=4):
        super(ESABotGAT, self).__init__()
        self.dropout = dropout

        # 👉 各自压缩维度按你的需求设置
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, 28),  # 32维度
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 36),  # 52维度
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, 12),  # 12维度
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, 40),  # 20维度
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(  # 新增的特征 12维
            nn.Linear(new_feature_size, 12),  # 12维度
            nn.LeakyReLU()
        )

        # 👇 总维度改成 128（32 + 52 + 12 + 20 + 12）
        self.linear_relu_input = nn.Sequential(
            nn.Linear(128, 128),  # 重新调整总输入维度为128
            nn.LeakyReLU()
        )

        # 使用 GATConv 替代 RGCNConv
        self.gat1 = GATConv(128, 128, heads=num_heads, dropout=self.dropout)  # 注意这里
        self.gat2 = GATConv(128 * num_heads, 128, dropout=self.dropout)  # 修改为 num_heads * 128

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(128, 2)

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)  # 处理新特征

        # 拼接时加入新特征，注意新的特征维度
        x = torch.cat((d, t, n, c, nf), dim=1)  # 拼接操作
        x = self.linear_relu_input(x)

        # 使用 GATConv 进行信息聚合
        x = self.gat1(x, edge_index)  # 第一层 GAT
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)  # 第二层 GAT

        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x










class ESABotGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1, embedding_dimension=128, dropout=0.3):
        super(ESABotGCN, self).__init__()
        self.dropout = dropout

        # 各自压缩维度按您的要求设置
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, 28),  # 将 description 特征压缩到 28 维
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 36),  # 将 tweet 特征压缩到 36 维
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, 12),  # 将 num_prop 特征压缩到 12 维
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, 40),  # 将 cat_prop 特征压缩到 40 维
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(  # 将新特征压缩到 12 维
            nn.Linear(new_feature_size, 12),  # 12 维度
            nn.LeakyReLU()
        )

        # 拼接特征后输入的维度为 128
        self.linear_relu_input = nn.Sequential(
            nn.Linear(128, 128),  # 将拼接后的特征维度压缩为 128 维
            nn.LeakyReLU()
        )

        # GCN 层
        self.gcn1 = GCNConv(128, 128)  # 使用 GCNConv 代替 RGCNConv
        self.gcn2 = GCNConv(128, 128)  # 使用 GCNConv 代替 RGCNConv

        # 输出层
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(128, 2)  # 二分类输出

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        # 处理各个输入特征
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)  # 处理新特征

        # 拼接所有特征
        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)

        # 第一个 GCN 层
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二个 GCN 层
        x = self.gcn2(x, edge_index)

        # 输出层
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x

    


class ESABotRGAT(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1,
                 des_dim=28, tweet_dim=36, num_dim=12, cat_dim=40, new_dim=12, 
                 embedding_dimension=128, num_relations=2, num_heads=4, dropout=0.3):
        super(ESABotRGAT, self).__init__()
        self.dropout = dropout
        self.num_heads = num_heads

        # 输入特征的线性变换（可自定义压缩维度）
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, des_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, tweet_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, num_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, cat_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(
            nn.Linear(new_feature_size, new_dim),
            nn.LeakyReLU()
        )

        # 总特征拼接后的维度
        total_dim = des_dim + tweet_dim + num_dim + cat_dim + new_dim

        self.linear_relu_input = nn.Sequential(
            nn.Linear(total_dim, embedding_dimension),
            nn.LeakyReLU()
        )

        # R-GAT 图卷积层
        self.rgat1 = RGATConv(
            embedding_dimension,
            embedding_dimension // num_heads,
            heads=num_heads,
            num_relations=num_relations
        )
        self.rgat2 = RGATConv(
            embedding_dimension,
            embedding_dimension,
            heads=1,
            num_relations=num_relations
        )

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)

        # 特征拼接
        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)

        # 图卷积传播
        x = self.rgat1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgat2(x, edge_index, edge_type)

        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class ESABotMLP(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1, dropout=0.3):
        super(ESABotMLP, self).__init__()
        self.dropout = dropout

        # 👉 各自压缩维度按你的需求设置
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, 28),  # 32维度
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 36),  # 52维度
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, 12),  # 12维度
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, 40),  # 20维度
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(  # 新增的特征 12维
            nn.Linear(new_feature_size, 12),  # 12维度
            nn.LeakyReLU()
        )

        # 👇 总维度改成 128（32 + 52 + 12 + 20 + 12）
        self.linear_relu_input = nn.Sequential(
            nn.Linear(128, 128),  # 重新调整总输入维度为128
            nn.LeakyReLU()
        )

        # 多层感知机（MLP）部分
        self.mlp1 = nn.Linear(128, 128)  # 第一层全连接层
        self.mlp2 = nn.Linear(128, 128)  # 第二层全连接层
        self.mlp3 = nn.Linear(128, 2)    # 输出层，进行二分类

    def forward(self, des, tweet, num_prop, cat_prop, new_feature):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)  # 处理新特征

        # 拼接时加入新特征，注意新的特征维度
        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)

        # 通过 MLP 层进行前向传播
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout 层
        x = F.relu(self.mlp1(x))  # 第一层全连接
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout 层
        x = F.relu(self.mlp2(x))  # 第二层全连接
        x = self.mlp3(x)  # 输出层

        return x


class ESABotRGCN_4layers(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1, dropout=0.3):
        super(ESABotRGCN_4layers, self).__init__()
        self.dropout = dropout

        # 各自压缩维度按你的需求设置
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, 28),  # 32维度
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 36),  # 52维度
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, 12),  # 12维度
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, 40),  # 20维度
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(  # 新增的特征 12维
            nn.Linear(new_feature_size, 12),  # 12维度
            nn.LeakyReLU()
        )

        # 总维度改成 128（32 + 52 + 12 + 20 + 12）
        self.linear_relu_input = nn.Sequential(
            nn.Linear(128, 128),  # 重新调整总输入维度为128
            nn.LeakyReLU()
        )

        # 四个图卷积层
        self.rgcn1 = RGCNConv(128, 128, num_relations=2)
        self.rgcn2 = RGCNConv(128, 128, num_relations=2)
        self.rgcn3 = RGCNConv(128, 128, num_relations=2)
        self.rgcn4 = RGCNConv(128, 128, num_relations=2)

        # 输出层
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(128, 2)  # 二分类

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)  # 处理新特征

        # 拼接时加入新特征，注意新的特征维度
        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)

        # 四层图卷积
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn3(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn4(x, edge_index, edge_type)
        
        # 输出处理
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x

class ESABotRGCN_3layers(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1, dropout=0.3):
        super(ESABotRGCN_3layers, self).__init__()
        self.dropout = dropout

        # 各自压缩维度按你的需求设置
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, 28),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 36),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, 12),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, 40),
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(
            nn.Linear(new_feature_size, 12),
            nn.LeakyReLU()
        )

        # 总维度改成 128（32 + 52 + 12 + 20 + 12）
        self.linear_relu_input = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )

        # 三个图卷积层
        self.rgcn1 = RGCNConv(128, 128, num_relations=2)
        self.rgcn2 = RGCNConv(128, 128, num_relations=2)
        self.rgcn3 = RGCNConv(128, 128, num_relations=2)

        # 输出层
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(128, 2)  # 二分类

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)

        # 拼接时加入新特征，注意新的特征维度
        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)

        # 三层图卷积
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn3(x, edge_index, edge_type)
        
        # 输出处理
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x

class ESABotRGCN_1layer(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1, dropout=0.3):
        super(ESABotRGCN_1layer, self).__init__()
        self.dropout = dropout

        # 各自压缩维度按你的需求设置
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, 28),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 36),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, 12),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, 40),
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(
            nn.Linear(new_feature_size, 12),
            nn.LeakyReLU()
        )

        # 总维度改成 128（32 + 52 + 12 + 20 + 12）
        self.linear_relu_input = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )

        # 单层图卷积
        self.rgcn1 = RGCNConv(128, 128, num_relations=2)

        # 输出层
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(128, 2)  # 二分类

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)

        # 拼接时加入新特征，注意新的特征维度
        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)

        # 单层图卷积
        x = self.rgcn1(x, edge_index, edge_type)

        # 输出处理
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class ESABotRGCN_5layers(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1, dropout=0.3):
        super(ESABotRGCN_5layers, self).__init__()
        self.dropout = dropout

        # 各自压缩维度按你的需求设置
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, 28),  # 32维度
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 36),  # 52维度
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, 12),  # 12维度
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, 40),  # 20维度
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(  # 新增的特征 12维
            nn.Linear(new_feature_size, 12),  # 12维度
            nn.LeakyReLU()
        )

        # 总维度改成 128（32 + 52 + 12 + 20 + 12）
        self.linear_relu_input = nn.Sequential(
            nn.Linear(128, 128),  # 重新调整总输入维度为128
            nn.LeakyReLU()
        )

        # 五个图卷积层
        self.rgcn1 = RGCNConv(128, 128, num_relations=2)
        self.rgcn2 = RGCNConv(128, 128, num_relations=2)
        self.rgcn3 = RGCNConv(128, 128, num_relations=2)
        self.rgcn4 = RGCNConv(128, 128, num_relations=2)
        self.rgcn5 = RGCNConv(128, 128, num_relations=2)  # 第五层

        # 输出层
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(128, 2)  # 二分类

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)  # 处理新特征

        # 拼接时加入新特征，注意新的特征维度
        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)

        # 五层图卷积
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn3(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn4(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn5(x, edge_index, edge_type)

        # 输出处理
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x
class ESABotRGCN_8layers(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=11, new_feature_size=1, dropout=0.3):
        super(ESABotRGCN_8layers, self).__init__()
        self.dropout = dropout

        # 各自压缩维度按你的需求设置
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, 28),  # 32维度
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, 36),  # 52维度
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, 12),  # 12维度
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, 40),  # 20维度
            nn.LeakyReLU()
        )
        self.linear_relu_new_feature = nn.Sequential(  # 新增的特征 12维
            nn.Linear(new_feature_size, 12),  # 12维度
            nn.LeakyReLU()
        )

        # 总维度改成 128（32 + 52 + 12 + 20 + 12）
        self.linear_relu_input = nn.Sequential(
            nn.Linear(128, 128),  # 重新调整总输入维度为128
            nn.LeakyReLU()
        )

        # 八个图卷积层
        self.rgcn1 = RGCNConv(128, 128, num_relations=2)
        self.rgcn2 = RGCNConv(128, 128, num_relations=2)
        self.rgcn3 = RGCNConv(128, 128, num_relations=2)
        self.rgcn4 = RGCNConv(128, 128, num_relations=2)
        self.rgcn5 = RGCNConv(128, 128, num_relations=2)
        self.rgcn6 = RGCNConv(128, 128, num_relations=2)
        self.rgcn7 = RGCNConv(128, 128, num_relations=2)
        self.rgcn8 = RGCNConv(128, 128, num_relations=2)  # 第八层

        # 输出层
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(128, 2)  # 二分类

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        nf = self.linear_relu_new_feature(new_feature)  # 处理新特征

        # 拼接时加入新特征，注意新的特征维度
        x = torch.cat((d, t, n, c, nf), dim=1)
        x = self.linear_relu_input(x)

        # 八层图卷积
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn3(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn4(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn5(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn6(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn7(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn8(x, edge_index, edge_type)

        # 输出处理
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x
