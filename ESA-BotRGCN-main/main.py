import torch
from torch import nn
from sklearn.metrics import f1_score, matthews_corrcoef

from Dataset import Twibot20
from utils import accuracy, init_weights
from model import (
    ESABotRGCN, ESABotGAT, ESABotGCN, ESABotNoGCN, ESAFastBotRGCN,
    ESABotRGCN_1layer, ESABotRGCN_3layers, ESABotRGCN_4layers,
    ESABotRGCN_5layers, ESABotRGCN_8layers, ESABotRGCNWithAttention
)

# ====== Configurations ======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size, dropout = 128, 0.3
lr, weight_decay = 1e-3, 5e-3
epochs = 90
target_acc = 0.876

# ====== Load Dataset ======
dataset = Twibot20(device=device, process=False, save=True)
(
    des_tensor, tweets_tensor, num_prop, category_prop,
    new_feature, edge_index, edge_type, labels,
    train_idx, val_idx, test_idx
) = dataset.dataloader()

# ====== Model Selection ======
model = ESABotRGCN(num_prop_size=7, cat_prop_size=3).to(device)

# ✅ Optional models:
# model = ESAFastBotRGCN(num_prop_size=7, cat_prop_size=3).to(device)
# model = ESABotGAT(num_prop_size=7, cat_prop_size=3).to(device)
# model = ESABotRGCN_4layers(num_prop_size=7, cat_prop_size=3).to(device)
# model = ESABotRGCNWithAttention(
#     des_size=768, tweet_size=768,
#     num_prop_size=7, cat_prop_size=3,
#     new_feature_size=1, embedding_dimension=embedding_size,
#     dropout=dropout
# ).to(device)

# ====== Training Setup ======
model.apply(init_weights)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# ====== Training Function ======
def train(epoch):
    model.train()
    output = model(des_tensor, tweets_tensor, num_prop, category_prop, new_feature, edge_index, edge_type)
    loss = loss_fn(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Train Acc: {acc_train:.4f} | Val Acc: {acc_val:.4f}")
    return acc_train.item(), loss.item()

# ====== Testing Function ======
def test():
    model.eval()
    output = model(des_tensor, tweets_tensor, num_prop, category_prop, new_feature, edge_index, edge_type)
    loss = loss_fn(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])

    pred = output.argmax(dim=1).cpu().numpy()
    true = labels.cpu().numpy()
    f1 = f1_score(true[test_idx], pred[test_idx])
    mcc = matthews_corrcoef(true[test_idx], pred[test_idx])

    print("\n📊 Test Results:")
    print(f"  Loss      : {loss.item():.4f}")
    print(f"  Accuracy  : {acc_test:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  MCC       : {mcc:.4f}")

    return acc_test.item(), loss.item(), f1

# ====== Training Loop ======
for epoch in range(epochs):
    acc_train, _ = train(epoch)
    if acc_train >= target_acc:
        print(f"\n✅ Early stopping at epoch {epoch+1} (Train Acc: {acc_train:.4f})")
        break

# ====== Final Evaluation ======
print("\n🔍 Running final evaluation on test set...")
test()
