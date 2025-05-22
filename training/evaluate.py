import torch

# 评估函数 用来评估模型效果
def evaluate_model(model, classifier, dataloader, device):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, labels in dataloader:
            feats = feats.to(device)
            labels = labels.to(device)
            embeddings = model(feats).squeeze(1)
            outputs = classifier(embeddings)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"🎯 验证集准确率: {acc:.4f}")
    return acc