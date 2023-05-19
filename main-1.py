import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
import json
import torchtext

# 数据预处理
TEXT = torchtext.legacy.data.Field(tokenize='spacy')
LABEL = torchtext.legacy.data.Field(dtype=torch.float)

# 加载数据集
def load_dataset(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]
    return data_list

train_data = load_dataset('train.json')
test_data = load_dataset('test.json')

train_examples = [data.Example.fromdict(example, fields={"text": ('text', TEXT), "label": ('label', LABEL)})
                  for example in train_data]
test_examples = [data.Example.fromdict(example, fields={"text": ('text', TEXT), "label": ('label', LABEL)})
                 for example in test_data]

train_dataset = data.Dataset(train_examples, fields={"text": TEXT, "label": LABEL})
test_dataset = data.Dataset(test_examples, fields={"text": TEXT, "label": LABEL})

TEXT.build_vocab(train_dataset, max_size=10000)
LABEL.build_vocab(train_dataset)

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_dataset, test_dataset), batch_size=64, device=torch.device('cuda'))

# 定义模型
class TextClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)

# 初始化模型和优化器
input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1

model = TextClassifier(input_dim, embedding_dim, hidden_dim, output_dim)
model = model.to(torch.device('cuda'))

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# 训练模型
def train(model, iterator, optimizer, criterion):
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text = batch.text
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

# 测试模型
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    predictions_list = []
    ids_list = []

    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()

            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct += (rounded_preds == batch.label).float().sum()

            predictions_list.extend(rounded_preds.tolist())
            ids_list.extend(batch.ID.tolist())

    accuracy = correct / len(iterator.dataset)
    return epoch_loss / len(iterator), accuracy, predictions_list, ids_list

# 训练和评估模型
N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train(model, train_iterator, optimizer, criterion)
    train_loss, train_acc, _, _ = evaluate(model, train_iterator, criterion)
    test_loss, test_acc, test_preds, test_ids = evaluate(model, test_iterator, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2%}')
    print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc:.2%}')

    # 保存测试集的预测结果为JSON文件
    test_results = []
    for id, label in zip(test_ids, test_preds):
        test_result = {"ID": id, "label": int(label)}
        test_results.append(test_result)

    with open('test_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=4)
