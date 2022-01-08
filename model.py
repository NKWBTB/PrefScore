import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import config as CFG

class Scorer(nn.Module):
    def __init__(self):
        super(Scorer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(CFG.BERT_MODEL)
        self.model = BertModel.from_pretrained(CFG.BERT_MODEL)
        self.fc = nn.Linear(self.model.config.hidden_size, 1)
    
    def forward(self, article, summary):
        inputs = self.tokenizer(article, summary, padding='longest', truncation="longest_first" , return_tensors='pt').to(CFG.DEVICE)
        outputs = self.model(**inputs)
        x = self.fc(outputs.pooler_output)
        return x

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.base_model = Scorer()
    
    def forward(self, article, summary1, summary2):
        out1 = self.base_model(article, summary1)
        out2 = self.base_model(article, summary2)
        out = torch.cat((out1, out2), -1)
        return out

class CustomDataset(Dataset):
    def __init__(self, datapath, nums=None):
        self.data = []
        with open(datapath, "r", encoding="utf-8") as f:
            for line in f:
                elements = line.split('\t')
                size = len(elements)
                for i in range(1, size-1):
                    self.data.append([elements[0], elements[i], elements[i+1]])
                # Limit the number of lines used
                if nums is not None:
                    nums -= 1
                    if nums == 0:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]


def train_model(model, train_set, max_iter=CFG.MAX_ITERATION):
    epochs = CFG.EPOCHS
    optimizer = torch.optim.Adam(model.parameters(), lr = CFG.LR)
    loss_fn = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_set, batch_size=CFG.BATCH_SIZE, shuffle=True)

    while epochs > 0:
        running_loss = 0.0
        model.train()
        with tqdm(total=min(len(train_dataloader), max_iter)) as pbar:
            for j, (article, sum1, sum2) in enumerate(train_dataloader):
                if j >= max_iter:
                    break
                output = model(article, sum1, sum2)
                labels = torch.tensor([0]*len(article), dtype=torch.long).to(CFG.DEVICE)
                loss = loss_fn(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                pbar.update(1)
                if j % 100 == 99:
                    pbar.write("Iteration {}, Loss {}".format(j+1, running_loss))
                    running_loss = 0
        
        epochs -= 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model from a preprocessed dataset.")
    parser.add_argument('--dataset', '-d', default='billsum',
                        help="Support 'billsum', 'big_patent' or 'scientific_papers'.")
    args = parser.parse_args()

    DATASET=args.dataset

    train_set = CustomDataset(os.path.join(CFG.DATASET_ROOT, DATASET, CFG.METHOD, 'train.tsv'))
    print(len(train_set))

    model = Siamese()
    model.to(CFG.DEVICE)

    print("Training from", DATASET)
    train_model(model, train_set)

    CKPT_PATH = os.path.join(CFG.RESULT_ROOT, DATASET, CFG.METHOD, "model.pth")
    if not os.path.exists(os.path.dirname(CKPT_PATH)):
        os.makedirs(os.path.dirname(CKPT_PATH))

    scorer = model.base_model
    torch.save(scorer.state_dict(), CKPT_PATH)



