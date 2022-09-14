import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from transformers import BertTokenizer, BertModel, BertTokenizerFast, logging
from tqdm import tqdm
import config as CFG


logging.set_verbosity_error()

class Scorer(nn.Module):
    def __init__(self, separateEncode = False, use_pooler = True, use_lscore = False):
        super(Scorer, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(CFG.BERT_MODEL)
        self.separateEncode = separateEncode
        self.use_pooler = use_pooler
        self.similarity = 'Cosine' #'InnerProduct' #    
        self.use_lscore = use_lscore
        self.model = BertModel.from_pretrained(CFG.BERT_MODEL)
        if not self.separateEncode:
            self.fc = nn.Linear(self.model.config.hidden_size, 1)
        if self.use_lscore:
            self.decoder_l = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, bias=True),
                nn.GELU(),
                nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=True)
            )
    
    def encode(self, text):
        inputs = self.tokenizer(list(text), padding="max_length", truncation=True , return_tensors='pt').to(CFG.DEVICE)
        outputs = self.model(**inputs)
        return outputs, inputs

    def l_score(self, summary_inputs, summary_outputs):
        sum_seq_output = summary_outputs.last_hidden_state
        input_ids = summary_inputs["input_ids"]
        input_mask = summary_inputs["attention_mask"]
        batch_size = summary_outputs.last_hidden_state.shape[0]

        score = self.decoder_l(sum_seq_output) #.unsqueeze(0)
        score = F.log_softmax(score, dim=2)

        # temp = torch.zeros(batch_size, self.model.config.max_position_embeddings, self.model.config.vocab_size).to(CFG.DEVICE)
        # one_hot_input_ids = temp.scatter_(2, input_ids.view(batch_size, -1, 1), 1).float()

        score = torch.sum(torch.gather(score, 2, input_ids.view(batch_size, -1, 1)).view(batch_size, -1), dim=-1, keepdim=True) / \
            (torch.sum(input_mask, dim=-1, keepdim=True).float())
        score = (score+200)/100
        return score
    
    def s_score(self, article_outputs, summary_outputs):
        if self.similarity == 'InnerProduct':
            x = torch.sum(article_outputs.pooler_output * summary_outputs.pooler_output, dim=-1, keepdim=True) \
                if self.use_pooler \
                else torch.sum(article_outputs.last_hidden_state[:, 0] * summary_outputs.last_hidden_state[:, 0], dim=-1, keepdim=True)
        elif self.similarity == 'Cosine':
            x = F.cosine_similarity(article_outputs.pooler_output, summary_outputs.pooler_output).view(-1, 1) \
                if self.use_pooler \
                else F.cosine_similarity(article_outputs.last_hidden_state[:, 0], summary_outputs.last_hidden_state[:, 0]).view(-1, 1)
        return x
    
    def forward(self, article, summary):
        if not self.separateEncode:
            inputs = self.tokenizer(article, summary, padding='max_length', truncation="longest_first" , return_tensors='pt').to(CFG.DEVICE)
            outputs = self.model(**inputs)
            x = self.fc(outputs.pooler_output) if self.use_pooler else self.fc(outputs.last_hidden_state[:, 0])
        else:
            article_outputs, article_inputs = self.encode(article)
            summary_outputs, summary_inputs = self.encode(summary)
            x = self.s_score(article_outputs, summary_outputs)
            if self.use_lscore:
                x += self.l_score(summary_inputs, summary_outputs)
        return x

class Siamese(nn.Module):
    def __init__(self, separateEncode = False, use_pooler = True, use_lscore = False):
        super(Siamese, self).__init__()
        self.base_model = Scorer(separateEncode, use_pooler, use_lscore)
    
    def forward(self, article, summary1, summary2):
        if not self.base_model.separateEncode:
            out1 = self.base_model(article, summary1)
            out2 = self.base_model(article, summary2)
        else:
            article_outputs, article_inputs = self.base_model.encode(article)
            summary1_outputs, summary1_inputs = self.base_model.encode(summary1)
            summary2_outputs, summary2_inputs = self.base_model.encode(summary2)

            out1 = self.base_model.s_score(article_outputs, summary1_outputs)
            out2 = self.base_model.s_score(article_outputs, summary2_outputs)
            if self.base_model.use_lscore:
                out1 += self.base_model.l_score(summary1_inputs, summary1_outputs)
                out2 += self.base_model.l_score(summary2_inputs, summary2_outputs)
    
        return torch.cat((out1, out2), -1)

class CustomDataset(Dataset):
    def __init__(self, datapath, nums=None, hierarchical=False):
        self.data = []
        print("Hierarchichal", hierarchical)
        with open(datapath, "r", encoding="utf-8") as f:
            for line in f:
                elements = line.split('\t')
                size = len(elements)
                for i in range(1, size-1):
                    if hierarchical:
                        self.data.append([elements[0], elements[i], elements[i+1]])
                    else:
                        self.data.append([elements[0], elements[1], elements[i+1]])
                # Limit the number of lines used
                if nums is not None:
                    nums -= 1
                    if nums == 0:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]

class AntiRougeDataset(CustomDataset):
    def __init__(self, datapath):
        self.data = []
        with open(datapath, "r", encoding="utf-8") as f:
            for line in f:
                elements = line.split('\t')
                size = len(elements)
                for i in range(3, size-1, 2):
                    self.data.append([elements[0], elements[1], elements[i]])

def train_model(model, train_set, max_iter=CFG.MAX_ITERATION, loss_func='CrossEntropyLoss', margin=0.0, shuffle=True):
    optimizer = torch.optim.Adam(model.parameters(), lr = CFG.LR)
    loss_fn = nn.CrossEntropyLoss()
    if loss_func == 'MarginRankingLoss': loss_fn = nn.MarginRankingLoss(margin=margin)

    train_dataloader = DataLoader(train_set, batch_size=CFG.BATCH_SIZE, shuffle=shuffle)
    print(loss_func, margin, shuffle)

    running_loss = 0.0
    model.train()
    num_iter = 0
    with tqdm(total=max_iter) as pbar:
        while num_iter < max_iter:
            for j, (article, sum1, sum2) in enumerate(train_dataloader):
                if num_iter >= max_iter:
                    break
                output = model(article, sum1, sum2)
                if loss_func == 'CrossEntropyLoss':
                    labels = torch.tensor([0]*len(article), dtype=torch.long).to(CFG.DEVICE)
                    loss = loss_fn(output, labels)
                else:
                    labels = torch.tensor([1]*len(article), dtype=torch.long).to(CFG.DEVICE)
                    loss = loss_fn(output[:, 0], output[:, 1], labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                num_iter += 1
                pbar.update(1)
                if num_iter % 1000 == 999:
                    pbar.write("Iteration {}, Loss {}".format(num_iter+1, running_loss))
                    running_loss = 0
        

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



