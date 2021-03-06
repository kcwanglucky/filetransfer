import argparse
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from IPython.display import clear_output
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class OnlineQueryDataset(Dataset):
    """
    實作一個可以用來讀取訓練 / 測試集的 Dataset，此 Dataset 每次將 tsv 裡的一筆成對句子
    轉換成 BERT 相容的格式，並回傳 3 個 tensors：
    - tokens_tensor：兩個句子合併後的索引序列，包含 [CLS] 與 [SEP]
    - segments_tensor：可以用來識別兩個句子界限的 binary tensor
    - label_tensor：將分類標籤轉換成類別索引的 tensor, 如果是測試集則回傳 None
    """
    def __init__(self, mode, df, tokenizer, path = None):
        """
        Args:
            mode: in ["train", "test", "val"]
            df: first column - label; second column - question
            tokenizer: one of bert tokenizer
            perc: percentage of data to put in training set
            path: if given, then read df from the path(ex training set)
        """
        assert mode in ["train", "val", "test"]
        self.mode = mode

        if path: 
            self.df = pd.read_csv(path, sep="\t").fillna("")
        else:
            self.df = df
        self.len = len(self.df)
        self.tokenizer = tokenizer 
    
    #@pysnooper.snoop()  # 加入以了解所有轉換過程
    def __getitem__(self, idx):
        """定義回傳一筆訓練 / 測試數據的函式"""
        if self.mode == "test":
            text = self.df.iloc[idx, 1]
            label_tensor = None
        elif self.mode == "val":
            label, text = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)
        else:
            label, text = self.df.iloc[idx, :].values
            # 將label文字也轉換成索引方便轉換成 tensor
            label_tensor = torch.tensor(label)
        
        # create BERT tokens for sentence
        word_pieces = ["[CLS]"]
        tokens = self.tokenizer.tokenize(text)
        word_pieces += tokens + ["[SEP]"]
        len_a = len(word_pieces)
        
        # convert tokens to tokensid
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # set every non [sep] token to 1, else 0
        segments_tensor = torch.tensor([1] * len_a, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

def create_mini_batch(samples):
    """
    實作可以一次回傳一個 mini-batch 的 DataLoader
    這個 DataLoader 吃我們上面定義的 OnlineQueryDataset，
    回傳訓練 BERT 時會需要的 4 個 tensors
    它會對前兩個 tensors 作 zero padding，並產生前面說明過的 masks_tensors
    Returns:
        tokens_tensors  : (batch_size, max_seq_len_in_batch)
        segments_tensors: (batch_size, max_seq_len_in_batch)
        masks_tensors   : (batch_size, max_seq_len_in_batch)
        label_ids       : (batch_size)
    """

    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # 訓練集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

def get_num_labels(df):
    num_labels = len(df['index'].value_counts())
    return num_labels

def create_index2label(df):
    """
    Return:
        A dictionary {class_index: class_label}
    """
    index = df['index']
    index2label = {idx:val for idx, val in enumerate(index.unique())}
    return index2label

class Model():
    def __init__(self, df, model_name):
        self.df = df

        self.num_labels = get_num_labels(df)
        self.index2label = create_index2label(df)
        self.model = BertForSequenceClassification.from_pretrained(
                    model_name, num_labels = self.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def get_num_label():
        return self.num_labels

    def get_index2label():
        return self.index2label

    def train(self, trainloader, valloader, epochs):
        
        # 讓模型跑在 GPU 上並取得訓練集的分類準確率
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        model = self.model.to(device)
        pred, acc = get_predictions(model, trainloader, compute_acc=True)
        
        # 使用 Adam Optim 更新整個分類模型的參數
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        for epoch in range(epochs):
            
            running_loss = 0.0
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            print('Training...')

            # 訓練模式
            model.train()

            for data in trainloader: # trainloader is an iterator over each batch
                tokens_tensors, segments_tensors, \
                masks_tensors, labels = [t.to(device) for t in data]

                # 將參數梯度歸零
                optimizer.zero_grad()
                
                # forward pass
                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors, 
                                labels=labels)

                loss = outputs[0]
                # backward
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # 紀錄當前 batch loss
                running_loss += loss.item()
                
                # 計算分類準確率
            logit, acc = get_predictions(model, trainloader, compute_acc=True)

            print('loss: %.3f, acc: %.3f' % (running_loss, acc))    
            print("")
            print("Running Validation...")
            
            _, acc = get_predictions(model, valloader, compute_acc=True)

            # Report the final accuracy for this validation run.
            print("  Accuracy: {0:.2f}".format(acc))
        self.model = model
        return model

    def save_model(self, output_dir):
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)
        # Save a trained model, configuration and tokenizer using 'save_pretrained()'.
        # They can then be reloaded using 'from_pretrained()'
        model = self.model
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        #torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    def predict(self, testloader):
        clear_output()
        model = self.model

        predictions = get_predictions(model, testloader).detach().cpu().numpy()
        self.predict = predictions
        return predictions

def get_predictions(model, dataloader, compute_acc=False):
    """ Use the given "model" to make predictions on the "dataloader"
    Args:
        model: a pytorch BertForSequenceClassification model
        dataloader: the dataloader to make prediction
        compute_acc: whether to output accuracy score
    Returns:
        predictions (list): The classification result
        acc: The accuracy score
    """

    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions

def write_prediction(pred, output_name):
    if not os.path.exists("prediction"):
        os.makedirs("prediction")
    print("Saving prediction to %s" % os.path.join("prediction", output_name, ".txt"))
    with open(os.path.join("prediction", output_name, ".txt"), 'w') as opt:
        opt.write('%s' % pred)

def plain_accuracy(label, pred):
    return (label == pred).sum().item()/len(label)

