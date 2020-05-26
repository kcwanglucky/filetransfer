import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#from transformers import BertModel, BertTokenizer
#import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from bert_classification import *

def preprocess_app_comment(df, verbose = False):
    """，轉化 app comment 成為後續 model 可以理解的格式，並回傳清理過後的 dataframe
    Args:
        df (DataFrame): pd.Dataframe from the app comment raw files
        verbose (int): If True, print 各類別對應的筆數
    Returns:
        DataFrame: The DataFrame with only relevant data
    """
    df.rename(columns={"評論標題":"title", "評論內容": "question", "類別": "index"}, inplace = True)
    df.dropna(axis = 0, how = 'any', subset=["question", "index"], inplace = True)
    if verbose:
        print(df['index'].value_counts())
    df = df.fillna('')
    return df

def filter_toofew_toolong(df, min_each_group, max_length):
    """ Filter out groups with data fewer than min_each_group and filter 
    out data longer than max_length
    """
    df = df[~(df.question.apply(lambda x : len(x)) > max_length)]

    counts = df["index"].value_counts()
    idxs = np.array(counts.index)
    
    # index numbers of groups with count >= mineachgroup
    list_idx = [i for i, c in zip(idxs, counts) if c >= min_each_group]

    # filter out data with "index" in list_idx 
    df = df[df["index"].isin(list_idx)]
    return df

def append_question_title(df):
    df['question'] = df['title']+ df['question'].astype(str)
    return df

class AppCommentData():
    def __init__(self, df, mode, tokenizer = None, batch_size = None):
        """
        Args:
            mode: in ["train", "test", "val", "all"]
            tokenizer: one of bert tokenizer
        """
        self.df = df
        self.mode = mode
        self.tokenizer = tokenizer
        self.batch_size = batch_size
    
    def reindex(self, label2index):
        """Reindex the df given the mapping in label2index so that 
        it can be fed to model
        Called to reindex the train val test data to the label in "all" data
        """
        df_reindex = self.df.copy()
        def get_index_for_label(label):
            return label2index[label]
        df_reindex["index"] = df_reindex["index"].apply(get_index_for_label)
        self.df_reindex = df_reindex

    def get_index2label(self):
        """
        Return:
            A dictionary {class_index: class_label}
        """
        index = self.df['index']
        index2label = {idx:val for idx, val in enumerate(index.unique())}
        return index2label

    def get_label2index(self):
        index2label = self.get_index2label()
        return {lab:idx for idx, lab in index2label.items()}

    def get_num_index(self):
        return len(self.get_index2label())

    def get_index_dist(self, verbose = False):
        """得到此 dataframe 的訓練分布 (分類:筆數) return the mapping of value and occurrence
           df: "pd.DataFrame" type
        """
        each_count = self.df['index'].value_counts()
        if verbose:
            print(self.mode + " data 各類分布: \n")
            for index in each_count.index:
                print("{:15} 數量: {}".format(index, each_count[index] ))
        return each_count

    def plot_pie(self, fontprop):
        plt.figure(figsize=(10, 10), facecolor="w")
        index_dist = self.get_index_dist()

        plt.title("{} data Distribution ({} data)".format(self.mode, sum(index_dist)), fontsize=16)
        patches,l_text, p_text = plt.pie(index_dist, autopct="%1.1f%%",
                                        textprops = {"fontsize" : 12}, labeldistance=1.05)
        for i, t in enumerate(l_text): 
            t.set_text(index_dist.index[i])
            t.set_fontproperties(fontprop)
            #t.set_color('red')
            pct = float(p_text[i].get_text().strip('%'))
            if pct < 2:
                p_text[i].set_text("")
            #p_text[i].set_color('red')
        plt.show()

    def get_dataset(self):
        """label2index contains label to index mapping as in the all dataset"""
        if self.mode == "test":
            return OnlineQueryDataset(self.mode, self.df, self.tokenizer)
        else:
            return OnlineQueryDataset(self.mode, self.df_reindex, self.tokenizer)

    def get_dataloader(self):
        """Return a dataloader that can be fed to """
        shuffle = True if self.mode == "train" else False
        return DataLoader(self.get_dataset(), batch_size=self.batch_size, shuffle = shuffle,  
                            collate_fn=create_mini_batch)
    """Usage:
    app_data = AppCommentData(df)
    app_data.plot_pie()
    """

def sample_from_each_group(data, fraction):
    """從各類別 random sample 出 fraction 比例的資料集
    Args:    
        data: df data that includes the "index" and "question" column
        fraction: the fraction of data you want to sample (ex: 0.7)
    """ 
    def sampleClass(classgroup):
        """This function will be applied on each group of instances of the same
        class in data
        """
        return classgroup.sample(frac = fraction)
    samples = data.groupby('index').apply(sampleClass)
    
    # If you want an index which is equal to the row in data where the sample came from
    # If you don't change it then you'll have a multiindex with level 0
    samples.index = samples.index.get_level_values(1)
    return samples


def output_split(df, fraction = 0.7):
    """將原本全部的cleaned data依照指定的比例分成train/val/test set，
    並output成tsv檔到環境中(檔名ex: 70%train.tsv)
    Args:
        df: df data that includes the "index" and "question" column
        fraction: fraction of all data to be assigned to training set
    """
    df_train = sample_from_each_group(df, fraction)
    df_remain = pd.concat([df_train, df]).drop_duplicates(keep=False)
    df_val = df_remain.sample(frac = 0.5, random_state = 5)
    df_test = pd.concat([df_val, df_remain]).drop_duplicates(keep=False)
    del df_remain

    print("訓練樣本數：", len(df_train))
    print("validation樣本數：", len(df_val))
    print("預測樣本數：", len(df_test))
    return (df_train, df_val, df_test)

def get_confusion_matrix(true_label, predictions, num_index):
    """Return a group-to-group comparison matrix and a list of list storing
       the index of comments that are assigned to a wrong group
    Args:
        true_label: an array that stores the truth label of the test set
        predictions: an array that stores the model prediction
        num_index: how many classes used in the model
    Returns:
        class_matrix (list of list): The confusion matrix
        false_group (list of list): false_group[i] - comments in group i that are
            assigned to other groups
    """
    class_matrix = np.zeros(shape=(num_index, num_index))
    false_group = [[] for _ in range(num_index)]
    for idx, true, pred in zip(range(len(predictions)),true_label, predictions):
        class_matrix[true][pred] += 1
        if true != pred:
            false_group[true].append(idx)
    return class_matrix, false_group

def print_acc(class_matrix):
    """print the accuracy given a confusion matrix"""
    total = 0
    num_index = len(class_matrix)
    for i in range(num_index):
        total += class_matrix[i][i]
    print("Accuracy: {0}%".format(100 * total/np.sum(class_matrix)))