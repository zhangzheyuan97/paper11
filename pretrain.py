from nltk import word_tokenize
from string import punctuation
import csv
from tqdm import tqdm
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import torch
import pickle as pkl
import os
def readcsv(dataset):
    all_text_list = []
    train_file = dataset + "/data/train.csv"
    dev_file = dataset + "/data/dev.csv"
    test_file = dataset + "/data/test.csv"
    file_list = [train_file,dev_file,test_file]
    for file in file_list:
        text_list = []
        with open(file,"r",encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line in tqdm(reader):
                content = line['data']  # str  int
                text_list.append(content)
        all_text_list = all_text_list + text_list
    return all_text_list   #得到所有句子的列表
def pre_process(comment_list):
    num_punc = "123456789" #去掉数字
    word_list = []
    for item in comment_list:
        clean_text = ''.join(char.lower() for char in item if char not in punctuation and char not in num_punc)
        #去掉数字和违规字符 并转为小写
        word_list.append(word_tokenize(clean_text))
    #一句话str变为单词列表
    return word_list
def create_dic(word_list):
    words_source = [word for sentence in word_list for word in sentence]
    various_words = list(set(words_source))
    int_word = dict(enumerate(various_words,1))
    word_int = {w: int(i) for i, w in int_word.items()}
    word_int["<unk>"] = 0
    int_word[0] = "<unk>"
    return int_word,word_int
def bulid_pretrain_model(pretrain_path,dic_len,word_to_idx): #输入的是词典大小
    tmp_file = get_tmpfile(pretrain_path)
    wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
    vocab_size = dic_len + 1
    embed_size = 300  # 维度需要和预训练词向量维度统一
    weight = torch.zeros(vocab_size + 1, embed_size)
    for i in range(len(wvmodel.index2word)):
        try:
            index = word_to_idx[wvmodel.index2word[i]]
        except:
            continue

        weight[index, :] = torch.from_numpy(wvmodel.get_vector(wvmodel.index2word[i]))
    return weight
if __name__ == '__main__':
    dict_path = os.getcwd() + "/dict.pth"
    glove_path = os.getcwd() +"/glove.txt"
    dataset_list = ["Amazon_CLoth","Amazon_instant_video","COVID","IMDB","SST","SST2","Tweet"]
    all_text_list = []
    for item in dataset_list:
        all_text_list = all_text_list + readcsv(item) #所有句子的列表
    all_word_list = pre_process(all_text_list) #所有单词的列表 example:[["i like"],["he like"]]
    int_word_dict,word_int_dict = create_dic(all_word_list)
    dic_len = len(int_word_dict)
    weight = bulid_pretrain_model(glove_path,dic_len = dic_len,word_to_idx= word_int_dict)
    final_dict = {}
    final_dict["weight"] = weight
    final_dict["word_int"] = word_int_dict
    final_dict["int_word"] = int_word_dict
    print(len(final_dict["word_int"]))
    torch.save(final_dict,dict_path)
