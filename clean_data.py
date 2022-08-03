import numpy as np
import pandas as pd
import csv
import pkuseg
import pprint
seg = pkuseg.pkuseg(model_name="C:/Users/dell/Desktop/weibo_model")
from zhon.hanzi import punctuation
from gsdmm import MovieGroupProcess

# set seed for reproducibility
np.random.seed(493)
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
ps = nltk.porter.PorterStemmer()
import unicodedata
import re
# to print out all the outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


############################################################################   2
# load data and stop words
df = pd.read_csv("C:/Users/dell/Desktop/gender_data/gen1_sample.csv",encoding='utf-8',chunksize=None)
stopword_list=[]
with open('stop_words.txt','r',encoding='utf-8') as fp:
    for line in fp:
        line = line.replace("\n", '')
        stopword_list.append(line)
##########################################################################3
### clean the data , remove stop words
def basic_clean(original):
    #word = unicodedata.normalize('NFKD', original)
    #     .encode('ascii', 'ignore') \
    #     .decode('utf-8', 'ignore')
    # print(word)
    word=re.sub(r'[%s]+' %punctuation,' ', original)
    word = word.replace('\n', ' ')
    word = word.replace('\t', ' ')
    return word
def remove_stopwords(original):
      # 以默认配置加载模型
    words = seg.cut(original)  # 进行分词
    for i in range(len(words)-1, -1, -1):
        if words[i] in (stopword_list):
            del words[i]
    original_nostop = ' '.join(words)
    return original_nostop
def stem(original):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in original.split()]
    original_stemmed = ' '.join(stems)
    return original_stemmed
# words='我爱北京天安门，有人说我们今天去吃一顿好吃的吧，好嘛'
# print(remove_stopwords(basic_clean(words)))
########清理数据
docs = []
number=0
words_clean=''
with open("C:/Users/dell/Desktop/gender_data/gen1_sample_clean.csv",'w',encoding='utf-8',newline='') as fp:

    for sentence in df.text:
        f_csv = csv.writer(fp)
        words = word_tokenize(stem(remove_stopwords(basic_clean(sentence))))

        f_csv.writerow(words)
        number+=1
        print(str(number)+' finished')
