import numpy as np
import pandas as pd
import pkuseg
import pprint
seg = pkuseg.pkuseg(model_name="C:/Users/dell/Desktop/weibo_model")
from gsdmm import MovieGroupProcess
# set seed for reproducibility
np.random.seed(493)
import nltk

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

###########################################parameter
beta=0.1
alpha=0.1
k=20
n_iters=32
############################################################################   2
# load data
#df: original data including all information
df = pd.read_csv("C:/Users/dell/Desktop/gender_data/gen1_sample.csv",encoding='utf-8',chunksize=None)
print('df loading finished')
#doc:just include all tweet text which have been cleaned(remove stopwords,punctuation,etc)
docs = []
with open('C:/Users/dell/Desktop/gender_data/gen1_sample_clean.csv','r',encoding='utf-8') as fp:
    number=0
    for line in fp:
        word=line.replace('\n','').split(',')
        docs.append(word)
        print(str(number)+' finished')
        number+=1

##################################################################4  4
#clustering
mgp = MovieGroupProcess(K=k, alpha= alpha, beta=beta, n_iters=n_iters)
vocab = set(x for doc in docs for x in doc)
n_terms = len(vocab)
y = mgp.fit(docs, n_terms)
#################################################################5
#display results
doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topic :', doc_count)
top_index = doc_count.argsort()[-k:][::-1]
print('Most important clusters (by number of docs inside):', top_index)
#################################################################7
def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts =sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print('Cluster %s : %s'%(cluster,sort_dicts))
        print('-'*120)
################################################################  8
top_words(mgp.cluster_word_distribution, top_index, 7)
##########################################################  9
topic_dict = {}
topic_names=[]
#create topic name list(topic1 topic2.....)
for topic_index in range(1,k+1):
    topic_names.append('Topic #'+str(topic_index))

for i, topic_num in enumerate(top_index):
    topic_dict[topic_num]=topic_names[i]

def create_topics_dataframe(data_text=df.text,  mgp=mgp, threshold=0.3, topic_dict=topic_dict, stem_text=docs):
    result = pd.DataFrame(columns=['text', 'topic', 'stems'])
    for i, text in enumerate(data_text):
        result.at[i, 'text'] = text
        result.at[i, 'stems'] = stem_text[i]
        prob = mgp.choose_best_label(stem_text[i])
        if prob[1] >= threshold:
            result.at[i, 'topic'] = topic_dict[prob[0]]
        else:
            result.at[i, 'topic'] = 'Other'
    return result

#create dataframe, match the topics to the text
dfx = create_topics_dataframe(data_text=df.text,  mgp=mgp, threshold=0.3, topic_dict=topic_dict, stem_text=docs)
pprint.pprint(dfx.head())
print(dfx.topic.value_counts(dropna=False))