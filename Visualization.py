import csv
import pandas as pd
import sys
import re

import matplotlib.pyplot as pPlot
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder

from pandas import DataFrame
from collections import Counter

#visualisation of results Jaccard micro
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.34, 0.089, 0.3,0.083]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('Jaccard micro')
pPlot.show()

#visualisation of results Jaccard macro
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.099, 0.018, 0.087,0.017]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('Jaccard macro')
pPlot.show()





#visualisation of results Recall micro
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.34, 0.089, 0.3,0.083]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('Recall micro')
pPlot.show()

#visualisation of results Recall macro
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.11, 0.018, 0.133,0.017]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('Recall macro')
pPlot.show()






#visualisation of results Precision micro
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.65, 0.87, 0.3,0.84]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('Precision micro')
pPlot.show()

#visualisation of results Precision macro
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.21, 0.1, 0.14,0.1]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('Precision macro')
pPlot.show()







#visualisation of results Hamming loss
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.001, 0.001, 0.002,0.001]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('Hamming loss')
pPlot.show()


#visualisation of results F1 macro
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.139, 0.028, 0.13,0.027]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('F1 macro')
pPlot.show()
#visualisation of results F1 macro
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.139, 0.028, 0.13,0.027]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('F1 macro')
pPlot.show()


#visualisation of results F1 macro
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.452, 0.163, 0.302,0.152]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('F1 macro')
pPlot.show()

#visualisation of results F1     micro
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.452, 0.163, 0.302,0.152]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('F1 micro')
pPlot.show()
#visualisation of result exact matching score
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [0.036, 0.0038, 0.054,0.0042]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('Exact Matching score')
pPlot.show()

#visualisation of result time
x = ["MLKNN", "BR-RF","LP-RF","CC-RF"]
y = [23.50, 62.5, 16.67,66.67]
pPlot.plot(x, y )
pPlot.xlabel('Models')
pPlot.ylabel('TrainingTime')
pPlot.show()

maxInt=(sys.maxsize)
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

def read_data(s):
    dataset = []
    input_file = csv.reader(open(s,encoding='utf-8'))
    for row in input_file:
        if(len(row)>0):
            dataset.append(row)
    return pd.DataFrame(dataset,index=None)

content=read_data("content.csv")
label=read_data("label.csv")
dataset = open("content.csv", "r",encoding='utf-8').read()
#print(content)
#print(label)
labels=[]
contents=[]
alllabels=[]
print(len(content))
print(len(label))
cont=[]
lab=[]
result = pd.merge(content, label,left_on=0,right_on=0)
for row in result['1_y']:
   r=len(row)
   labels.append(re.findall(r"'(.*?)'", row, re.DOTALL))
i=0;
for con in result['1_x']:
   for r in range(0,len(labels[i])):
      cont.append(re.findall(r"'(.*?)'", con, re.DOTALL))
   i = i + 1
for i in range(0,len(labels)):
    for r in labels[i]:
        alllabels.append(r)

print(len(alllabels))
print(len(cont))
#to know in how many documents each label is present
know=(Counter(alllabels))
print((know))
Cars = {'Content': cont,
        'label': alllabels
        }
#some visualisation to get a gist of data
df = DataFrame(Cars,columns= ['Content', 'label'])
print(df.head(10))
df.to_csv("output.csv")

#visualsation of content length
lens = content[1].str.len()
lens.hist(bins = np.arange(0,5000,50))
pPlot.plot(lens)
pPlot.show()

#word cloud
def create_word_cloud(string):
   maskArray = np.array(Image.open("cloud.png"))
   cloud = WordCloud(background_color = "white", max_words = 200, mask = maskArray, stopwords = set(STOPWORDS))
   cloud.generate(string)
   cloud.to_file("wordCloud.png")

create_word_cloud(dataset)

# create groups of documents with respect to label counts
j=0
for con in result['1_x']:
   i = 0
   for r in range(0,len(labels[j])):
      cont.append(re.findall(r"'(.*?)'", con, re.DOTALL))
      i = i + 1
   j=j+1
   lab.append(i)

content['count']=lab
print(content)

print(content.groupby('count').count())
contentsub=content.head(500)
#contentsub.plot(x=0,y='count','.')
pPlot.bar(contentsub[0], contentsub['count'], alpha=0.2
                   )
pPlot.show()

#label cooccurence graph
graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
edge_map = graph_builder.transform(result['1_x'])
print("{} labels, {} edges".format(3000, len(edge_map)))
print(edge_map)

