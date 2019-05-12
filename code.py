#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:57:13 2019

"""

from sklearn import tree 
import numpy as np 
from sklearn.metrics import accuracy_score

with open("/Users/haojie/Desktop/sw/3.27/avila/avila-tr.txt", "r")as file:
    data = file.read().split("\n")

with open("/Users/haojie/Desktop/sw/3.27/avila/avila-ts.txt", "r")as file1:
    data1 = file1.read().split("\n")

label={"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "H":8, "I":9, "W":10, 
       "X":11, "Y":12}


def data_process(data):
    train_X = []
    train_Y = []
    for i in range(len(data)-1):
        sample = data[i].split(",")
        x = [] 
        y = [] 
        for j in range(len(sample)-1):
            x.append(float(sample[j]))
        target = label[sample[-1]]
        y.append(target)
        train_X.append(x)
        train_Y.append(y)
    return train_X, train_Y

a, train_Y = data_process(data)
b, test_Y = data_process(data1)    

import numpy as np 
train_X=np.matrix(a)[:, 4:]
test_X=np.matrix(b)[:, 4:]

from sklearn.tree import DecisionTreeClassifier
import graphviz 
import pydotplus
from IPython.display import Image
model_tree = DecisionTreeClassifier(criterion='entropy',max_depth=3, min_samples_split=100, min_samples_leaf=1000)
model_tree.fit(train_X, train_Y)
#data_feature_name= ["Inter dis", "Upper margin", "Lower margin", "Exploitation", "Row num", 
#        "Modular Ratio", "Inter spacing", "Weight", "Peak Num", "MR/IS"]
data_target_name=["A", "B", "C", "D", "E", "F", "G", "H", "I", "W", "X", "Y"]
data_feature_name= [ "Row num", "Modular Ratio", "Inter spacing", "Weight", "Peak Num", "MR/IS"]
dot_tree = tree.export_graphviz(model_tree,out_file=None,feature_names=data_feature_name, class_names=data_target_name)
graph = pydotplus.graph_from_dot_data(dot_tree)
img = Image(graph.create_png())
graph.write_png("out.png")




clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_Y)

test_pred = clf.predict(test_X)
accuracy_score(test_Y, test_pred)

"""
from matplotlib import pyplot
acc_test=np.zeros(20)
acc_train=np.zeros(20)
clf=list(range(0,20))
for i in range(0,20):
    clf[i]=tree.DecisionTreeClassifier(max_depth=i+1)
    clf[i]=clf[i].fit(train_X,train_Y)
    predict_test=clf[i].predict(test_X)
    acc_test[i]=accuracy_score(test_Y, predict_test)
    predict_train=clf[i].predict(train_X)
    acc_train[i]=accuracy_score(train_Y, predict_train)

pyplot.figure(1)
pyplot.plot(range(1,21),acc_train,label='Train data')
pyplot.plot(range(1,21),acc_test,label='Test data')
pyplot.title('Max depth VS Accuracy rate')
pyplot.xlabel('Max depth')
pyplot.ylabel('Accuracy rate')
pyplot.legend()


import graphviz 
list_er=list(acc_test)
ind=list_er.index(max(acc_test))

dot_data=tree.export_graphviz(clf[ind],out_file=None)
graph=graphviz.Source(dot_data)
graph.render("few_features")


import mglearn 
mglearn.plots.plot_animal_tree()
    

import matplotlib.pyplot as plt
figure1 = [k[1] for k in train_X]
figure2 = [k[2] for k in train_X]
figu1=[]
figu2=[]
for k in train_X:
    if k[1]>40 or k[2]>40:
        continue 
    else:
        figu1.append(k[1])
        figu2.append(k[2])
#labels = ["Intercolumnar distance", "Upper margin", "Lower margin", "Exploitation", 
#          "Row number", "Modular ratio", "Interlinear spacing", 
#          "Weight", "Peak number", "mr/is"]
labels=["Upper margin", "Lower margin"]
plt.boxplot([figu1, figu2], labels=labels, sym ="o", whis = 1.5)
plt.show()


figure7 = []
figure8 = []
for k in train_X:
    if k[7]>10 or k[8]>10:
        continue 
    else:
        figure7.append(k[7])
        figure8.append(k[8])
  
yy=[]
figure=[k[0] for k in train_X]
for i in range(len(data)-1):
        sample = data[i].split(",")
        yy.append(sample[-1])
        
# Create the plot object
_, ax = plt.subplots()

# Plot the data, set the size (s), color and transparency (alpha)
# of the points
ax.scatter(yy,figure, s = 10, color = 'r', alpha = 0.75)
# Label the axes and provide a title
ax.set_title("Scatter plot between Intercolumnar distance and copyist")
ax.set_xlabel("Copyost")
ax.set_ylabel("Intercolumnar distance")



def count_elements(seq) -> dict:
     hist = {}
     for i in seq:
         hist[i] = hist.get(i, 0) + 1
     return hist

counted=count_elements(test_pred)
true_y = [k[0] for k in test_Y]
counted_true=count_elements(true_y)
gradeGroup={"A":4287, "B":5, "C":103, "D":356, "E":1082, "F":1957, "G":439, "H":538, "I":832, "W":47, 
      "X":525, "Y":266}
gradeGroup1={"A":4286, "B":5, "C":103, "D":353, "E":1095, "F":1962, "G":447, "H":520, "I":832, "W":45, 
      "X":522, "Y":267}
xticks=["A","B","C","D","E","F","G","H","I","W","X","Y"]
plt.bar(range(12), [gradeGroup.get(xtick, 0) for xtick in xticks], align='center',yerr=0.000001)
plt.bar(range(12), [gradeGroup1.get(xtick, 0) for xtick in xticks], align='center',yerr=0.000001)
plt.xticks(range(12), xticks)
plt.xlabel('Copyists')
plt.ylabel('Frequency')
plt.title('Copyists prediction')
plt.show()

from sklearn.metrics import confusion_matrix
conmat=confusion_matrix(test_Y, test_pred)
target_names = ["A","B","C","D","E","F","G","H","I","W","X","Y"]
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Paired):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(target_names)))
    plt.xticks(xlocations, target_names, rotation=90)
    plt.yticks(xlocations, target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(conmat)

from sklearn.metrics import classification_report
target_names = ["A","B","C","D","E","F","G","H","I","W","X","Y"]
print(classification_report(true_y, test_pred, target_names=target_names))












