from sklearn.datasets import load_iris
from sklearn import tree
from os import system
from graphviz import *
import os


def graph_render(source_data,file_name):
    graph=Source(source_data)
    graph.format='jpg'
    graph.filename=file_name
    graph.render(view=False)

def main():
    os.environ["PATH"] += os.pathsep + 'C:/Users/jhyun/Anaconda3/envs/Algorithm/Library/bin/graphviz'
    iris=load_iris()
    clf=tree.DecisionTreeClassifier()
    clf.fit(iris.data,iris.target)
    dot_data=tree.export_graphviz(clf,out_file=None,
                                  feature_names=iris.feature_names,
                                  class_names=iris.target_names,
                                  filled=True,rounded=True,
                                  special_characters=True)

    clf2=tree.DecisionTreeClassifier(criterion="entropy")
    clf2.fit(iris.data,iris.target)
    dot_data2=tree.export_graphviz(clf2,out_file=None,
                                  feature_names=iris.feature_names,
                                  class_names=iris.target_names,
                                  filled=True,rounded=True,
                                  special_characters=True)

    clf3=tree.DecisionTreeClassifier(criterion="entropy",max_depth=2)
    clf3.fit(iris.data,iris.target)
    dot_data3=tree.export_graphviz(clf3,out_file=None,
                                  feature_names=iris.feature_names,
                                  class_names=iris.target_names,
                                  filled=True,rounded=True,
                                  special_characters=True)

    graph_render(dot_data,'graph1')
    graph_render(dot_data2,'graph2')
    graph_render(dot_data3,'graph3')

if __name__ == "__main__":
    main()