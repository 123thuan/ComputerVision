import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing

import math
import matplotlib.pyplot as plt
import numpy as np

import operator

def load_data(filename):
    d = csv.reader(open(filename))

    x = []
    y = []

    for line in d:
        x.append([float(line[0]), float(line[1])])
        y.append(int(line[2]))

    return x, y

def run():

    x, y = load_data("alturapeso.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)#tách tập dữ liệu với tỷ lệ đúng là 0,5 = 50%

    obj = KNeighborsClassifier(n_neighbors = 9)# mõi đối tượng ta xét với 9 đói tượng xung quang

    obj.fit(x_train, y_train) #chọn dữ liệu phù hợp từ tập train

    x_real_muie = []
    x_real_omi = []
    y_real_muie = []
    y_real_omi = []

    for i in range(len(x)):
        if y[i] == 1:
            x_real_muie.append(x[i][0])
            y_real_muie.append(x[i][1])
        else:
            x_real_omi.append(x[i][0])
            y_real_omi.append(x[i][1])

    plt.plot(x_real_muie, y_real_muie , marker='+', linestyle='none', markersize=5.5, color='#F073EC')
    plt.plot(x_real_omi, y_real_omi , marker='.', linestyle='none', markersize=3.5, color='#50B5EF')
        
    plt.ylabel('cân nặng')
    plt.xlabel('chiều cao')

    res = obj.predict(x_test)

    new = [1.7, 70]

    res2 = obj.predict([new])

    print(res2[0]) 

    plt.plot(new[0], new[1] , marker='s', linestyle='none', markersize=7.5, color='green')#vi trí của giá trị mới
    plt.annotate('new individual', xy=(new[0], new[1]), xytext=(1.80, 70), arrowprops=dict(facecolor='#777777', shrink=0.25, width=1.3, headwidth=8, headlength=5))#mũi tên chỉ nhân vật mới

    if res2[0] == 1:
        print('The new individial is a woman')
    else:
        print('The new individual is a man')

    accuracy = accuracy_score(y_test, res)
    print('acuracia: ', accuracy)
    plt.show()
    
    return accuracy


if __name__ == "__main__":
    
    acc = [] 
    repeat = []

    for i in range(1):
        acc.append(run())
        repeat.append(i)