import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC




with open('..\\data_train.json') as json_file:
    ds = pd.read_json(json_file ,orient='records')
    dataset = ds.iloc[:30000, [0,1,2,3]]
    print(len(dataset))
    unique_name = set()
    names = {}
    for i in ds['text']:
        words = i.split()
        words =[re.sub('[^a-zA-Z]', "", c).lower() for c in words]
        for i in words:
            if i in names:
                names.update({i:names.get(i) + 1})
            else:
                names.update({i:1})
    q = sorted(names, key=names.get, reverse = True)[100:600]
    for i in q:
        if i != 'stars' and i !='useful' and i!= 'cool' and i!= 'funny':
            dataset[i] = 0
    dataset['stars1'] = 0
    dataset['useful1'] = 0
    dataset['cool1'] = 0
    dataset['funny1'] = 0
    print(dataset)
    for num, i in enumerate(ds.loc[:30000, ['text']]):
        words = i.split()
        words = [re.sub('[^a-zA-Z]', "", c).lower() for c in words]
        for i in words:
            if i in q and i != 'stars' and i != 'useful' and i != 'cool' and i != 'funny':
                print(i)
                dataset.at[num, i] += 1
            if i == 'stars':
                dataset.at[num, 'stars1'] += 1
            if i == 'useful':
                dataset.at[num, 'useful1'] += 1
            if i == 'cool':
                dataset.at[num, 'cool1'] += 1
            if i == 'funny':
                dataset.at[num, 'funny1'] += 1

    y = dataset.iloc[:, :1].values
    x = dataset.iloc[:, 1:].values
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2)


    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    #classifier = KNeighborsClassifier(n_neighbors = 10)
    #classifier = LogisticRegression(multi_class='multinomial', solver = 'newton-cg')
    #classifier = DecisionTreeClassifier()
    #classifier = Perceptron()
    #classifier = GaussianNB()
    classifier = SVC()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    score = classifier.score(x_test, y_test)

    print('score is: ', score)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
