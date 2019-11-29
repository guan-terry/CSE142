import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


with open('..\\data_train.json') as json_file:
    dataset = pd.read_json(json_file ,orient='records')
    dataset = dataset[:1000]
    #print(dataset)
    unique_name = set()
    dataset['stars1'] = 0
    dataset['useful1'] = 0
    dataset['cool1'] = 0
    dataset['funny1'] = 0
    print(dataset)
    for num, i in enumerate(dataset['text']):
        words = i.split()
        words = [re.sub('[^a-zA-Z]', "", c).lower() for c in words]
        for i in words:
            if i not in unique_name and i != 'stars' and i != 'useful' and i != 'cool' and i != 'funny':
                dataset[i] = 0
            if i == 'stars':
                dataset.at[num, 'stars1'] += 1
            if i == 'useful':
                dataset.at[num, 'useful1'] += 1
            if i == 'cool':
                dataset.at[num, 'cool1'] += 1
            if i == 'funny':
                dataset.at[num, 'funny1'] += 1
        unique_name = unique_name.union(set(words))
        print(num)
    #print(dataset)
    y = dataset.iloc[:, :1].values
    x = dataset.iloc[:, 1:len(dataset.columns)].values
    #print(y)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors = 10)
    classifier.fit(x_train, y_train.ravel())
    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))

    print(classification_report(y_test, y_pred))
