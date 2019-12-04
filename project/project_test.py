import json
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

nltk.download('stopwords')

with open('..\\data_train.json') as json_file:
    ds = pd.read_json(json_file, orient = 'records')
    test_dataset = ds.iloc[40000:50001, ~ds.columns.isin(['date','text'])]
    training_words = [];
    with open('words.txt', 'r') as wordList:
        for items in wordList:
            training_words.append(items.strip())

    for i in training_words:
        i = i.strip()
        if i != ''  and i != 'stars' and i !='useful' and i!= 'cool' and i!= 'funny' and i not in set(stopwords.words('english')):
            test_dataset[i] = 0
    test_dataset['stars1'] = 0
    test_dataset['useful1'] = 0
    test_dataset['cool1'] = 0
    test_dataset['funny1'] = 0
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(test_dataset)
    for nums, i in enumerate(ds['text'].loc[40000:50000]):
        num = nums+40000
        words = i.split()
        words = [re.sub('[^a-zA-Z]', "", c).lower() for c in words]
        for i in words:
            if i != '' and i in training_words and i != 'stars' and i != 'useful' and i != 'cool' and i != 'funny':
                test_dataset.at[num, i] += 1
            if i == 'stars':
                test_dataset.at[num, 'stars1'] += 1
            if i == 'useful':
                test_dataset.at[num, 'useful1'] += 1
            if i == 'cool':
                test_dataset.at[num, 'cool1'] += 1
            if i == 'funny':
                test_dataset.at[num, 'funny1'] += 1

    with open('filename.pickle', 'rb') as handle:
        classifier = pickle.load(handle)

    y_test = test_dataset['stars'].values
    #print(test_dataset.loc[:, test_dataset.columns!= 'stars'])
    x_test = test_dataset.loc[:,test_dataset.columns != 'stars'].values


    y_pred = classifier.predict(x_test)
    #score = classifier.score(y_test, y_pred)

    #print('score is: ', score)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
