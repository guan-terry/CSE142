import json
import pandas as pd
import re
import nltk
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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler



nltk.download('stopwords')

with open('..\\data_train.json') as json_file:
    ds = pd.read_json(json_file ,orient='records')
    dataset = ds.iloc[:30000, ~ds.columns.isin(['date', 'text'])]
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

    q = sorted(names, key=names.get, reverse = True)[100:1000]
    for i in q:
        if i != 'stars' and i !='useful' and i!= 'cool' and i!= 'funny' and i not in  set(stopwords.words('english')):
            dataset[i] = 0
    dataset['stars1'] = 0
    dataset['useful1'] = 0
    dataset['cool1'] = 0
    dataset['funny1'] = 0
    for num, i in enumerate(ds.loc[:40000, ['text']]):
        words = i.split()
        words = [re.sub('[^a-zA-Z]', "", c).lower() for c in words]
        for i in words:
            if i in q and i != 'stars' and i != 'useful' and i != 'cool' and i != 'funny':
                dataset.at[num, i] += 1
            if i == 'stars':
                dataset.at[num, 'stars1'] += 1
            if i == 'useful':
                dataset.at[num, 'useful1'] += 1
            if i == 'cool':
                dataset.at[num, 'cool1'] += 1
            if i == 'funny':
                dataset.at[num, 'funny1'] += 1

    #goes through the test data
    test_dataset = ds.iloc[40000:50000, ~ds.columns.isin(['date', 'text'])]
    for i in q:
        if i != 'stars' and i !='useful' and i!= 'cool' and i!= 'funny' and i not in  set(stopwords.words('english')):
            test_dataset[i] = 0
    test_dataset['stars1'] = 0
    test_dataset['useful1'] = 0
    test_dataset['cool1'] = 0
    test_dataset['funny1'] = 0
    for num, i in enumerate(ds.loc[30000:40000, ['text']]):
        words = i.split()
        words = [re.sub('[^a-zA-Z]', "", c).lower() for c in words]
        for i in words:
            if i in q and i != 'stars' and i != 'useful' and i != 'cool' and i != 'funny':
                test_dataset.at[num, i] += 1
            if i == 'stars':
                test_dataset.at[num, 'stars1'] += 1
            if i == 'useful':
                test_dataset.at[num, 'useful1'] += 1
            if i == 'cool':
                test_dataset.at[num, 'cool1'] += 1
            if i == 'funny':
                test_dataset.at[num, 'funny1'] += 1

    ros = RandomUnderSampler(random_state=0)

    y_train = dataset['stars'].values
    x_train = dataset.loc[:,dataset.columns != 'stars'].values
    y_test = test_dataset['stars'].values
    x_test = test_dataset.loc[:,dataset.columns != 'stars'].values
    x_train, y_train = ros.fit_resample(x_train, y_train)
    #x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=.2)
    print(len(x_train))
    print(len(x_train[0]))

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors = 50)#n=10 24% 3 and 5 weighted 17% when undersampled
    #classifier = LogisticRegression(multi_class='multinomial', solver = 'newton-cg') #46% weigted to 5 #38% undersampled
    #classifier = DecisionTreeClassifier() #45% weigted 5 more distributed than everything else 38% when undersampled
    #classifier = Perceptron() #22% weighted to 4 undersmapled to 22
    #classifier = GaussianNB() #46% Weighted to 5 but equallyt distributed # 18% When undersampled
    #classifier = SVC() #45% - 44% 43% undersampled and 100 most common words 45 for undersampled and 500 words
    #classifier = LinearSVC(multi_class='crammer_singer') #21% - 25% 18% when undersampled
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    score = classifier.score(x_test, y_test)

    print('score is: ', score)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
