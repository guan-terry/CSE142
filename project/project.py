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
#opens the file to trian with
with open('..\\data_train.json') as json_file:
    ds = pd.read_json(json_file ,orient='records')
    dataset = ds.iloc[:300000, ~ds.columns.isin(['date', 'text'])]
    unique_name = set()
    names = {}
	
	#parses the most common text to use as training instances
    for i in ds['text'].iloc[:300000]:
        words = i.split()
        words =[re.sub('[^a-zA-Z]', "", c).lower() for c in words]
        for i in words:
            if i in names:
                names.update({i:names.get(i) + 1})
            else:
                names.update({i:1})
 
    q = sorted(names, key=names.get, reverse = True)[:2000]
    q = [r for r in q if r not in set(stopwords.words('english'))]
    
    # Adds the most common words to the dataset to train with
    for i in q:
        if i != '' and i != 'stars' and i !='useful' and i!= 'cool' and i!= 'funny' and i not in set(stopwords.words('english')):
            dataset[i] = 0
    # Adds these variables if they ever show up so not to overwrite original stars, useful, cool, and funny
    dataset['stars1'] = 0
    dataset['useful1'] = 0
    dataset['cool1'] = 0
    dataset['funny1'] = 0
    #iterates over the train data and adds the words in if they appear
    for num, i in enumerate(ds['text'].iloc[:300000]):
        words = i.split()
        words = [re.sub('[^a-zA-Z]', "", c).lower() for c in words]
        for i in words:
            if i!= '' and i in q and i != 'stars' and i != 'useful' and i != 'cool' and i != 'funny':
                dataset.at[num,i] += 1
            if i == 'stars':
                dataset.at[num, 'stars1'] += 1
            if i == 'useful':
                dataset.at[num, 'useful1'] += 1
            if i == 'cool':
                dataset.at[num, 'cool1'] += 1
            if i == 'funny':
                dataset.at[num, 'funny1'] += 1

    #undersamples the data so data isn't weighted to 5 as heavily
	#splits the data into features and labels
    ros = RandomUnderSampler(random_state=0)
    y_train = dataset['stars'].values
    x_train = dataset.loc[:,dataset.columns != 'stars'].values
    print(dataset.loc[:,dataset.columns != 'stars'])
    x_train, y_train = ros.fit_resample(x_train, y_train)

    #Tried these classifiers
    #classifier = KNeighborsClassifier(n_neighbors = 20) #n = 20 38% 500
    #classifier = DecisionTreeClassifier(max_depth = 70) #38% with 500 words
    #classifier = Perceptron() # 47% with 500 words
    #classifier = GaussianNB() #52% 1000 words
    #classifier = SVC() #56.8 % accuracy with 1000 waords/55% 500/ 57.2% 1200 / 40000 train 10000 test 2000 words 58.0%
    #classifier = LinearSVC(multi_class='ovr') #56% 500 55% 2000 words
    #classifier = MLPClassifier() #2000 54%
    classifier = LogisticRegression(multi_class='ovr', solver = 'newton-cg') #56% 500 words 56% 2000 words 40000 train

    classifier.fit(x_train, y_train)
    #writes the classifier into a pickle file
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(classifier, handle)
    with open('words.txt', 'w') as f:
        for item in q:
            f.write("%s\n" % item)
