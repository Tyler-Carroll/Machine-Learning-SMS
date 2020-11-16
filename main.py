import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


#Read Data From CSV
sms = pd.read_csv("spam.csv", encoding='latin-1')
print(sms.head())


#Drop Undesired Columns, rename columns 1 and 2 to label and message
sms = sms.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
sms = sms.rename(columns={'v1' : 'label', 'v2' : 'message'})

print(sms.groupby('label').describe())

# applying a new feature of message length to see if theres anything of interest that we can find
sms['length'] = sms['message'].apply(len)
sms.head()

mpl.rcParams['patch.force_edgecolor'] = True
plt.style.use('seaborn-bright')
sms.hist(column='length', by='label', bins=50,figsize=(11,5))
# graph shows that the longer the message the more likely it is spam

text_feat = sms['message'].copy()

def process_Text(text):
    text = text.translate(str.maketrans('','', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    
    return " ".join(text)

text_feat = text_feat.apply(process_Text)

vectorizer = TfidfVectorizer("english")

features = vectorizer.fit_transform(text_feat)

features_training_Model, features_test_Model, labels_train_Model, labels_test_Model = train_test_split(features, sms['label'], test_size=0.3, random_state=111)

# importing ML algorithm libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

log_Reg = LogisticRegression(solver='liblinear', penalty="l1")
rfc = RandomForestClassifier(n_estimators=31, random_state=111)
bc = BaggingClassifier(n_estimators=9, random_state=111)
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)

clfs = { 'DT': dtc, 'LR': log_Reg, 'RF': rfc,'BgC': bc}

def train_classifier(clf, feature_train, labels_train):    
    clf.fit(feature_train, labels_train)

def predict_labels(clf, features):
    return (clf.predict(features))

pred_scores = []

for k,v in clfs.items():
    train_classifier(v, features_training_Model, labels_train_Model)
    pred = predict_labels(v,features_test_Model)
    pred_scores.append((k, [accuracy_score(labels_test_Model,pred)]))

df = pd.DataFrame.from_dict(dict(pred_scores),orient='index', columns=['Score'])
df

