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

