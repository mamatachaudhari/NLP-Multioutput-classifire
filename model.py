
"""
Created on Tue Jun  7 12:53:24 2022

@author: mamata.w
"""

import pandas as pd
import numpy as np

df = pd.read_excel("C:\\Users\\mamata.w\\Desktop\\Python\\Consol file for Disatnce all locations.xlsx")
df.head()

##Class Distribution
import seaborn as sns
#sns.countplot(X= "Plant2", data=df)

df.dtypes
df["Plant1"].value_counts()

from sklearn.multioutput import MultiOutputClassifier
#Text cleaning


from sklearn import preprocessing

df.isna().sum()
Xfeatures =df["BP Code"]
Xfeatures=Xfeatures.fillna(0)
Ylabels = df.iloc[: ,24:33]
Ylabels = Ylabels.fillna("Unknown")

from sklearn.model_selection import train_test_split
   


from sklearn.feature_extraction.text import CountVectorizer
import pickle

cv= CountVectorizer()
X = cv.fit_transform(Xfeatures)
pickle.dump(cv, open('transfrom.pkl','wb'))

x_train,x_test,y_train,y_test = train_test_split(X,Ylabels,test_size=0.05,random_state=7)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

model = MultiOutputClassifier(RandomForestClassifier())

model.fit(X,Ylabels)


model.score(X,Ylabels)
model.score(x_train,y_train)


pred=model.predict(x_test)



pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

