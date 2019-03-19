# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
data = pd.read_csv("Data.csv")
X = data.iloc[: , 0:-1].values
y = data.iloc[:,-1].values

# Taking Care Of Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(X[: , 1:3])
X[: , 1:3] = imputer.transform(X[: , 1:3])

# Encoding Categorial Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder_X = LabelEncoder()
X[: , 0] = label_encoder_X.fit_transform(X[: , 0])
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Spliting The Dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)