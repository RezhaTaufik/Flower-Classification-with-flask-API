import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle 

# load file
df = pd.read_csv("iris.csv")

print(df.head())

# Pilih Variabel dependen dan independen
x = df[["Sepal_Length","Sepal_Width","Petal_Length","Petal_Width"]] # Independen
y = df[["Class"]]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=50)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Method Model
classifier = RandomForestClassifier()

# Fit Model
classifier.fit(x_train, y_train)

# File pickle
pickle.dump(classifier, open("model.pkl", "wb"))