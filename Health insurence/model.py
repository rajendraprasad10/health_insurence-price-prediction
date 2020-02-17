import pandas as pd
import pickle

df = pd.read_csv('insurance.csv')

# encoding technics for changing the catagorical data to numrical data
df['sex'] = pd.get_dummies(df['sex'])
df['smoker'] = pd.get_dummies(df['smoker'])
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['region']= LE.fit_transform(df['region'])

y = df['charges']
x = df.drop('charges',axis=1)


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()

#Fitting model with trainig data
rfr.fit(x,y)
# Saving model to disk
pickle.dump(rfr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2]]))