import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('rockyou-withcount_small.txt', header=None, skipinitialspace=True, sep=' ', names=['counts', 'pw', 'rest_pw'])

#test
print(data['pw'].isna().sum())

data['pw'] = data['pw'] + data['rest_pw'].fillna('')
data['counts']=pd.to_numeric(data['counts'])
data['length']=data['pw'].str.len()

print(data.describe())

data.hist(column='length')
plt.show()

print("Anzahl aller Passwörter: "+str(data['counts'].sum())+'\n')
print("Anzahl verschiedener Passwörter: "+str(len (data['pw']))+'\n')

#print(data['counts'].mean())
print(data['counts'].var())
print(data['counts'].median())

print(data['pw'].isna().sum())


#print(data['length'].mean())
print(data['length'].var())
print(data['length'].median())