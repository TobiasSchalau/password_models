import pandas as pd
import matplotlib.pyplot as plt

#read
data=pd.read_csv('rockyou-withcount_small.txt', header=None, skipinitialspace=True, sep=' ', names=['counts', 'pw', 'rest_pw'])

#test
print('Anzahl NaN-Enträge'+str(data['pw'].isna().sum()))

#concatenate pws with space
data['pw'] = data['pw'] + data['rest_pw'].fillna('')
#convert to numeric
data['counts']=pd.to_numeric(data['counts'])
#add length of each pw as column
data['length']=data['pw'].str.len()

print(data.describe())

#read out mean, variance=std², range=min-max, percentiles
data.hist(column='length')
plt.show()

print("Anzahl aller Passwörter: "+str(data['counts'].sum())+'\n')
print("Anzahl verschiedener Passwörter: "+str(len (data['pw']))+'\n')

#print(data['counts'].mean())
#print(data['counts'].var())
#print(data['counts'].median())



#print(data['length'].mean())
#print(data['length'].var())
#print(data['length'].median())


#exercise d)
#P1 = data['pw'].contains()