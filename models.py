import pandas as pd
import matplotlib.pyplot as plt
import math

#read
data=pd.read_csv('rockyou-withcount_small.txt', header=None, skipinitialspace=True, sep=' ', names=['counts', 'pw', 'rest_pw'])

#concatenate pws with space
data['pw'] = data['pw'] + data['rest_pw'].fillna('')
#convert to numeric
data['counts']=pd.to_numeric(data['counts'])
#add length of each pw as column
data['length']=data['pw'].str.len()

#nan entries
print('Anzahl NaN-Enträge'+str(data['pw'].isna().sum()))

print(data.describe())

#read out mean, variance=std², range=min-max, percentiles
data.hist(column='length')
#plt.show()

#print("Anzahl aller Passwörter: "+str(data['counts'].sum())+'\n')
#print("Anzahl verschiedener Passwörter: "+str(len (data['pw']))+'\n')


#exercise d)

P1 = data[(data['pw'].str.len() >=7)&(data['pw'].str.len() <=32)]
print(P1)
P1=P1[P1['pw'].str.fullmatch(r'(\w*\d+\w*[A-Z]+\w*)|(\w*[A-Z]+\w*\d+\w*)')]


#exercise f)
#possible charecters = 2*alphabet + numbers
n = 52 + 10 
#2 characters are fixed -> 1 upper letter + 1 number
fixN = 26 +10
P1['prob']= P1.apply(lambda x: x[0]/(n**(len(x[1])-2)*fixN**2), axis=1)

#exercise g)
#shannon entropy of P1
shan = -sum(P1['prob'].apply(lambda x: x*math.log(x,2) ))
print('Shannon Entropy of P1: ' +str(shan))

#exercise h)
#Guesswork
guess = 0
probs = P1['prob'].tolist()
for i in  range(len(probs)):
    guess+=(i+1)*probs[i]

print('Expected number of guesses to break 200 passwords: '+ str(200*guess))

#l=50; k=200
#alpha = l/k = 50/200
alpha=50/200

beta=0
lambdaB=probs[beta]
while(alpha>lambdaB):
    if beta>=len(probs): break
    lambdaB+=probs[beta]
    beta+=1

#workfactor alpha
alphaWork = min(beta, lambdaB)

#alpha-guesswork
Galpha = (1-lambdaB)



print (P1)
