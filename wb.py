import wbdata
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

date = (datetime.datetime(2016, 1, 1),datetime.datetime(2016, 2, 1))
date1 = (datetime.datetime(2017, 1, 1),datetime.datetime(2017, 2, 1))
date2=(datetime.datetime(2012, 1, 1),datetime.datetime(2012, 2, 1))
#set up the countries I want

#set up the indicator I want (just build up the dict if you want more than one)

#grab indicators above for countires above and load into data frame
GNP= {'NY.GNP.PCAP.CD':'GNI per Capita'}
GNP_2016 = wbdata.get_dataframe(GNP, country="all", data_date=date,convert_date=False)

life = {'SP.DYN.LE00.IN':'life'}
life_2016 = wbdata.get_dataframe(life, country="all", data_date=date,convert_date=False)

ferility={'SP.DYN.TFRT.IN':'Total Fertility Rate'}
fer_2016 = wbdata.get_dataframe(ferility, country="all", data_date=date,convert_date=False)


population1={'SP.POP.TOTL':'Population_2016'}
population2={'SP.POP.TOTL':'Population_2017'}
pop_2016= wbdata.get_dataframe(population1, country="all", data_date=date,convert_date=False)
pop_2017= wbdata.get_dataframe(population2, country="all", data_date=date1,convert_date=False)



migration={'SM.POP.NETM':'Migration'}
migration_2017=wbdata.get_dataframe(migration, country="all", data_date=date2,convert_date=False)
#df is "pivoted", pandas' unstack fucntion helps reshape it into something plottable
result = pd.concat([GNP_2016, fer_2016,pop_2016,pop_2017,migration_2017,life_2016], axis=1, sort=False)
result1=result.dropna()

result1['pop_increase']=(result1['Population_2017']-result1['Population_2016'])/result1['Population_2017']
result1['migr/pop']=(result1['Migration'])/result1['Population_2017']

X1=result1.iloc[:,0:2].values
X2=result1.iloc[:,5:8].values
#remove nan
X = np.concatenate([X1,X2],axis=1)
X_1 = np.concatenate([X1,X2],axis=1)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_1=sc_X.fit_transform(X_1)

from sklearn.cluster import KMeans
wcss=[]
for i in range (1,14):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,14),wcss)
plt.show()

kmeans = KMeans(n_clusters=6,init = 'k-means++', max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X_1)
result1['scale']=y_kmeans
#gnp pop_increase
plt.xlabel('gnp')
plt.ylabel('pop increase')
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,3],s=100,c='red')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,3],s=100,c='blue')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,3],s=100,c='green')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,3],s=100,c='black')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,3],s=100,c='purple')
plt.scatter(X[y_kmeans==5,0],X[y_kmeans==5,3],s=100,c='cyan')

#GNP  ferity
plt.xlabel('gnp')
plt.ylabel('feriliy')
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='black')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='purple')
plt.scatter(X[y_kmeans==5,0],X[y_kmeans==5,1],s=100,c='cyan')

#GNP  migration
plt.xlabel('gnp')
plt.ylabel('migration')
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,4],s=100,c='red')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,4],s=100,c='blue')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,4],s=100,c='green')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,4],s=100,c='black')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,4],s=100,c='purple')
plt.scatter(X[y_kmeans==5,0],X[y_kmeans==5,4],s=100,c='cyan')


#migration  pop_increase
plt.xlabel('migration')
plt.ylabel('pop increase')
plt.scatter(X[y_kmeans==0,4],X[y_kmeans==0,3],s=100,c='red')
plt.scatter(X[y_kmeans==1,4],X[y_kmeans==1,3],s=100,c='blue')
plt.scatter(X[y_kmeans==2,4],X[y_kmeans==2,3],s=100,c='green')
plt.scatter(X[y_kmeans==3,4],X[y_kmeans==3,3],s=100,c='black')
plt.scatter(X[y_kmeans==4,4],X[y_kmeans==4,3],s=100,c='purple')
plt.scatter(X[y_kmeans==5,4],X[y_kmeans==5,3],s=100,c='cyan')

#ferility pop increase
plt.xlabel('ferility')
plt.ylabel('pop increase')
plt.scatter(X[y_kmeans==0,1],X[y_kmeans==0,3],s=100,c='red')
plt.scatter(X[y_kmeans==1,1],X[y_kmeans==1,3],s=100,c='blue')
plt.scatter(X[y_kmeans==2,1],X[y_kmeans==2,3],s=100,c='green')
plt.scatter(X[y_kmeans==3,1],X[y_kmeans==3,3],s=100,c='black')
plt.scatter(X[y_kmeans==4,1],X[y_kmeans==4,3],s=100,c='purple')
plt.scatter(X[y_kmeans==5,1],X[y_kmeans==5,3],s=100,c='cyan')

plt.xlabel('gnp')
plt.ylabel('life time')
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,2],s=100,c='red')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,2],s=100,c='blue')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,2],s=100,c='green')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,2],s=100,c='black')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,2],s=100,c='purple')
plt.scatter(X[y_kmeans==5,0],X[y_kmeans==5,2],s=100,c='cyan')
