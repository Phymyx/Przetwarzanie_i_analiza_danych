import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#MANIPULACJE DANYMI
dates=pd.date_range('20200301',periods=5)
df=pd.DataFrame(np.random.randn(5,3),index=dates,columns=list('ABC'))
df.index.name='data'
#print(df)

#GENEROWANIE TABELI ZŁOŻONEJ Z LICZB LOSOWYCH I INDEKSIE 'id'
#df2=pd.DataFrame(np.random.randint(10,size=(20,3)),columns=['A','B','C'])
df2=pd.DataFrame(np.random.randint(10,size=(20,3)),columns=list('ABC'))
df2.index.name='id'
#print(df2)
#print(df2.head(2))
#print(df2.tail(3))
#print(df2.index.name)
#print(df2.columns)
#print(df2.values)
#print(df2.sample(n=5))
#print(df2['A'])
#print(df2[['A','B']])
#print(df2[['A','B']].iloc[:3])
#print(df2[['A','B']].iloc[[4]])
#print(df2[['A','B']].iloc[[0,5,6,7]])

#FUNKCJA DESCRIBE
#print(df2.describe())
df3=df2.describe()
#print(df2.describe[df2.describe()>0])
df3wieksze0=df3[df3 > 0]
#print(df3wieksze0)
#print(df3[df3 > 0])
#print(df3wieksze0['A'])
#print(np.mean(df3))
#print(df3.mean(axis=1))

#FUNKCJA CONCAT
s1 = pd.DataFrame(np.random.randint(10,size=(5,3)),columns=list('ABC'))
s2 = pd.DataFrame(np.random.randint(20,size=(5,3)),columns=list('ABC'))
a = pd.concat([s1, s2])
#print(a)
#print(np.transpose(a))

#SORTOWANIE
indeks=np.arange(5)
df4 = pd.DataFrame({'x': pd.Series([1, 2, 3, 4, 5], index=indeks), 'y': pd.Series(['a', 'b', 'a', 'b', 'b'], index=indeks)})
df4.index.name='id'
#print(df4)
sort1=df4.sort_index(axis=1, ascending=False)
#print(sort1)
sort2=df4.sort_values(by='y', ascending=False)
#print(sort2)

#GRUPOWANIE DANYCH
slownik={'Day': ['Mon','Tue','Mon','Tue','Mon'], 'Fruit': ['Apple','Apple','Banana','Banana','Apple'], 'Pound': [10,15,50,40,5], 'Profit': [20,30,25,20,10]}
dfslownik = pd.DataFrame(slownik)
#print(dfslownik)
#print(dfslownik.groupby('Day').sum())  #pogrupowanie wszystkich danych ze względu na dzień(sumuje wartości "Pund" oraz wartości "Profit" i pokazuje dla odpowiednich dni (poniedziełek i wtorek)) w tym przypadku nie ma podziału na owoce
#print(dfslownik.groupby(['Day','Fruit']).sum()) #tak jak w przypadku powyżej jednak występuje tutaj podział na owoce

#WYPELNIANIE DANYCH
df5=pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
df5.index.name='id'
#print(df5)

df5['B']=1 #zamienia wyrażenia w kolumnie B na jedynki
#print(df5)
df5.iloc[1,2]=10 #zamiana wartości w wierszu z numerem 1 oraz w kolumnie numer  2 (C) na wartość 10
#print(df5)
df5[df5!=0]=-df5 #zmiana znaków  na przeciwny
#print(df5)

#UZUPEŁNIENIE DANYCH
df5.iloc[[0, 3], 1] = np.nan #wybiera na podanej pozycji i zamienia wartość na NaN
#print(df5)
df5.fillna(0, inplace=True) #na pozycjach z NaN ustawia wartość 0
#print(df5)
df5.iloc[[0, 3], 1] = np.nan
df5=df5.replace(to_replace=np.nan,value=-9999) #podmienia NaN na wartość -9999
#print(df5)
df5.iloc[[0, 3], 1] = np.nan
#print(pd.isnull(df5)) #pokazuje wartość Boolowską dla brakujących wartości

#ZADANIA
dfzadania = pd.DataFrame({'x': pd.Series([1, 2, 3, 4, 5]), 'y': pd.Series(['a', 'b', 'a', 'b', 'b'])})
#print(dfzadania)

#zad1
dfzadania1=dfzadania.groupby('x').sum()
#print(dfzadania1)
#print(np.mean(dfzadania1['x'])) #???

#zad2
#print(dfzadania['x'].value_counts())
#print(dfzadania['y'].value_counts())

#zad3
dfzadanie3=pd.read_csv("autos.csv")
#print(dfzadanie3)

#zad4
dfzadanie4=dfzadanie3.groupby('make').sum()
#print(dfzadanie4)
#print(np.mean(dfzadanie4['city-mpg']))
#print(dfzadanie4['city-mpg'])

#zad5
#print(dfzadanie3.groupby(['make','fuel-type']).sum())

#zad6
#print(np.polyfit(dfzadanie3['city-mpg'],dfzadanie3['length'],1))
#print(np.polyfit(dfzadanie3['city-mpg'],dfzadanie3['length'],2))
#print(np.polyval(dfzadanie3[['city-mpg','length']],1))
#print(np.polyval(dfzadanie3[['city-mpg','length']],2))

#zad7
#x=dfzadanie3['city-mpg']
#y=dfzadanie3['length']
#z=stats.pearsonr(x,y)
#print(z)

#zad8

#punkty=plt.scatter(x,y, color='r')
#print(punkty)
#plt.show()

#zad9
#y=dfzadanie3['length']
#kde=stats.gaussian_kde(y)
#xs=np.linspace(140, 210, num=50)
#y1=kde(xs)
#kde.set_bandwidth(bw_method='silverman')
#y2 = kde(xs)
#kde.set_bandwidth(bw_method=kde.factor / 3.)
#y3 = kde(xs)
#fig, ax = plt.subplots()
#ax.plot(y, np.full(y.shape, 1 / (4. * y.size)), 'bo', label="Data point")
#ax.plot(xs, y1, label='Scott (default)')
#ax.plot(xs, y2, label='Silverman')
#ax.plot(xs, y3, label='Const (1/3 * Silverman)')
#ax.legend()
#plt.show()

#140 do 210
# #scipy.stats.gaussian_kde

#zad10
y=dfzadanie3['length']  #długosc samochodu od 140 do 210
wid=dfzadanie3['width'] #szerokość samochodu od 60 do 73

#kde=stats.gaussian_kde(wid)
#xs2=np.linspace(60, 73, num=50)
#w1=kde(xs2)
#kde.set_bandwidth(bw_method='silverman')
#w2 = kde(xs2)
#kde.set_bandwidth(bw_method=kde.factor / 3.)
#w3 = kde(xs2)

kde=stats.gaussian_kde(y)
xs=np.linspace(140, 210, num=50)
y1=kde(xs)
kde.set_bandwidth(bw_method='silverman')
y2 = kde(xs)
kde.set_bandwidth(bw_method=kde.factor / 3.)
y3 = kde(xs)


#fig, ax = plt.subplots(2,1)
#ax[0].plot(y, np.full(y.shape, 1 / (4. * y.size)), 'bo', label="Data point")
#ax[0].plot(xs, y1, label='Scott (default)')
#ax[0].plot(xs, y2, label='Silverman')
#ax[0].plot(xs, y3, label='Const (1/3 * Silverman)')
#ax[0].legend()
#ax[0].set_xlabel('length')

#ax[1].plot(wid, np.full(wid.shape, 1 / (4. * wid.size)), 'bo', label="Data point")
#ax[1].plot(xs2, w1, label='Scott (default)')
#ax[1].plot(xs2, w2, label='Silverman')
#ax[1].plot(xs2, w3, label='Const (1/3 * Silverman)')
#ax[1].legend()
#ax[1].set_xlabel('width')

#fig.tight_layout()
#plt.show()
# 60 do 73
#zad11
y=dfzadanie3['length']
wid=dfzadanie3['width']

xmin = wid.min()
xmax = wid.max()
ymin = y.min()
ymax = y.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([wid, y])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
cfset = ax.contourf(X, Y, Z, cmap='pink')
ax.imshow(np.rot90(Z), cmap="pink", extent=[xmin, xmax, ymin, ymax])
ax.plot(wid, y, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
#plt.savefig('zad11')
#plt.savefig('zad11.pdf')
plt.show()



