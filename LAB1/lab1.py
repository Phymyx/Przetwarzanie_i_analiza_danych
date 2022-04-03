import numpy as np
from numpy.lib.stride_tricks import as_strided

#TABLICE
a=np.array([1,2,3,4,5,6,7])
b=np.array([[1,2,3,4,5],[6,7,8,9,10]])

bt=np.transpose(b)

c=np.arange(start=1, stop=101, step=1)

d=np.linspace(0.0, 2.0, num=10)

e=np.arange(start=0, stop=100, step=5)


#LICZBY LOSOWE
f=np.random.randn(5,4)
f=f.round(2)
#print(f)
i=np.random.randint(1000, size=100)
#print(i)
g=np.ones([3,2])
h=np.zeros([3,2])
#print(g)
#print(h)
xd=np.random.randint(1000, size=(5,5))
random_matrix_array=np.int32(xd)
#print(random_matrix_array)

#zadanie
zadanie_a=np.random.uniform(0, 10, size=(5,5))
zadanie_b=np.int32(zadanie_a)
a_round=zadanie_a.round()
a_int=np.int32(a_round)
#print(zadanie_a)
#print(zadanie_b)
#print(a_int)

#SEKCJA DANYCH
z=np.array([[1,2,3,4,5],[6,7,8,9,10]],dtype=np.int32)
z1=z.ndim   #2 wymiary
z2=z.size   #10 elementów
#print(z1)
#print(z2)
#print(z.item((0,3)))
#print(z.item((1)))
#print(z[0])
#print(z[:,0])
macierz=np.random.randint(100,size=(20,7))
#print(macierz[:,0:4])

#OPERACJE MATEMATYCZNE I LOGICZNE
matrix_a=np.array([[0,1,2],[7,5,2],[9,1,0]])
matrix_b=np.array([[5,2,6],[2,7,4],[8,5,2]])

#print(np.add(matrix_a,matrix_b))
#print(np.multiply(matrix_a,matrix_b))
#print(matrix_a/matrix_b)
#print(np.power(matrix_a, matrix_b))

#print(matrix_a <= 4)

#print(matrix_b.diagonal().sum())

#DANE STATYSTYCZNE
#print(np.sum(matrix_b))
#print(np.min(matrix_b))
#print(np.max(matrix_b))
#print(np.std(matrix_b)) #średnie odchylenie
#print(matrix_b)
#print(np.array([[np.mean(matrix_b[0])],[np.mean(matrix_b[1])],[np.mean(matrix_b[2])]])) #średnia dla wierszy
#print(np.array([np.mean(matrix_b[:,0]),np.mean(matrix_b[:,1]),np.mean(matrix_b[:,2])])) #śrenia dla kolumn

#RZUTOWANIE WYMIARÓW
tablica=np.arange(50)
#print(tablica)
#print(tablica.reshape(10,5))
#print(tablica.resize(10,5))
#print(tablica.ravel()) #spłaszcza
tab1=np.arange(5)
tab2=np.arange(4)
tab1_new=tab1[:, np.newaxis]
suma=tab1_new+tab2
tab2_new=tab2[:, np.newaxis]  #jedna jest transponowana
suma1=tab2_new+tab1
#print(suma)
#print(suma1)


#SORTOWANIE DANYCH
tab=np.random.randn(5,5)
#print(tab)
#np.argsort(x)
tab_sort1=np.sort(tab, axis=1) ##sortowanie wierszami
tab_sort=np.sort(tab, axis=0) #sortowanie kolumnami
#print(tab_sort1)
#print(tab_sort)

#zadanie
tab_b=np.array([(1,"MZ","mazowieckie"),(2,"ZP","zachodniopomorskie"),(3,"ML","małopolskie")])
matrix_tab_b=tab_b.reshape(3,3)
matrix_sort=np.sort(matrix_tab_b[:,1], axis=0)
#print(matrix_tab_b)
#print(matrix_sort)
#print(matrix_tab_b[1,2])

#ZADANIA PODSUMOWUJĄCE
#zad1
zad1=np.random.randint(100, size=(10,5))
#print(zad1)
#print(zad1.trace())  #suma głównej przekątnej macierzy
#print(np.diag(zad1)) #wyświetlenie wartości

#zad2
zad2_1=np.random.normal(5, size=5)
zad2_2=np.random.normal(5, size=5)
#print(zad2_1)
#print(zad2_2)
#print(np.multiply(zad2_1,zad2_2))

#zad3
zad3_1=np.random.randint(100, size=10)
zad3_2=np.random.randint(100, size=10)
#print(zad3_1)
#print(zad3_2)
zadanie3_1=zad3_1.reshape(2,5)
zadanie3_2=zad3_2.reshape(2,5)
#print(zadanie3_1)
#print(zadanie3_2)
#print(np.add(zadanie3_1,zadanie3_2))

#zad4
zad4_1=np.random.randint(10, size=(4,5))
zad4_2=np.random.randint(10, size=(5,4))
zad4_2=np.reshape(zad4_2, (4, 5))
#print(zad4_1)
#print(zad4_2)
dodawanie=np.add(zad4_1,zad4_2)
#print(dodawanie)

#zad5
#print(zad4_1)
#print(zad4_2)
#print(np.multiply(zad4_1[:,3],zad4_2[:,2]))

#zad6
zad6_1=np.random.normal(5, 5, size=(2,5))
zad6_1_2=np.random.normal(5, 5, size=(2,5))
zad6_2=np.random.uniform(0, 10, size=(2,5))
zad6_2_2=np.random.uniform(0, 10, size=(2,5))
srednia1=np.mean(zad6_1)
srednia2=np.mean(zad6_1_2)
srednia3=np.mean(zad6_2)
srednia4=np.mean(zad6_2_2)
#print("średnia")
#print(srednia1)
#print(srednia2)
#print(srednia3)
#print(srednia4)
odchylenie1=np.std(zad6_1)
odchylenie2=np.std(zad6_1_2)
odchylenie3=np.std(zad6_2)
odchylenie4=np.std(zad6_2_2)
#print("odchylenie")
#print(odchylenie1)
#print(odchylenie2)
#print(odchylenie3)
#print(odchylenie4)
var1=np.var(zad6_1)
var2=np.var(zad6_1_2)
var3=np.var(zad6_2)
var4=np.var(zad6_2_2)
#print("wariancja")
#print(var1)
#print(var2)
#print(var3)
#print(var4)

#zad7
macierz_a=np.random.randint(10, size=(2,2))
macierz_b=np.random.randint(10, size=(2,2))
#print(macierz_a)
#print(macierz_b)
#print(macierz_a * macierz_b)
#print(np.dot(macierz_a, macierz_b))
#dot daje iloczyn macierzy, natomiast "*" daje iloczyn sumy nad ostatnią osią

#zad8
zad8=np.random.randint(10, size=(5,5))
#print(zad8)
strides8=zad8.strides
block = (as_strided(zad8, strides=strides8, shape=(3, 5)))
#print(block)

#zad9
zad9_a=np.array([1,2,3,4,5])
zad9_b=np.array([6,7,8,9,10])
#Układanie tablicy wertykalnie (w rzędzie).
v_stack=np.vstack((zad9_a,zad9_b))
#Układanie tablicy horyzontalnie (w kolmach).
h_stack=np.hstack((zad9_a,zad9_b))
#print(v_stack)
#print(h_stack)


#zad10
zad10 = np.arange(0, 24)
zad10 = zad10.reshape(4,6)
strides=zad10.strides

block1 = np.max(as_strided(zad10, strides=strides, shape=(2, 3)))
block2 = np.max(as_strided(zad10 + zad10[0,3], strides=strides, shape=(2, 3)))
block3 = np.max(as_strided(zad10 + zad10[2,0], strides=strides, shape=(2, 3)))
block4 = np.max(as_strided(zad10 + zad10[2,3], strides=strides, shape=(2, 3)))

print(zad10)
print(strides)
print(block1, block2, block3, block4)
