import numpy as np
import matplotlib.pyplot as plt


#x = np.arange(0,4*np.pi,0.1)   # start,stop,step
#y = np.sin(x)

#plt.plot(x,y)
#plt.show()

fs = 48000 #częstotliwość próbkowania
f= 200     #częstotlwosc
n=np.arange(1000)    #liczba probek #1000
l_probek=fs/f
faza = (n * 2 * np.pi * f / fs)
sinus=np.sin(faza)
plt.plot(n, sinus)
plt.xlabel('nr próbki')
plt.ylabel('amplituda')
plt.title('Sygnał sinusoidalny')
plt.show()

dt=1/fs
start=0.0
end=1.0
for x in range(l_probek):
    print(x)
