import numpy as np
import pandas as pd
from pandas import DataFrame
import scipy.stats as sc
#from scipy.stats import entropy
from math import log
from sklearn.datasets import fetch_rcv1

rcv1 = fetch_rcv1()


def freq(nazwa):
    kol=zoo[nazwa]
    czestosc=zoo[nazwa].value_counts()
    return czestosc

def freq2(nazwa1, nazwa2):
    kol=zoo[[nazwa1,nazwa2]]
    czestosc1 = zoo[nazwa1].value_counts()
    czestosc2 = zoo[nazwa2].value_counts()
    return czestosc1, czestosc2

"""
def entropy2(nazwa):
    p_zoo=zoo[nazwa].value_counts()
    entrop=sc.entropy(p_zoo)
    return entrop
"""

def entropy(nazwa):
    p_zoo=zoo[nazwa].value_counts()
    total=0
    for p in p_zoo:
        p=p/sum(p_zoo)
        if p!=0:
            total+=p*log(p,2)
        else:
            total+=0

    total *= -1
    return total


def infogain(nazwa1,nazwa2):
    y = zoo[nazwa1].value_counts()
    x = zoo[nazwa2].value_counts()
    ent=entropy(nazwa2)
    total=0
    for i in x:
        total += sum([i])/sum(y)*ent

    gain = entropy(nazwa1) - total
    return gain





zoo=pd.read_csv('zoo.csv')
#print(zoo)

anim=freq('type')
#print(anim)

przyklad2=freq2('animal', 'eggs')
#print(przyklad2)

entropia=entropy('animal')
print(entropia)
ig=infogain('animal', 'eggs')
print(ig)

X=rcv1["data"]
Y=rcv1.target[:,87]

print(X,Y)
