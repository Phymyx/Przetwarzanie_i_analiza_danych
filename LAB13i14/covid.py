import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gompertz

def diff_fun(fun, h=1e-7):
    return lambda x: (fun(x + h) - fun(x - h)) / 2 / h

def func(t, N0, b, c):
    f = N0 * np.exp(-b * np.exp(-c * t))
    return f

url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
dt = pd.read_csv(url)
dc = dt
dd = dt
dr = dt

dc = dc.set_index(['Province/State', 'Country/Region', 'Lat', 'Long'])
dd = dd.set_index(['Province/State', 'Country/Region', 'Lat', 'Long'])
dr = dr.set_index(['Province/State', 'Country/Region', 'Lat', 'Long'])

dc = dc.stack()
dd = dd.stack()
dr = dr.stack()

dc = dc.reset_index()
dd = dd.reset_index()
dr = dr.reset_index()

dc['czas'] = pd.to_datetime(dc['level_4'])
dd['czas'] = pd.to_datetime(dd['level_4'])
dr['czas'] = pd.to_datetime(dr['level_4'])

dc['t'] = dc['czas'] - pd.Timestamp('2020/03/01')

pol = dc[dc["Country/Region"]=='Poland']

print(pol)

x = np.linspace(0,120,121)
y = func(x, )

popt, pcov = curve_fit(gompertz, x, y, p0=[1.,1.,1.])

t = np.linspace(0,120,121)
ym = pol(t, *popt)
plt.subplot(2,1,1)
plt.plot(t, ym, "g-")
plt.plot(x, y, ".")
plt.subplot(2,1,2)
plt.plot(x[1:],np.diff(y),'k.')
plt.plot(t, diff_fun(lambda x: pol(x, *popt))(t), 'r-')

