import numpy as np
from scipy.optimize import curve_fit

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.26899724779580747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7596293832300737, 1.2214165441700262, 1.044997340851686, 0.7705897033203687, 1.280988851538506, 1.4403870545862594, 1.439644250890874, 1.280013863325763, 0.9349706185969648, 0.7035165962457238, 0.914155421686925])

def func(x, b, c, d):
  return (b/(1.0+np.exp(-(c+d*x))))

popt, pcov = curve_fit(func,x,y, p0=[1.38577 , -200 , -12.598])
print popt, pcov
x_spread = x.max() - x.min()
y_spread = y.max()
x_shift = x.mean()
b_init = y_spread
d_init = -x_spread
c_init = d_init*x_shift

print [b_init, c_init, d_init]

popt, pcov = curve_fit(func,x,y)
print popt, pcov
popt, pcov = curve_fit(func,x,y, p0=[0.5, 0.5, 0.5])
print "p0=%s" % [0.5, 0.5, 0.5], popt, pcov
popt, pcov = curve_fit(func,x,y, p0=[b_init, c_init, d_init])
print "p0=%s" % [b_init, c_init, d_init], popt, pcov
