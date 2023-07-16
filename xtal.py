import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import ScalarFormatter
import math

uH = 1e-6
mH = 1e-3

pF = 1e-12
nF = 1e-9
uF = 1e-6

Hz = 1
MHz = 1e6
kHz = 1e3

ohm = 1
kohm = 1000
Mohm = 1e6


def reson(L, C):
     return 1 / (2 * math.pi * math.sqrt(L * C) )

def XL(f, L):
    return 2j * math.pi * f * L;

def XC(f, C):
    return -1j / (2 * math.pi * f * C)

def Series(lhs, rhs):
    return lhs + rhs

def Parallel(lhs, rhs):
    return (lhs * rhs) / (lhs + rhs)

def Xtal(f):
    s = Rm + XL(f, Lm) + XC(f, Cm)
    return Parallel(s, XC(f, Cs) )

def Xtal_imag(f):
     return Xtal(f).imag

def Xtal_real(f):
     return Xtal(f).real

def Xtal_C(f):
     X = Xtal(f).imag
     return 1 / (2 * math.pi * f * X)


# Motional
Cm = 0.05 * pF
Lm = 3 * mH
Rm = 10 * ohm
# shunt
Cs = 3 * pF;

print("Xtal")
print(f'    Motional series:')
print(f'        Lm: {Lm / mH} mH')
print(f'        Cm: {Cm / pF} pF')
print(f'        Rm: {Rm / ohm} ohm')
print(f'    Shunted by:')
print(f'        Cs: {Cs / pF} pF')
print("")

Fr = reson(Lm, Cm)
print(f'Series resonance of Lm and Cm, Fr: {Fr / MHz} MHz')
print(f'    X_Lm: {XL(Fr, Lm) / kohm} kohm')
print(f'    X_Cm: {XC(Fr, Cm) / kohm} kohm')
print("")

print('Vary frequency around the resonance of Lm and Cm and note that')
print('the Cs raises the zero-reactance frequency a very small amount,')
print('and also note that reading off Rm is not very critical with frequency.')
print('')

for offset in (-100, -10, -1, 0, 0.65, 1, 10, 100):
    print(f'Xtal @ Fr + {offset} Hz: {Xtal(Fr + offset)} ohm' )

Cp = ((0.05 * 3) / (0.05 + 3)) * pF
print('')
print(f'Cp: {Cp / pF} pF')
print(f'reson(Lm, Cp): {reson(Lm, Cp) / MHz} MHz')
print('')

def plot_it(f_start, f_end, f_step, func, capt=""):
    freq = np.arange(f_start, f_end, f_step)
    func_v = np.vectorize(func)
    y = func_v(freq)

    fig, ax = plt.subplots()
    ax.plot(freq, y)
    ax.set_title(capt)
    ax.set_xlabel("Frequency")
    ax.grid(True)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(6,6),  useOffset=False, useMathText=True)


def plot_X_and_C(f_start, f_end, f_step):
    freq = np.arange(f_start, f_end, f_step)

    func_imag = np.vectorize(Xtal_imag)
    y_imag = func_imag(freq)

    func_real = np.vectorize(Xtal_C)
    y_real = func_real(freq)

    fig, ax = plt.subplots()
    ax.plot(freq, y_imag)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Reactance")
    ax.ticklabel_format(axis='x', style='sci', scilimits=(6,6),  useOffset=False, useMathText=True)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3),  useOffset=False, useMathText=True)
    ax.set_title("R and C equivalent vs frequency")

    ax2 = ax.twinx()
    ax2.plot(freq, y_real, 'r-')
    ax2.tick_params('y', colors='r')
    ax2.set_ylabel("C equiv", color='r')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-9,-9),  useOffset=False, useMathText=True)

    ax.grid(True)

f_start = 12.9 * MHz
f_end = 13.2 * MHz
f_step = 10 * Hz

plot_it(f_start, f_end, f_step, Xtal_imag, "Reactance")
plot_it(f_start, f_end, f_step, Xtal_real, "Resistance")
plot_it(f_start, f_end, f_step, Xtal_C, "Series equiv. C")
plot_X_and_C(f_start, f_end, f_step)

plt.show()
