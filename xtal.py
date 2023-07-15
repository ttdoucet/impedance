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

