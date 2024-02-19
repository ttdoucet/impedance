#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cmath

uH = 1e-6
mH = 1e-3

pF = 1e-12
nF = 1e-9
uF = 1e-6

MHz = 1e6
kHz = 1e3
Hz = 1

class Port():
    def z(self, f): return 1 / self.y(f)
    def y(self, f): return 1 / self.z(f)

class Component(Port):
    def __init__(self):
        self.val = None
    def apply(self, v, f):
        self.v = [cmath.polar(vv)[0] for vv in v]
        self.phase = [180*cmath.polar(vv)[1]/np.pi for vv in v]
        self.f = f
        
class Res(Component):
    def __init__(self, ohms):
        self.val = ohms
    def z(self, f):
        return self.val * np.ones_like(f)
    
class Coil(Component):
    def __init__(self, henries):
        self.val = henries
    def z(self, f):
        return 2 * np.pi * f * self.val * 1j
    
class Cap(Component):
    def __init__(self, farads):
        self.val = farads
    def y(self, f):
        return 2 * np.pi * f * self.val * 1j

class Series(Port):
    def __init__(self, *elems):
        self.elems = elems
    
    def z(self, f):
        return sum([port.z(f) for port in self.elems])
    
    def apply(self, v, f):
        zt = self.z(f)
        for ckt in self.elems:
            ckt.apply( v * (ckt.z(f) / zt), f)
        
class Parallel(Port):
    def __init__(self, *elems):
        self.elems = elems
    
    def y(self, f):
        return sum([port.y(f) for port in self.elems])
    
    def apply(self, v, f):
        for ckt in self.elems:
            ckt.apply(v, f)

def Tee(p1, p2, p3, load):
    return Series(p1, Parallel(p2, Series(p3, load)))


def fplot(filt, source, load, f_start=1*MHz, f_end=30*MHz, f_step=1*kHz, use_kHz=False):
    """Filter response plot for filter fed by source impedance.
    
    The load is contained within the filter, the source impedance is not.
    """

    freq = np.arange(f_start, f_end, f_step)

    ref_filt = Series(source, load)
    ref_filt.apply(1.0, freq)
    ref_resp = 20*np.log10(load.v)
    
    fed = Series(source, filt)
    
    fed.apply(1.0, freq)
    response = 20 * np.log10(load.v)
        
    fig, ax = plt.subplots()
    if use_kHz:
        ax.plot(freq / kHz, response - ref_resp)
        ax.set_xlabel('Frequency (kHz)')
    else:          
        ax.plot(freq / MHz, response - ref_resp)
        ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Loss (dB)')
    ax.grid(True)


R_sp = Res(4)
sp_filt = Series(Cap(33 * uF), Coil((1.5 + 0.68) * mH), R_sp)

fplot(sp_filt, source=Res(1), load=R_sp, f_start=10*Hz, f_end=10*kHz, f_step=10*Hz, use_kHz=True)

plt.show()
