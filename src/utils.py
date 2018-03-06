import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from schemes import diff

class Data(object):
    def __init__(self):
        pass

def calc_dp(Retau, nu):
    utau = Retau*nu*2.0
    tauw = utau**2
    dp = -tauw*2.0
    return dp

def load_solution_komega(dir_):
    y = np.loadtxt("%s/y"%dir_).astype(np.float)
    u = np.loadtxt("%s/u"%dir_).astype(np.float)
    k = np.loadtxt("%s/k"%dir_).astype(np.float)
    omega = np.loadtxt("%s/omega"%dir_).astype(np.float)
    if 0:
        y_201 = np.loadtxt("y_41")
        print np.shape(y_201), np.shape(u)
        y -= 1e-13
        y[0] = 0.0
        f = interp1d(y_201, u)
        u = f(y)
        f = interp1d(y_201, k)
        k = f(y)
        f = interp1d(y_201, omega)
        omega = f(y)
    else:
        pass
    return y, u, k, omega

def load_solution_ktau(dir_):
    y = np.loadtxt("%s/y"%dir_).astype(np.float)
    u = np.loadtxt("%s/u"%dir_).astype(np.float)
    k = np.loadtxt("%s/k"%dir_).astype(np.float)
    tau = np.loadtxt("%s/tau"%dir_).astype(np.float)
    return y, u, k, tau

def load_solution_stressomega(dir_):
    try:
        y = np.loadtxt("%s/y"%dir_).astype(np.float)
    except:
        y = np.loadtxt("base_solution/y").astype(np.float)
    u = np.loadtxt("%s/u"%dir_).astype(np.float)
    omega = np.loadtxt("%s/omega"%dir_).astype(np.float)
    k = np.loadtxt("%s/k"%dir_).astype(np.float)
    dudy = diff(y, u)
    nut = k/(omega + 1e-16)
    R11 = np.loadtxt("%s/R11"%dir_).astype(np.float)[:201]
    R12 = np.loadtxt("%s/R12"%dir_).astype(np.float)[:201]
    R22 = np.loadtxt("%s/R22"%dir_).astype(np.float)[:201]
    R33 = np.loadtxt("%s/R33"%dir_).astype(np.float)[:201]
    return y, u, R11, R12, R22, R33, omega

def load_solution_lrr(dir_):
    n = 0
    y = np.loadtxt("%s/y"%dir_).astype(np.float)[n:]
    #y[0] = 0.0
    u = np.loadtxt("%s/u"%dir_).astype(np.float)[n:]
    omega = np.loadtxt("%s/omega"%dir_).astype(np.float)[n:]
    k = np.loadtxt("%s/k"%dir_).astype(np.float)[n:]
    epsilon = -abs(k*omega)
    R11 = np.loadtxt("%s/R11"%dir_).astype(np.float)[n:]
    R12 = np.loadtxt("%s/R12"%dir_).astype(np.float)[n:]
    R22 = np.loadtxt("%s/R22"%dir_).astype(np.float)[n:]
    R33 = np.loadtxt("%s/R33"%dir_).astype(np.float)[n:]
    return y, u, R11, R12, R22, R33, epsilon

def load_solution_laminar(dir_):
    y = np.loadtxt("%s/y"%dir_).astype(np.float)
    u = np.loadtxt("%s/u"%dir_).astype(np.float)
    return y, u

def lowpass(y, val):
    n = np.size(y)
    fac = 0.8
    nval = np.zeros_like(val)
    for j in range(200):
        for i in range(1, n-1):
            dy = y[i] - y[i-1]
            nval[i] = fac/2*val[i-1] + (1.0 - fac)*val[i] + fac/2*val[i+1]
        val[:] = nval[:]
    return val
        
def calc_initial_condition(y, Retau, nu):
    utau = Retau*nu*2.0
    dns = load_data()[0]
    
    ydns = dns.y*0.5
    u = dns.u*utau
    R11 = -dns.ub**2*utau**2
    R12 = dns.uv*utau**2
    R22 = -dns.vb**2*utau**2
    R33 = -dns.wb**2*utau**2
    epsilon = dns.eps*utau**4/nu

    f = interp1d(ydns, u)
    u = f(y)

    f = interp1d(ydns, R11)
    R11 = f(y)
    
    f = interp1d(ydns, R12)
    R12 = f(y)

    f = interp1d(ydns, R22)
    R22 = f(y)
    
    f = interp1d(ydns, R33)
    R33 = f(y)
    
    f = interp1d(ydns, epsilon)
    epsilon = f(y)

    # filter out the peak in R11
    R11 = lowpass(y, R11)
    k = -0.5*(R11 + R22 + R33)
    omega = -epsilon/(k + 1e-8)
    return u, R11, R12, R22, R33, k, epsilon, omega
    

def load_data():
    data = np.loadtxt("data/DNSsol.dat")
    y = data[:,0]
    yp = data[:,1]
    u = data[:,2]
    ub = data[:,3]
    vb = data[:,4]
    wb = data[:,5]
    k = 0.5*(ub**2 + vb**2 + wb**2)
    uv = data[:,10]
    
    kbal = np.loadtxt("data/kbal.dat")
    eps = kbal[:,2]
    kprod = kbal[:,3]

    DNS = Data()
    DNS.y = y
    DNS.yp = yp
    DNS.u = u
    DNS.ub = ub
    DNS.vb = vb
    DNS.wb = wb
    DNS.k = k 
    DNS.uv = uv
    DNS.eps = eps
    DNS.kprod = kprod
    
    wilcox = np.loadtxt("data/wilcox_sw.dat")
    yw = wilcox[:,2]
    uw = wilcox[:,3]
    uvw = -wilcox[:,4]
    kw = wilcox[:,5]
    ew = wilcox[:,6]
    pw = wilcox[:,7]
    Wilcox = Data()
    Wilcox.y = yw
    Wilcox.u = uw
    Wilcox.k = kw
    Wilcox.e = ew
    Wilcox.p = pw
    Wilcox.uv = uvw

    wilcox = np.loadtxt("data/wilcox.dat")
    yw = wilcox[:,2]
    uw = wilcox[:,3]
    uvw = -wilcox[:,4]
    kw = wilcox[:,5]
    ew = wilcox[:,6]
    pw = wilcox[:,7]
    Wilcox_kw = Data()
    Wilcox_kw.y = yw
    Wilcox_kw.u = uw
    Wilcox_kw.k = kw
    Wilcox_kw.e = ew
    Wilcox_kw.p = pw
    Wilcox_kw.uv = uvw

    return DNS, Wilcox, Wilcox_kw
