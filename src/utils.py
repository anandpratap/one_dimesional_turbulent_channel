import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from schemes import diff
import sys
sys.path.append("1d_dns_data_loader")
from dnsdataloader import DNSDataLoader

class Data(object):
    def __init__(self):
        pass

def calc_utau(Retau, nu):
    utau = Retau*nu*2.0
    return utau

def calc_dp(Retau, nu):
    utau = calc_utau(Retau, nu)
    tauw = utau**2
    dp = -tauw*2.0
    return dp

def load_solution_komega(filename="solution_kom.npz"):
    data = np.load(filename)
    y = data["y"]
    u = data["u"]
    k = data["k"]
    omega = data["omega"]
    return y, u, k, omega

def load_solution_laminar(filename="solution_laminar.npz"):
    data = np.load(filename)
    y = data["y"]
    u = data["u"]
    return y, u

def calc_initial_condition(y, Retau, nu):
    loader = DNSDataLoader(Retau, y)
    data = loader.data

    utau = calc_utau(Retau, nu)
    

    
    yw = data["y"]
    uw = data["u+"]*utau
    kw = data["k+"]*(utau*utau)
    epsw = data["dissip"]*utau**4/nu

    #epsilon = dns.eps*utau**4/nu

    f = interp1d(yw, uw)
    u = f(y)

    f = interp1d(yw, epsw)
    eps = f(y)

    f = interp1d(yw, kw)
    k = f(y)
    ksmall = 1e-14
    omega = -eps/(k + ksmall)
    return u, k, omega
