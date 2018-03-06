import numpy as np

def diff(y, u):
    deta = 1.0
    ny = np.size(y)
    up = np.zeros(ny, dtype = u.dtype)

    up[1:ny-1] = (u[2:ny] - u[0:ny-2])/(y[2:ny] - y[0:ny-2])
    up[0] = (u[1] - u[0])/(y[1] - y[0])
    up[-1] = (u[-1] - u[-2])/(y[-1] - y[-2])

    return up

def diff2(y, u):
    deta = 1.0
    ny = np.size(y)
    y_eta = np.zeros(ny, dtype = u.dtype)
    u_eta2 = np.zeros(ny, dtype = u.dtype)
    y_eta2 = np.zeros(ny, dtype = u.dtype)

    u_y = diff(y, u)

    y_eta2[1:ny-1] = (y[2:ny] - 2.*y[1:ny-1] + y[0:ny-2])/deta**2
    u_eta2[1:ny-1] = (u[2:ny] - 2.*u[1:ny-1] + u[0:ny-2])/deta**2
    y_eta[1:ny-1] = (y[2:ny] - y[0:ny-2])/(2.*deta)

    y_eta2[0] = (y[0] - 2.0*y[1] + y[2])/deta**2
    u_eta2[0] = (u[0] - 2.0*u[1] + u[2])/deta**2
    y_eta[0] = (y[1] - y[0])/deta

    
    y_eta2[-1] = (y[-1] - 2.0*y[-2] + y[-3])/deta**2
    u_eta2[-1] = (u[-1] - 2.0*u[-2] + u[-3])/deta**2
    y_eta[-1] = (y[-1] - y[-2])/deta
    
    u_yy = -(u_y*y_eta2 - u_eta2)/(y_eta**2 + 1.e-100)

    return u_yy


def diff2(y, u):
    n = np.size(u)
    upp = np.zeros(n, dtype=u.dtype)
    h = y[1:n-1]-y[0:n-2]
    r = (y[2:n]-y[1:n-1])/h
    upp[1:n-1] = (2*u[0:n-2]*r + u[1:n-1]*(-2*r-2.) + 2*u[2:n]) /(h**2*(r**2+r))
    return upp
