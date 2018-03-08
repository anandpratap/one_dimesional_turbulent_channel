import argparse
import numpy as np
import matplotlib.pyplot as plt
from objectives import TikhonovObjective
from laminar import LaminarEquation
from utils import calc_dp, load_solution_komega, calc_initial_condition
from schemes import diff, diff2
#import komegaf as komegaf
import sys
import adolc as ad
sys.path.append("1d_dns_data_loader")
from dnsdataloader import DNSDataLoader
from nn import NeuralNetwork

def get_var(q):
    n = np.size(q)
    ny = n/3
    u = q[0:n:3]
    k = q[1:n:3]
    omega = q[2:n:3]
    return u, k, omega

            
class KOmegaEquation(LaminarEquation):
    def __init__(self, y, u, k, omega, Retau, verbose=False, model=None):
        self.y = np.copy(y)
        ny = np.size(self.y)
        self.verbose = verbose
        self.q = np.zeros(3*ny, dtype=np.float)
        self.Retau = Retau
        self.nu = 1e-4
        self.q[0:3*ny:3] = u[:]
        self.q[1:3*ny:3] = k[:]
        self.q[2:3*ny:3] = omega[:]

        self.writedir = "."
        self.tol = 1e-11
        self.ny = ny
        self.n = self.ny*3
        self.maxiter = 10
        self.dt = 1e6
        self.force_boundary = False
        
        self.neq = 1
        self.rho = 1.0
        self.dp = calc_dp(self.Retau, self.nu)
        self.sigma_w = 0.5
        self.beta_0 = 0.0708
        self.gamma_w = 13.0/25.0
        self.sigma_k = 0.6
        self.beta_s = 0.09
        self.model = model
        if self.model == None or self.model == "linear":
            self.beta = np.ones(ny, dtype=np.float)
        elif self.model == 'nn':
            self.nn = NeuralNetwork(sizes = [1, 3, 1])
            self.beta = np.random.randn(self.nn.n)*1e-2
            self.nn.set_from_vector(self.beta)
            #self.beta = np.ones(ny, dtype=np.float)
#        self.nn = NeuralNetwork()

    def calc_momentum_residual(self, q):
        u, k, omega = get_var(q)
        y = self.y
        uy = diff(y, u)
        uyy = diff2(y, u)
        nut = k/(omega+1e-16)
        nuty = diff(y, nut)
        R = self.nu*uyy - self.dp/self.rho
        R = R + nut*uyy + nuty*uy;
        R[0] = -u[0]
        R[-1] = (1.5*u[-1] - 2.0*u[-2] + 0.5*u[-3])/(y[-1] - y[-2])
        return R

    def calc_k_residual(self, q):
        nu = self.nu
        sigma_k = self.sigma_k
        beta_s = self.beta_s
        y = self.y
        u, k, omega = get_var(q)
        uy = diff(self.y, u)
        ky = diff(self.y, k)
        kyy = diff2(self.y, k)
        omegay = diff(self.y, omega)
        R = k/omega*uy**2 - beta_s*k*omega + nu*kyy + sigma_k*(kyy*k/omega + ky*(ky/omega - k/omega**2*omegay))
        R[0] = -k[0]
        R[-1] = 1/(y[-1] - y[-2])*(1.5*k[-1] - 2*k[-2] + 0.5*k[-3])
        return R

    def calc_omega_residual(self, q):
        nu = self.nu
        sigma_w = self.sigma_w
        beta_0 = self.beta_0
        gamma_w = self.gamma_w
        y = self.y
        u, k, omega = get_var(q)
        uy = diff(self.y, u)
        ky = diff(self.y, k)
        omegay = diff(self.y, omega)
        omegayy = diff2(self.y, omega)
        if self.model == None:
            fac = self.beta
        elif self.model == "linear":
            fac = 1.0 + 2*y*y
            #fac[0] = 1.0
        elif self.model == "nn":
            #print self.beta.shape
            self.nn.set_from_vector(self.beta)
            fac = 1.0 + self.nn.veval(self.y)
            #fac[0] = 1.0
        #print fac.shape
        R = fac*gamma_w*uy**2 - beta_0*omega**2 + nu*omegayy + sigma_w*(omegay*(ky/omega - k*omegay/omega**2) + k/omega*omegayy)
        R[0] = -(omega[0] - 5000000*nu/0.005**2)
        R[-1] = 1/(y[-1] - y[-2])*(1.5*omega[-1] - 2.0*omega[-2] + 0.5*omega[-3])
        return R
   # 
   # def calc_delR_delbeta(self, q):
   #     dRdbeta = komegaf.calc_delr_delbeta(self.y.astype(np.float64), q.astype(np.float64), np.float64(self.dp), self.beta.astype(np.#float64))
    #    return dRdbeta

#    def calc_residual_jacobian(self, q):
 #       dRdqf = komegaf.calc_jacobian(self.y.astype(np.float64), q.astype(np.float64), np.float64(self.dp), self.beta.astype(np.floa#t64))
  #      return dRdqf.T

    def calc_residual(self, q, dtype=None):
        R = np.zeros_like(q)
        if dtype == ad.adouble:
            R = ad.adouble(R)
        n = self.n
        R[0::3], R[1::3], R[2::3] = self.calc_momentum_residual(q), self.calc_k_residual(q), self.calc_omega_residual(q)
        #R = komegaf.calc_residual(self.y.astype(np.float64), q.astype(np.float64), np.float64(self.dp), self.beta.astype(np.float64))
        return R

    def calc_dt(self):
        dt = np.zeros(self.n)
        for i in range(0, self.n, 3):
            dt[i] = self.dt*100000
            dt[i+1] = self.dt*10
            dt[i+2] = self.dt*1
        return dt

    def boundary(self, q):
        if self.force_boundary:
            self.plot(q)
            q[0] = 0.0
            q[1] = 0.0
            q[2] = 5000000*self.nu/0.005**2
            q[-1] = q[-4]
            q[-2] = q[-5]
            q[-3] = q[-6]

    def postprocess(self, q):
        q = q.astype(np.float64)
        n = self.n
        u, k, omega = get_var(q)
        self.utau = self.Retau*self.nu*2.0
        self.yp = self.y*self.utau/self.nu
        self.up = u/self.utau
        self.kp = k/self.utau**2
        self.omegap = omega*self.nu/self.utau**2
        
    def save(self, q):
        q = q.astype(np.float64)
        n = self.n
        u, k, omega = get_var(q)
        np.savez("solution_kom.npz", y=self.y, u=u, k=k, omega=omega)
        #np.savetxt("%s/u"%self.writedir, u)
        #np.savetxt("%s/k"%self.writedir, k)
        #np.savetxt("%s/omega"%self.writedir, omega)

    def plot(self, q):
        u, k, omega = get_var(q)
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.subplot(311)
        plt.semilogx(self.y, u, 'r-')
        plt.subplot(312)
        plt.semilogx(self.y, k, 'r-')
        plt.subplot(313)
        plt.semilogx(self.y, omega, 'r-')
        plt.pause(0.0001)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--Retau", type=float, default=550.0, required=True, help="Reynolds number.")
    parser.add_argument("--dt", type=float, default=1.0, required=True, help="Solver time step.")
    parser.add_argument("--tol", type=float, default=1e-10, required=True, help="Solver convergence tolerance.")
    parser.add_argument("--maxiter", type=int, default=10, required=True, help="Solver max iteration.")
    parser.add_argument("--force_boundary", action="store_true", help="Force boundary.")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    args = parser.parse_args()

    Retau = args.Retau
    dt = args.dt
    tol = args.tol
    maxiter = args.maxiter
    force_boundary = args.force_boundary
    verbose = args.verbose

    dirname ="base_solution"
    y, u, k, omega = load_solution_komega("solution_kom_base.npz")
    ub = u.copy()
    #ui, ki, omegai = calc_initial_condition(y, Retau, 1e-4)

    # plt.figure()
    # plt.plot(u)
    # plt.plot(ui)

    # plt.figure()
    # plt.plot(omega)
    # plt.plot(omegai)
    #plt.show()
    
    Retau = Retau
    eqn = KOmegaEquation(y, u, k, omega, Retau, verbose=verbose, model='nn')
    eqn.dt = dt
    eqn.tol = tol
    eqn.maxiter = maxiter
    eqn.force_boundary = force_boundary
    #eqn.writedir = "solution"
    eqn.solve()
    #dns, wilcox_sw, wilcox = load_data()

    ui, ki, omegai = calc_initial_condition(y, Retau, eqn.nu)
    
    loader = DNSDataLoader(Retau, y)
    data = loader.data
    print data.keys()
    plt.ioff()
    plt.figure(11)
    plt.semilogx(eqn.yp, eqn.up, 'r-', label=r'$k-\omega$')
    #plt.semilogx(data["y+"], data["u+"], 'b-', label=r'DNS')
 #   plt.semilogx(eqn.yp,ui/eqn.utau, 'g-', label=r'Init')

    #plt.semilogx(wilcox.y, wilcox.u, 'g-', label=r'Wilcox $k-\omega$')
    plt.xlabel(r"$y^+$")
    plt.ylabel(r"$u^+$")
    plt.legend(loc=2)
    plt.tight_layout()
    
    plt.figure(2)
    plt.semilogx(eqn.yp, eqn.kp, 'r-', label=r'$k-\omega$')
    plt.semilogx(data["y+"], data["k+"], 'b-', label=r'DNS')
  #  plt.semilogx(eqn.yp, ki/eqn.utau/eqn.utau, 'g-', label=r'Init')

#    plt.figure(3)
#    plt.semilogx(eqn.yp, eqn.omegap, 'r-', label=r'$k-\omega$')
#    plt.semilogx(data["y+"], data["k+"], 'b-', label=r'DNS')
#omega*self.nu/self.utau**2
#    plt.semilogx(eqn.yp,omegai/eqn.utau/eqn.utau*eqn.nu, 'g-', label=r'Init')

    #plt.semilogx(dns.yp[::5], dns.k[::5], 'bo', label=r'DNS', mfc="white")
    #plt.semilogx(wilcox.y, wilcox.k, 'g-', label=r'Wilcox $k-\omega$')
    
    plt.xlabel(r"$y^+$")
    plt.ylabel(r"$u^+$")
    plt.legend(loc=2)
    plt.tight_layout()
    eqn.save(eqn.q)
    
    #plt.show()
    yl, ul, kl, omegal = load_solution_komega("solution_kom_linear.npz")

    plt.figure(11)
    plt.semilogx(eqn.yp, ul/eqn.utau, 'b-', label=r'QUAD')

    udns = ul
    eqn.objective = TikhonovObjective(udns, fac=1e-8, var=0, nvar=3)
    dJ = eqn.calc_sensitivity()

    plt.figure()
    plt.plot(dJ)
    for i in range(100000):
        eqn.solve()
        dJ = eqn.calc_sensitivity()
        eqn.beta = eqn.beta - dJ/np.abs(dJ)*0.00001
        #eqn.beta[0] = eqn.beta[0] - dJ[0]/np.linalg.norm(dJ[0])*0.1
        J = eqn.objective.objective(eqn.q, eqn.beta)
        if i%1000:
            print i, J
        #print dJ.shape
        #print eqn.beta.shape
        # print 
        #print nn_
        #eqn.beta = eqn.beta - dJ/np.abs(dJ).max()*0.05
        
        #plt.figure()
        #plt.plot(eqn.beta)
        #plt.show()
        
    plt.figure(11)
    plt.semilogx(eqn.yp, eqn.up, 'c-', label=r'$k-\omega$')
    plt.show()

    tfac = 1.0 + 2.0*eqn.y*eqn.y
    fac = 1.0 + eqn.nn.veval(eqn.y)
    
    plot(eqn.y, tfac, "b-")
    plot(eqn.y, fac, "rx-")
    show()
    
