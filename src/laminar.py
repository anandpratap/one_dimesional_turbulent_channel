import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from utils import calc_dp, load_solution_laminar
from schemes import diff, diff2
from objectives import TestObjective
import adolc as ad

def calc_jacobian(*args, **kwargs):
    try:
        tag = kwargs["tag"]
    except:
        tag = 0

    try:
        sparse = kwargs["sparse"]
    except:
        sparse = True

    if sparse:
        try:
            shape = kwargs["shape"]
        except:
            raise ValueError("'shape' should be passed to calculate sparse jacobian!")

        
        options = np.array([0,0,0,0],dtype=int)
        result = ad.colpack.sparse_jac_no_repeat(tag, *args, options=options)
        nnz = result[0]
        ridx = result[1]
        cidx = result[2]
        values = result[3]
        assert nnz > 0
        jac = sp.csr_matrix((values, (ridx, cidx)), shape=shape)
        jac = jac.toarray()
    else:
        jac = ad.jacobian(tag, *args)
    return jac

class LaminarEquation(object):
    def __init__(self, y, u, Retau):
        self.y = np.copy(y)
        self.q = np.copy(u.astype(np.float))
        self.Retau = Retau

        self.verbose = True 
        self.n = np.size(y)
        self.writedir = "."
        self.maxiter = 20
        self.tol = 1e-13
        self.dt = 1e10
        self.neq = 1
        self.nu = 1e-4
        self.rho = 1.0
        self.dp = calc_dp(self.Retau, self.nu)
        self.beta = np.ones_like(u)
        self.objective = TestObjective()
        
    def calc_residual(self, q, dtype=None):
        R = np.zeros_like(q)
        if dtype == ad.adouble:
            R = ad.adouble(R)
        R[:] = self.calc_momentum_residual(q)
        return R
        
    def calc_momentum_residual(self, q):
        u = q[0:self.n]
        y = self.y
        uyy = diff2(y, u)
        R = self.beta*self.nu*uyy - self.dp/self.rho
        R[0] = -u[0]
        R[-1] = (1.5*u[-1] - 2.0*u[-2] + 0.5*u[-3])/(y[-1] - y[-2])
        return R
    
    def calc_delJ_delbeta(self, q, beta):
        n = np.size(beta)
        beta_c = beta.copy()
        beta = ad.adouble(beta)
        tag = 1
        ad.trace_on(tag)
        ad.independent(beta)
        F = self.objective.objective(q, beta)
        ad.dependent(F)
        ad.trace_off()
        beta = beta_c
        dJdbeta = calc_jacobian(beta, tag=tag, sparse=False)
        return dJdbeta

    def calc_delJ_delq(self, q, beta):
        n = np.size(q)
        q_c = q.copy()
        q = ad.adouble(q)
        tag = 2
        
        ad.trace_on(tag)
        ad.independent(q)
        F = self.objective.objective(q, beta)
        ad.dependent(F)
        ad.trace_off()
        q = q_c
        dJdq = calc_jacobian(q, tag=tag, sparse=False)
        return dJdq

    def calc_psi(self, q, beta):
        dRdq = self.calc_residual_jacobian(q).astype(np.float)
        dJdq = self.calc_delJ_delq(q, beta)
        psi = linalg.solve(dRdq.transpose(), -dJdq.transpose())
        return psi

    def calc_delR_delbeta(self, q):
        nb = np.size(self.beta)
        n = np.size(q)
        beta_c = self.beta.copy()
        self.beta = ad.adouble(self.beta)
        tag = 3
        ad.trace_on(tag)
        ad.independent(self.beta)
        R = self.calc_residual(q, dtype=ad.adouble)
        ad.dependent(R)
        ad.trace_off()
        self.beta = beta_c
        dRdbeta = calc_jacobian(self.beta, tag=tag, sparse=False)
        return dRdbeta

    def calc_sensitivity(self):
        q = self.q.astype(np.float)
        #print self.beta
        beta = self.beta.astype(np.float)
        psi = self.calc_psi(q, beta)
        delJdelbeta = self.calc_delJ_delbeta(q, beta)
        delRdelbeta = self.calc_delR_delbeta(q)
        dJdbeta = delJdelbeta + psi.transpose().dot(delRdelbeta)
        return dJdbeta.T

    def calc_sensitivity_fd(self, dbeta=1e-4):
        n = self.beta.size
        q = self.q
        beta = self.beta
        F = self.objective.objective(q, beta)
        dJdbeta = np.zeros_like(beta)
        Fb = self.objective.objective(q, beta)
        for i in range(n):
            beta[i] += dbeta
            self.solve()
            F = self.objective.objective(q, beta)
            beta[i] -= dbeta
            dJdbeta[i] = (F - Fb)/dbeta
        return dJdbeta

    
    def calc_residual_jacobian(self, q, dq=1e-25):
        n = np.size(q)
        q_c = q.copy()
        q = ad.adouble(q)
        tag = 0
        ad.trace_on(tag)
        ad.independent(q)
        R = self.calc_residual(q)
        ad.dependent(R)
        ad.trace_off()
        options = np.array([0,0,0,0],dtype=int)
        q = q_c

        dRdq = calc_jacobian(q, tag=tag, shape=(n, n))
        
        #pat = ad.sparse.jac_pat(tag, q_c, options)
        #if 1:
        #    result = ad.colpack.sparse_jac_no_repeat(tag, q, options)
        #else:
        #    result = ad.colpack.sparse_jac_repeat(tag, q, nnz, ridx, cidx, values)
            
        #nnz = result[0]
        #ridx = result[1]
        #cidx = result[2]
        #values = result[3]
        #dRdq = sp.csr_matrix((values, (ridx, cidx)), shape=(n, n))
        
        #dRdq = np.zeros([n, n], dtype=q.dtype)
        #for i in range(n):
        #    q[i] = q[i] + 1j*dq
        #    R = self.calc_residual(q)
        #    dRdq[:,i] = np.imag(R[:])/dq
        #    q[i] = q[i] - 1j*dq
        return dRdq

    def calc_dt(self):
        return self.dt*np.ones(self.n)

    def step(self, q, dt):
        R = self.calc_residual(q)
        dRdq = self.calc_residual_jacobian(q)
        dt = self.calc_dt()
        A = np.zeros_like(dRdq)
        n = self.n
        for i in range(0, n):
            A[i,i] = 1./dt[i]
        A = A - dRdq
        dq = linalg.solve(A, R)
        l2norm = np.sqrt(sum(R**2))/np.size(R)
        return dq, l2norm
        
    def boundary(self, q):
        pass

    def solve(self):
        q = np.copy(self.q)
        dt = self.dt
        for i in range(self.maxiter):
            dq, l2norm = self.step(q, dt)
            q[:] = q[:] + dq[:]
            self.boundary(q)
            if self.verbose:
                print "Iteration: %i Norm: %1.2e"%(i, l2norm)
                self.save(q)
            if l2norm < self.tol:
                self.postprocess(q)
                break
        
        self.postprocess(q)
        self.q[:] = q[:]
        if l2norm > 1e-2:
            return True
        else:
            return False

    def plot(self):
        plt.figure(1)
        plt.plot(self.y, self.q[0:self.n], 'r-')
        plt.show()

    def postprocess(self, q):
        q = q.astype(np.float64)
        n = self.n
        u = q[0:n]
        self.utau = self.Retau*self.nu*2.0
        self.yp = self.y*self.utau/self.nu
        self.up = u/self.utau
        self.uap = self.analytic_solution()/self.utau

    def save(self, q):
        q = q.astype(np.float64)
        n = self.n
        u = q[0:n]
        np.savetxt("%s/u"%self.writedir, u)
 
    def analytic_solution(self):
        return self.dp/(2*self.nu*self.rho)*(self.y**2 - self.y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Retau", type=float, default=550.0, required=True, help="Reynolds number.")
    parser.add_argument("--dt", type=float, default=1.0, required=True, help="Solver time step.")
    parser.add_argument("--tol", type=float, default=1e-10, required=True, help="Solver convergence tolerance.")
    parser.add_argument("--maxiter", type=int, default=10, required=True, help="Solver max iteration.")
    parser.add_argument("--force_boundary", action="store_true", help="Force boundary.")
    args = parser.parse_args()

    Retau = args.Retau
    dt = args.dt
    tol = args.tol
    maxiter = args.maxiter
    force_boundary = args.force_boundary

    dirname ="base_solution"
    y, u = load_solution_laminar(dirname)
    Retau = Retau
    eqn = LaminarEquation(y, u, Retau)
    eqn.dt = dt
    eqn.tol = tol
    eqn.maxiter = maxiter
    eqn.force_boundary = force_boundary
    eqn.writedir = "solution"
    eqn.solve()
    # dns = load_data()[0]
    
    plt.ioff()
    plt.figure(1)
    plt.semilogx(eqn.yp, eqn.up, 'r-', label=r'$k-\omega$')
    plt.semilogx(eqn.yp[::5], eqn.uap[::5], 'bo', label=r'Analytic', mfc="white")
    plt.xlabel(r"$y^+$")
    plt.ylabel(r"$u^+$")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()
