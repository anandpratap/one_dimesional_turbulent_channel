import numpy as np

# test function
class TestObjective(object):
    def __init__(self):
        pass

    def objective(self, val, param):
        return sum(val**2) + sum(param**2)


class BayesianObjective(object):
    def __init__(self, val_target, param_prior, sigma_obs, sigma_prior):
        self.val_target = val_target
        self.param_prior = param_prior
        self.sigma_obs = sigma_obs
        self.sigma_prior = sigma_prior

    def objective(self, val, param):
        assert(np.size(val) == np.size(self.val_target))
        assert(np.size(param) == np.size(self.param_prior))
        J_obs = sum((val - self.val_target)**2)/self.sigma_obs**2
        J_prior = sum((param - self.param_prior)**2)/self.sigma_prior**2
        J = 0.5*(J_obs + J_prior)
        return J

    def jac_objective(self, val, param, i):
        assert(np.size(val) == np.size(self.val_target))
        assert(np.size(param) == np.size(self.param_prior))
        J_obs = val[i] - self.val_target[i]
        return J_obs

class TikhonovObjective(object):
    def __init__(self, val_target, fac, var, nvar):
        self.val_target = val_target
        self.fac = fac
        self.var = var
        self.nvar = nvar
    def objective(self, val, param):
        assert(np.size(val)/self.nvar == np.size(self.val_target))
        J_obs = sum((val[self.var::self.nvar] - self.val_target)**2)
        J_reg = sum(param**2)
        J = J_obs + J_reg*self.fac
        return J

    def jac_objective(self, val, param, i):
        assert(np.size(val) == np.size(self.val_target))
        assert(np.size(param) == np.size(self.param_prior))
        J_obs = val[i] - self.val_target[i]
        return J_obs
