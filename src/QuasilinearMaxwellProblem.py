from firedrake import *


class QuasilinearMaxwellProblem:
    def __init__(self, msh, fe_type=1):
        self.msh = msh
        self.fe_type = fe_type
        self._set_function_spaces()
        self._set_functions()

    def _set_function_spaces(self, p=1):
        if self.fe_type  == 1:
            self.V = FunctionSpace(self.msh, 'N1curl', 1)
            self.S = FunctionSpace(self.msh, 'CG', 1)
        elif self.fe_type == 2:
            self.V = FunctionSpace(self.msh, 'N2curl', 1)
            self.S = FunctionSpace(self.msh, 'CG', 2)
        else:
            raise NotImplementedError
        self.W = self.V*self.S
        self.DG0 = FunctionSpace(self.msh, 'DG', 0)
        self.CGp = FunctionSpace(self.msh, 'CG', p)

    def _set_functions(self):
        pass

    def solve(self, return_solver=False):
        w = Function(self.W)
        u, p = split(w)
        v, q = TestFunctions(self.W)
        alpha,  beta = self.alpha(u), self.beta
        f, g = self.f, self.g
        gamma = self.gamma
        F = (dot(alpha*curl(u),curl(v))+gamma*dot(beta*u,v)-gamma*dot(beta*grad(p),v)+dot(beta*grad(p),grad(q)))*dx-(dot(f,v)-g*q)*dx
        problem = NonlinearVariationalProblem(F, w,
                                              bcs=[DirichletBC(self.W.sub(0), 0, "on_boundary"),
                                                   DirichletBC(self.W.sub(1), 0, "on_boundary")])
        solver = NonlinearVariationalSolver(problem, solver_parameters={})
        solver.solve()
        if return_solver:
            return w, solver
        return w
    
    def compute_error_indicator(self, w):
        u, p = split(w)
        n = FacetNormal(self.msh)
        h = CellDiameter(self.msh)

        alpha,  beta = self.alpha(u), self.beta
        f, g = self.f, self.g
        gamma = self.gamma

        RT1 = -g + div(beta*grad(p))
        RT2 = div(beta*(grad(p)-u))
        RT3 = f - curl(alpha * curl(u)) + gamma*beta*(grad(p)-u)
        JF1 = jump(dot(beta*grad(p), n))
        JF2 = jump(dot(beta*(grad(p)-u), n))
        JF3 = jump(cross(alpha*curl(u), n))

        phi = TestFunction(self.DG0)

        eta_sq1= assemble(RT1**2*phi*h**2*dx + JF1**2*avg(phi*h)*dS)
        eta_sq2= assemble(RT2**2*phi*h**2*dx + JF2**2*avg(phi*h)*dS)
        eta_sq3= assemble(dot(RT3, RT3)*phi*h**2*dx + dot(JF3, JF3)*avg(phi*h)*dS)

        eta1 = Function(self.DG0)
        eta1.dat.data[:] = np.sqrt(eta_sq1.dat.data[:])
        eta2 = Function(self.DG0)
        eta2.dat.data[:] = np.sqrt(eta_sq2.dat.data[:])
        eta3 = Function(self.DG0)
        eta3.dat.data[:] = np.sqrt(eta_sq3.dat.data[:])
        return eta1, eta2, eta3

    def mark(self, eta1, eta2, eta3, theta):
        eta12 = Function(self.DG0)
        eta12.dat.data[:] = np.sqrt(eta1.dat.data[:]**2 + eta2.dat.data[:]**2)
        eta12_Max = np.max(eta12.dat.data[:])
        eta3_Max = np.max(eta3.dat.data[:])
        mark = Function(self.DG0)
        mark.interpolate(conditional(eta12>=theta*eta12_Max, 1.0, conditional(eta3>=theta*eta3_Max, 1.0, 0.0)))
        return mark
