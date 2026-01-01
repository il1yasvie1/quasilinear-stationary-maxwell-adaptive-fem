from firedrake import *


q_degree = 16
dx = dx(metadata={'quadrature_degree': q_degree})
dS = dS(metadata={'quadrature_degree': q_degree})
ds = ds(metadata={'quadrature_degree': q_degree})


class QuasilinearMaxwellProblem:
    def __init__(self, msh, f, g, alpha, beta, family="N1curl"):
        self.msh = msh
        self.f, self.g, self.alpha, self.beta = f, g, alpha, beta
        self.family = family
        self._set_function_spaces()

    def _set_function_spaces(self):
        if self.family  == 'N1curl':
            self.V = FunctionSpace(self.msh, 'N1curl', 1)
            self.S = FunctionSpace(self.msh, 'CG', 1)
        elif self.family == 'N2curl':
            self.V = FunctionSpace(self.msh, 'N2curl', 1)
            self.S = FunctionSpace(self.msh, 'CG', 2)
        else:
            raise NotImplementedError
        self.W = self.V*self.S
        self.DG0 = FunctionSpace(self.msh, 'DG', 0)

    def solver(self, w0, solver_parameters={}):
        msh = self.msh
        W = self.W
        f, g = self.f, self.g
        alpha, beta = self.alpha, self.beta
        gamma = pow(msh.num_cells(), -1/3)

        u, p = split(w0)
        v, q = TestFunctions(W)
       
        F = (dot(alpha(u)*curl(u),curl(v))+gamma*dot(beta(u)*u,v)-gamma*dot(beta(u)*grad(p),v)+dot(beta(u)*grad(p),grad(q)))*dx-(dot(f,v)-g*q)*dx
        problem = NonlinearVariationalProblem(F, w0,
                                              bcs=[DirichletBC(W.sub(0), 0, "on_boundary"),
                                                   DirichletBC(W.sub(1), 0, "on_boundary")])
        solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
        return solver


def compute_estimators(w, problem):
    msh = problem.msh
    DG0 = problem.DG0
    u, p = split(w)
    n = FacetNormal(msh)
    h = CellDiameter(msh)

    f, g = problem.f, problem.g
    alpha, beta = problem.alpha, problem.beta
    gamma = pow(msh.num_cells(), -1/3)

    RT1 = -g + div(beta(u)*grad(p))
    RT2 = div(beta(u)*(grad(p)-u))
    RT3 = f - curl(alpha(u) * curl(u)) + gamma*beta(u)*(grad(p)-u)
    JF1 = jump(dot(beta(u)*grad(p), n))
    JF2 = jump(dot(beta(u)*(grad(p)-u), n))
    JF3 = jump(cross(alpha(u)*curl(u), n))

    phi = TestFunction(DG0)
    eta1_squared= assemble(RT1**2*phi*h**2*dx + JF1**2*avg(phi*h)*dS)
    eta2_squared= assemble(RT2**2*phi*h**2*dx + JF2**2*avg(phi*h)*dS)
    eta3_squared= assemble(dot(RT3, RT3)*phi*h**2*dx + dot(JF3, JF3)*avg(phi*h)*dS)

    eta1 = Function(DG0)
    eta1.dat.data[:] = np.sqrt(eta1_squared.dat.data[:])
    eta2 = Function(DG0)
    eta2.dat.data[:] = np.sqrt(eta2_squared.dat.data[:])
    eta3 = Function(DG0)
    eta3.dat.data[:] = np.sqrt(eta3_squared.dat.data[:])
    return eta1, eta2, eta3


def compute_markers(eta1, eta2, eta3, theta12, theta3):
    DG0 = eta1.function_space()
    eta12 = Function(DG0)
    eta12.dat.data[:] = np.sqrt(eta1.dat.data[:]**2 + eta2.dat.data[:]**2)
    eta12_Max = np.max(eta12.dat.data[:])
    eta3_Max = np.max(eta3.dat.data[:])
    markers = Function(DG0)
    markers.interpolate(conditional(eta12>=theta12*eta12_Max, 1.0, conditional(eta3>=theta3*eta3_Max, 1.0, 0.0)))
    return markers
