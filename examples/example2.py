from src.QuasilinearMaxwellProblem import QuasilinearMaxwellProblem
from firedrake import *
from src.utils import solve_adaptive, error_analysis
from netgen.csg import *


class Problem(QuasilinearMaxwellProblem):
    def __init__(self, msh, fe_type):
        super().__init__(msh, fe_type)
        
    def _set_functions(self):
        x, y, z = SpatialCoordinate(self.msh)
        r = x**2 + y**2
        h = CellDiameter(self.msh)
        
        self.f = as_vector([0, 0, 100*conditional(lt(r, 1e-3), 1, 0)]) 
        self.g = Constant(0)

        def _alpha(u):
            ncurlu = sqrt(dot(curl(u), curl(u)))
            alpha_expr = 1 - conditional(gt(x, 0), conditional(lt(x, 1), 1, 0), 0)/(4*(1 + ncurlu**2)) - conditional(gt(y, 0), conditional(lt(y, 1), 1, 0), 0)/(4*(1 + ncurlu**2))
            return Function(self.CGp).interpolate(alpha_expr)
        
        self.alpha = _alpha
        self.beta = Constant(1.0)
        self.gamma = pow(h, 1/3)


def compute_exact_solution(msh):
    x, y, z = SpatialCoordinate(msh)
    ux = pi*cos(pi*x)*sin(pi*y)*sin(pi*z)
    uy = pi*cos(pi*y)*sin(pi*x)*sin(pi*z)
    uz = pi*cos(pi*z)*sin(pi*y)*sin(pi*x)
    return as_vector([ux, uy, uz])


if __name__ == "__main__":
    cube1 = OrthoBrick(Pnt(-1,-1,0), Pnt(0,0,1))
    cube2 = OrthoBrick(Pnt(0,-1,0), Pnt(1,0,1))
    cube3 = OrthoBrick(Pnt(-1,0,0), Pnt(0,1,1))
    geo = CSGeometry()
    geo.Add(cube1 + cube2 + cube3)
    ngmsh = geo.GenerateMesh(maxh=1.0)
    msh = Mesh(ngmsh)

    prob = Problem(msh, fe_type=2)
    w, dofs, errs, noes, nits = solve_adaptive(prob, compute_exact_solution, max_iters=8, theta=0.9)
    VTKFile('./examples/outputs/example2.pvd').write(w.sub(0))
    error_analysis(dofs, errs, noes, path='./examples/outputs/example2_error_analysis.png')
