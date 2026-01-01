import os
os.environ["OMP_NUM_THREADS"] = "1"
from src.QuasilinearMaxwellProblem import QuasilinearMaxwellProblem, compute_estimators, compute_markers
from firedrake import *
from netgen.csg import *
import matplotlib.pyplot as plt


FAMILY = 'N2curl'
THETA12 = 0.6
THETA3 = 0.6
NUM_REFINE_ADAPTIVE = 5
MAX_DOFS = 5e5
NUM_REFINE_UNIFORM = 0
SOLVER_PARAMS = {}
# SOLVER_PARAMS={'ksp_rtol': 1e-8,
#                'ksp_type': 'fgmres',
#                'pc_type': 'fieldsplit',
#                'pc_mat_type': 'mumps',
#                'pc_fieldsplit_type': 'schur',
#                'pc_fieldsplit_schur_fact_type': 'full',
#                'fieldsplit_0_ksp_type': 'preonly',
#                'fieldsplit_0_ksp_rtol': 1e-16,
#                'fieldsplit_0_pc_type': 'lu',
#                'fieldsplit_0_pc_mat_type': 'mumps',
#                'fieldsplit_1_ksp_type': 'preonly',
#                'fieldsplit_1_ksp_rtol': 1e-16,
#                'fieldsplit_1_pc_type': 'none',
#                'fieldsplit_1_pc_mat_type': 'mumps',
#                }
PATH = f'outputs/example1/{FAMILY}'


def set_functions(msh):
    x, y, z = SpatialCoordinate(msh)
    ux = pi*cos(pi*x)*sin(pi*y)*sin(pi*z)
    uy = pi*cos(pi*y)*sin(pi*x)*sin(pi*z)
    uz = pi*cos(pi*z)*sin(pi*y)*sin(pi*x)
    uex = as_vector([ux, uy, uz])
    f = Constant((0, 0, 0))
    g = -3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    def alpha(u):
        return 1 -0.5/(1 + dot(curl(u), curl(u)))
    def beta(u):
        return 1.0
    return f, g, alpha, beta, uex


if __name__ == "__main__":
    cube1 = OrthoBrick(Pnt(-1,-1,0), Pnt(0,0,1))
    cube2 = OrthoBrick(Pnt(0,-1,0), Pnt(1,0,1))
    cube3 = OrthoBrick(Pnt(-1,0,0), Pnt(0,1,1))
    geo = CSGeometry()
    geo.Add(cube1 + cube2 + cube3)
    ngmsh = geo.GenerateMesh(maxh=1.0)
    msh0 = Mesh(ngmsh)

    msh0 = msh0.refine_marked_elements(
        Function(FunctionSpace(msh0, 'DG', 0)).interpolate(1.0))
    msh0 = msh0.refine_marked_elements(
        Function(FunctionSpace(msh0, 'DG', 0)).interpolate(1.0))


    # ''' UNIFORM REFINEMENT '''
    # mh = MeshHierarchy(msh0, NUM_REFINE_UNIFORM)
    # dofs_uniform = []
    # errs_uniform = []   
    # print(f"{'Iter':<6} {'DOFs':<8} {'Hcurl Error':<12}")
    # print("-" * 26)
    # for idx in range(NUM_REFINE_UNIFORM + 1):
    #     f, g, alpha, beta, uex = set_functions(mh[idx])
    #     problem = QuasilinearMaxwellProblem(mh[idx],f,g,alpha,beta,FAMILY)
    #     if problem.W.sub(0).dim() > MAX_DOFS:
    #         print(f"STOP UNIFORM REFINEMENT! CURRENT DOFS: {problem.W.sub(0).dim()}")
    #         break
    #     w = Function(problem.W)
    #     solver = problem.solver(w, SOLVER_PARAMS)
    #     solver.solve()
    #     u = w.sub(0)
    #     errs_uniform.append(norm(u - uex, 'Hcurl'))
    #     dofs_uniform.append(u.function_space().dim())
    #     print(f"{idx:<6} {dofs_uniform[-1]:<8} {errs_uniform[-1]:<12.6e}")
    # print("-" * 26)

    ''' N1curl STATS '''
    # dofs_uniform = [47, 262, 1700, 12136, 91472][2:]
    # errs_uniform = [3.332162e+00, 2.904975e+00, 1.877877e+00, 1.088275e+00, 5.779967e-01][2:]
    ''' N2curl STATS '''
    dofs_uniform = [94, 524, 3400, 24272, 182944][2:]
    errs_uniform = [2.608260e+00, 1.275112e+00, 4.498282e-01, 1.252988e-01, 3.281977e-02][2:]

    coeffs_errs_uniform = -np.polyfit(np.log(dofs_uniform), np.log(errs_uniform), 1)[0]
    print(f"Convergence rate of uniform refinement: {coeffs_errs_uniform:.6f}")


    ''' ADAPTIVE REFINEMENT '''
    msh_adaptive = msh0
    dofs_adaptive = []
    errs_adaptive = []
    noes_adaptive = []
    sols_adaptive = []

    print(f"{'Iter':<6} {'DOFs':<8} {'Hcurl Error':<12} {'Estimator':<12}")
    print("-" * 38)
    for idx in range(NUM_REFINE_ADAPTIVE + 1):

        file = VTKFile(f'{PATH}/meshes/mesh-{idx}.pvd')
        file.write(msh_adaptive)

        f, g, alpha, beta, uex = set_functions(msh_adaptive)
        problem = QuasilinearMaxwellProblem(msh_adaptive,f,g,alpha,beta,FAMILY)

        if problem.W.sub(0).dim() > MAX_DOFS:
            print(f"STOP ADAPTIVE REFINEMENT! CURRENT DOFS: {problem.W.sub(0).dim()}")
            break

        w = Function(problem.W)
        solver = problem.solver(w, SOLVER_PARAMS)
        solver.solve()
        u = w.sub(0)
        sols_adaptive.append(u)
        errs_adaptive.append(norm(u - uex, 'Hcurl'))
        dofs_adaptive.append(u.function_space().dim())

        eta1, eta2, eta3 = compute_estimators(w, problem)
        noes_adaptive.append(np.sqrt(norm(eta1)**2 + norm(eta2)**2 + norm(eta3)**2))
        print(f"{idx:<6} {dofs_adaptive[-1]:<8} {errs_adaptive[-1]:<12.6e} {noes_adaptive[-1]:<12.6e}")

        if idx < NUM_REFINE_ADAPTIVE:
            markers = compute_markers(eta1, eta2, eta3, THETA12, THETA3)
            msh_adaptive = msh_adaptive.refine_marked_elements(markers)
    print("-" * 38)

    coeffs_errs_adaptive = -np.polyfit(np.log(dofs_adaptive), np.log(errs_adaptive), 1)[0]
    coeffs_noes_adaptive = -np.polyfit(np.log(dofs_adaptive), np.log(noes_adaptive), 1)[0]
    print(f"Convergence rate of true H(curl) error: {coeffs_errs_adaptive:.6f}")
    print(f"Convergence rate of error estimator: {coeffs_noes_adaptive:.6f}")


    ''' ERROR-ANALYSIS-PLOT '''
    plt.figure(figsize=(10, 8))
    label_errs_adaptive = f"Hcurl Error of Adapative Refine (order ≈ {coeffs_errs_adaptive:.4f})"
    label_noes_adaptive = f"Error Estimator (order ≈ {coeffs_noes_adaptive:.4f})"
    label_errs_uniform = f"Hcurl Error of Uniform Refine (order ≈ {coeffs_errs_uniform:.4f})"

    plt.loglog(dofs_uniform, errs_uniform, 'bo-', linewidth=3, markersize=8, label=label_errs_uniform)
    plt.loglog(dofs_adaptive, errs_adaptive, 'ro-', linewidth=3, markersize=8, label=label_errs_adaptive)
    plt.loglog(dofs_adaptive, noes_adaptive, 'go-', linewidth=3, markersize=8, label=label_noes_adaptive)
    
    plt.xlabel('log(Degrees of Freedom)', fontsize=14)
    plt.ylabel('log(Values)', fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{PATH}/error_analysis-{FAMILY}', dpi=300)
