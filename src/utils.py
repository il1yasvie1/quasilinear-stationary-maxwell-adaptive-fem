from firedrake import *
import matplotlib.pyplot as plt


def solve_adaptive(problem, compute_exact_solution, max_iters, theta):
    dofs = []
    errs = []
    noes = []
    nits = []
    print(f"{'Iter':<6} {'DOFs':<8} {'Hcurl Error':<12} {'Estimator':<12} {'Number of Iterations':<12}")
    print("-" * 45)
    for iter in range(max_iters):
        w, solver = problem.solve(return_solver=True)
        dofs.append(w.sub(0).function_space().dim())
        nits.append(solver.snes.getIterationNumber())
        u_exact = compute_exact_solution(problem.msh)
        errs.append(norm(u_exact - w.sub(0), 'Hcurl'))
        eta1, eta2, eta3 = problem.compute_error_indicator(w)
        noes.append(np.sqrt(norm(eta1, 'L2')**2 + norm(eta2, 'L2')**2 + norm(eta3, 'L2')**2))
        print(f"{iter:<6} {dofs[-1]:<8} {errs[-1]:<12.6e} {noes[-1]:<12.6e} {nits[-1]:<12}")
        if iter < max_iters - 1:
            mark = problem.mark(eta1, eta2, eta3, theta)
            problem.msh = problem.msh.refine_marked_elements(mark)
            problem._set_function_spaces()
            problem._set_functions()
    return w, dofs, errs, noes, nits


def error_analysis(dofs, errs, noes, path):
    coeffs_errs = -np.polyfit(np.log(dofs), np.log(errs), 1)[0]
    coeffs_noes = -np.polyfit(np.log(dofs), np.log(noes), 1)[0]
    print(f"Convergence rate of true H(curl) error: {coeffs_errs:.2f}")
    print(f"Convergence rate of error estimator: {coeffs_noes:.2f}")

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.loglog(dofs, errs/errs[0], 'bo-', linewidth=3, markersize=8, label='True H(curl) Error')
    plt.loglog(dofs, noes/noes[0], 'ro-', linewidth=3, markersize=8, label='Error Estimator')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Relative Value\n(1.0 = 100% of initial error)')
    plt.title('Error v.s. Estimator Convergence')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(path)
