import os

for lamb in [0.005, 0.01, 0.02, 0.04, 0.08]:
    for rho in [1, 5, 10, 20]:
        for K_iter in [1, 2, 5]:
            os.system(
                f"python inverse_problem_solver_IXI_3d_reconstruction_total.py --M_iter 1 --K_iter {K_iter} --lamb {lamb} --rho {rho}"
            )
