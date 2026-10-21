"""Translate saveData.m and preserve YALMIP's ordering at the MOSEK interface."""

from warnings import warn

import numpy as np
from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK


def saveData(filename, X, labels=None, format="%6.4e", flag=True):
    if not flag:
        return

    X = np.asarray(X)
    with open(filename, "w", encoding="utf-8") as f:
        if labels is not None:
            for i in range(X.shape[1]):
                f.write(str(labels[i]) + "\t")
            f.write("\n")
        for row in X:
            for value in row:
                f.write((format % value) + "\t")
            f.write("\n")


def solve(problem, **options):
    """Solve the unchanged problem, matching the original MOSEK input ordering."""
    if options.get("solver") == "MOSEK":
        options["solver"] = _MatlabOrderMOSEK(problem.variables())
    return problem.solve(**options)


class _MatlabOrderMOSEK(MOSEK):
    """Only reorder existing affine coefficients; retain CVXPY solution recovery."""

    def __init__(self, source_variables):
        super().__init__()
        self.source_variable_ids = sorted(variable.id for variable in source_variables)

    def name(self):
        return "MATLAB_ORDER_MOSEK"

    def apply(self, problem):
        data, inverse = super().apply(problem)
        if "dualized" not in data:
            raise ValueError("The MATLAB ordering adapter supports the real affine SDPs in this repository.")
        canonical = data["param_prob"]
        if set(canonical.var_id_to_col) != set(self.source_variable_ids):
            raise ValueError("The MOSEK ordering adapter requires the original decision variables.")

        # Both toolboxes pack symmetric entries in upper-triangle row order.
        # Only the order of the original Variable objects differs.
        order = np.concatenate([
            canonical.var_id_to_col[var_id] + np.arange(canonical.id_to_var[var_id].size)
            for var_id in self.source_variable_ids
        ]).astype(int)
        if not np.array_equal(np.sort(order), np.arange(len(data["b"]))):
            raise ValueError("Invalid canonical decision-variable permutation.")
        reverse_order = np.argsort(order)
        data["A"] = data["A"][order, :].tocsc()
        data["A"].sort_indices()
        data["b"] = data["b"][order]
        data["A_bar_data"] = sorted(
            [(int(reverse_order[i]), j, triples) for i, j, triples in data["A_bar_data"]],
            key=lambda item: (item[0], item[1]),
        )
        inverse["declaration_row_order"] = order
        return data, inverse

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        import mosek

        task = mosek.Task()
        try:
            local_options = dict(solver_opts)
            requested_params = dict(solver_opts.get("mosek_params", {}))
            if "mosek_params" in local_options:
                local_options["mosek_params"] = dict(requested_params)
            options = MOSEK.handle_options(task, verbose, local_options)
            # Use CVXPY's unchanged scalar-cone construction. Register PSD
            # matrices in bulk, as the original MATLAB/MOSEK interface does.
            linear_data = dict(data)
            linear_data["K_dir"] = dict(data["K_dir"], s=[])
            MOSEK._build_dualized_task(task, linear_data)
            dimensions = data["K_dir"]["s"]
            if dimensions:
                task.appendbarvars(dimensions)
                cj, cr, cc, cv = [], [], [], []
                for j, (rows, columns, values) in sorted(data["c_bar_data"]):
                    for row, column, value in zip(rows, columns, values, strict=True):
                        if value != 0:
                            cj.append(j); cr.append(row); cc.append(column); cv.append(value)
                task.putbarcblocktriplet(cj, cr, cc, cv)
                ai, aj, ar, ac, av = [], [], [], [], []
                for i, j, (rows, columns, values) in data["A_bar_data"]:
                    for row, column, value in zip(rows, columns, values, strict=True):
                        if value != 0:
                            ai.append(i); aj.append(j)
                            ar.append(row); ac.append(column); av.append(value)
                task.putbarablocktriplet(ai, aj, ar, ac, av)

            # YALMIP retains MOSEK's default; CVXPY normally disables it.
            for key, value in requested_params.items():
                if ((isinstance(key, str) and key.strip() == "MSK_IPAR_INTPNT_BASIS")
                        or key == mosek.iparam.intpnt_basis):
                    if isinstance(value, str):
                        task.putparam("MSK_IPAR_INTPNT_BASIS", value)
                    else:
                        task.putintparam(mosek.iparam.intpnt_basis, value)
                    break
            else:
                task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.always)
            if options["save_file"]:
                task.writedata(options["save_file"])
            result = task.optimize()
            if result == mosek.rescode.trm_max_time:
                warn("Optimization terminated by time limit; solution may be imprecise or absent.")
            if verbose:
                task.solutionsummary(mosek.streamtype.msg)
            return {"task": task, "solver_options": options}
        except BaseException:
            task.__exit__(None, None, None)
            raise

    def invert(self, solver_output, inverse_data):
        order = inverse_data["declaration_row_order"]
        solution = super().invert(solver_output, inverse_data)
        if solution.primal_vars is not None:
            for var_id, value in solution.primal_vars.items():
                original = np.empty_like(value)
                original[order] = value
                solution.primal_vars[var_id] = original
        return solution
