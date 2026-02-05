
# koopman_mpc_pdp_adjoint.py
# Based on user's main_mpc_pdp_cost_learning_corrected.py
# Key fix: add outer-chain term d x / d theta via policy sensitivity pi_x and true dynamics adjoint.

from casadi import *
import casadi as ca
import numpy as np
import logging
import time
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directory (next to this script)
from pathlib import Path
output_dir = Path(__file__).resolve().parent

# ==============================================================================
# Koopman EDMDc (unchanged)
# ==============================================================================
class Koopman_SysID:
    def __init__(self, n_state, n_control, n_observables, phi_func, C_matrix=None, project_name='Koopman EDMDc'):
        self.project_name = project_name
        self.n_state = n_state
        self.n_control = n_control
        self.n_observables = n_observables
        self.phi_func = phi_func

        if C_matrix is None:
            logger.info("No C_matrix provided, assuming C = [I | 0].")
            self.C = np.zeros((n_state, n_observables))
            if n_state <= n_observables:
                self.C[:, :n_state] = np.eye(n_state)
            else:
                raise ValueError("Provide C_matrix when n_state > n_observables.")
        else:
            assert isinstance(C_matrix, np.ndarray)
            assert C_matrix.shape == (n_state, n_observables)
            self.C = C_matrix

        self.A = None
        self.B = None
        self.is_fitted_ = False

        # Validate phi
        test_x = np.zeros(self.n_state)
        test_z = self.phi_func(test_x)
        if test_z.shape != (self.n_observables,):
            raise ValueError("phi_func output dim mismatch")

    def fit(self, state_traj_list, control_traj_list, lambda_reg=1e-6):
        assert len(state_traj_list) == len(control_traj_list)
        Z_curr, Z_next, U_curr = [], [], []
        total_points = 0

        for states, controls in zip(state_traj_list, control_traj_list):
            if np.any(np.isnan(states)) or np.any(np.isinf(states)) or np.any(np.isnan(controls)) or np.any(np.isinf(controls)):
                continue
            horizon = controls.shape[0]
            if states.shape[0] != horizon + 1:
                continue
            for t in range(horizon):
                x_k = states[t, :]
                x_k_next = states[t+1, :]
                u_k = controls[t, :]
                z_k = self.phi_func(x_k)
                z_k_next = self.phi_func(x_k_next)
                if np.any(np.isnan(z_k)) or np.any(np.isinf(z_k)) or np.any(np.isnan(z_k_next)) or np.any(np.isinf(z_k_next)):
                    continue
                Z_curr.append(z_k); Z_next.append(z_k_next); U_curr.append(u_k)
                total_points += 1

        if total_points == 0:
            raise ValueError("No valid points for Koopman fit")

        Z_curr_mat = np.array(Z_curr).T
        Z_next_mat = np.array(Z_next).T
        U_curr_mat = np.array(U_curr).T

        Gamma = np.vstack([Z_curr_mat, U_curr_mat]).astype(np.float64)
        GGT = Gamma @ Gamma.T + lambda_reg * np.eye(self.n_observables + self.n_control)
        ZG = Z_next_mat.astype(np.float64) @ Gamma.T
        try:
            AB_T = np.linalg.solve(GGT, ZG.T)
            AB = AB_T.T
        except np.linalg.LinAlgError:
            AB = ZG @ np.linalg.pinv(GGT)

        self.A = AB[:, :self.n_observables]
        self.B = AB[:, self.n_observables:]
        self.is_fitted_ = True
        return self

    def get_model(self):
        if not self.is_fitted_:
            return None, None, self.C, self.phi_func, self.n_observables
        return self.A, self.B, self.C, self.phi_func, self.n_observables

# ==============================================================================
# OCSys (same structure; Koopman integrated)
# ==============================================================================
class OCSys:
    def __init__(self, project_name="my optimal control system"):
        self.project_name = project_name
        self.state_orig_sym = None
        self.n_state_orig = None
        self.control = None
        self.n_control = None
        self.auxvar = None
        self.n_auxvar = None

        self.state = None
        self.n_state = None

        self._state_orig_lb = []
        self._state_orig_ub = []
        self.state_lb = []
        self.state_ub = []
        self.control_lb = []
        self.control_ub = []

        self.dyn = None
        self.dyn_fn = None
        self.path_cost = None
        self.path_cost_fn = None
        self.final_cost = None
        self.final_cost_fn = None

        self.using_koopman = False
        self.A_np = None
        self.B_np = None
        self.C_np = None
        self.phi_func_np = None
        self.n_lifted_state = None
        self.A_casadi = None
        self.B_casadi = None
        self.C_casadi = None

        self._pmp_diff_done = False

    def setAuxvarVariable(self, auxvar_sym):
        if auxvar_sym is None:
            self.auxvar = SX.sym('p_dummy')
        elif isinstance(auxvar_sym, (ca.SX, ca.MX)):
            self.auxvar = auxvar_sym
        else:
            raise TypeError("auxvar_sym must be None or CasADi SX/MX.")
        self.n_auxvar = self.auxvar.numel()
        self._pmp_diff_done = False

    def setStateVariable(self, state_orig_sym, state_lb=[], state_ub=[]):
        assert isinstance(state_orig_sym, (ca.SX, ca.MX))
        if self.using_koopman:
            self.using_koopman = False
        self.state_orig_sym = state_orig_sym
        self.n_state_orig = self.state_orig_sym.numel()

        self._state_orig_lb = state_lb if len(state_lb) == self.n_state_orig else self.n_state_orig * [-1e20]
        self._state_orig_ub = state_ub if len(state_ub) == self.n_state_orig else self.n_state_orig * [1e20]

        self.state = self.state_orig_sym
        self.n_state = self.n_state_orig
        self.state_lb = self._state_orig_lb
        self.state_ub = self._state_orig_ub
        self._pmp_diff_done = False

    def setControlVariable(self, control_sym, control_lb=[], control_ub=[]):
        assert isinstance(control_sym, (ca.SX, ca.MX))
        self.control = control_sym
        self.n_control = self.control.numel()
        self.control_lb = control_lb if len(control_lb) == self.n_control else self.n_control * [-1e20]
        self.control_ub = control_ub if len(control_ub) == self.n_control else self.n_control * [1e20]
        self._pmp_diff_done = False

    def setKoopmanModel(self, A_np, B_np, C_np, phi_func_np, n_lifted):
        if self.state_orig_sym is None or self.control is None:
            raise RuntimeError("Call setStateVariable and setControlVariable first.")
        assert isinstance(A_np, np.ndarray) and A_np.shape == (n_lifted, n_lifted)
        assert isinstance(B_np, np.ndarray) and B_np.shape == (n_lifted, self.n_control)
        assert isinstance(C_np, np.ndarray) and C_np.shape == (self.n_state_orig, n_lifted)
        self.A_np, self.B_np, self.C_np, self.phi_func_np = A_np, B_np, C_np, phi_func_np
        self.n_lifted_state = n_lifted
        self.A_casadi = SX(A_np); self.B_casadi = SX(B_np); self.C_casadi = SX(C_np)
        self.state = SX.sym('z', self.n_lifted_state)
        self.n_state = self.n_lifted_state
        self.state_lb = self.n_state * [-1e20]
        self.state_ub = self.n_state * [1e20]
        self.using_koopman = True
        self._pmp_diff_done = False

    def setDyn(self, ode_expr=None):
        if self.state is None or self.control is None:
            raise RuntimeError("Call setStateVariable and setControlVariable first.")
        if self.auxvar is None:
            self.setAuxvarVariable(None)

        if self.using_koopman:
            self.dyn = mtimes(self.A_casadi, self.state) + mtimes(self.B_casadi, self.control)
            self.dyn_fn = ca.Function('koopman_dynamics', [self.state, self.control, self.auxvar], [self.dyn],
                                      ['z', 'u', 'p'], ['z_next'])
        else:
            if ode_expr is None:
                raise ValueError("ode_expr required if not in Koopman mode.")
            self.dyn = ode_expr
            self.dyn_fn = ca.Function('dynamics', [self.state, self.control, self.auxvar], [self.dyn],
                                      ['x', 'u', 'p'], ['x_next'])
        self._pmp_diff_done = False

    def _create_cost_func(self, cost_lambda, cost_type="path"):
        if self.state is None or self.control is None or self.auxvar is None:
            raise RuntimeError("Set vars first.")
        if self.state_orig_sym is None:
            raise RuntimeError("Need state_orig_sym.")

        if cost_type == "path":
            lambda_args_sym = [self.state_orig_sym, self.control, self.auxvar]
            input_vars_fn = [self.state, self.control, self.auxvar]
            func_name_prefix = "path_cost"
            input_names_fn = ['state', 'u', 'p']
            output_name_fn = 'path_cost_val'
        else:
            lambda_args_sym = [self.state_orig_sym, self.auxvar]
            input_vars_fn = [self.state, self.auxvar]
            func_name_prefix = "final_cost"
            input_names_fn = ['state', 'p']
            output_name_fn = 'final_cost_val'

        cost_expr_orig_sym = cost_lambda(*lambda_args_sym)
        if self.using_koopman:
            x_reconstructed = mtimes(self.C_casadi, self.state)
            cost_expr = substitute(cost_expr_orig_sym, self.state_orig_sym, x_reconstructed)
            func_name = f"koopman_{func_name_prefix}_transformed"
        else:
            cost_expr = cost_expr_orig_sym
            func_name = f"original_{func_name_prefix}"

        cost_fn = ca.Function(func_name, input_vars_fn, [cost_expr], input_names_fn, [output_name_fn])
        return cost_expr, cost_fn

    def setPathCost(self, path_cost_lambda):
        self.path_cost, self.path_cost_fn = self._create_cost_func(path_cost_lambda, "path")
        self._pmp_diff_done = False

    def setFinalCost(self, final_cost_lambda):
        self.final_cost, self.final_cost_fn = self._create_cost_func(final_cost_lambda, "final")
        self._pmp_diff_done = False

    def ocSolver(self, ini_state_orig, horizon, auxvar_value=None, print_level=0, costate_option=0):
        assert self.dyn_fn is not None and self.path_cost_fn is not None and self.final_cost_fn is not None
        assert horizon > 0
        ini_state_orig_np = np.array(ini_state_orig).flatten()
        assert ini_state_orig_np.shape == (self.n_state_orig,)

        if auxvar_value is None:
            if self.n_auxvar > 0:
                raise ValueError("auxvar_value must be provided if n_auxvar > 0.")
            auxvar_value_np = np.array([])
        else:
            auxvar_value_np = np.array(auxvar_value).flatten()
            if self.n_auxvar > 0 and auxvar_value_np.shape != (self.n_auxvar,):
                raise ValueError("auxvar_value dim mismatch")

        if self.using_koopman:
            ini_state_lifted_np = self.phi_func_np(ini_state_orig_np)
            ini_state_for_nlp = ini_state_lifted_np.tolist()
            current_n_state_nlp = self.n_lifted_state
            state_lb_nlp = self.state_lb
            state_ub_nlp = self.state_ub
        else:
            ini_state_for_nlp = ini_state_orig_np.tolist()
            current_n_state_nlp = self.n_state_orig
            state_lb_nlp = self.state_lb
            state_ub_nlp = self.state_ub

        w, w0, lbw, ubw = [], [], [], []
        g, lbg, ubg = [], [], []
        J = 0

        Xk = MX.sym('X0', current_n_state_nlp)
        w += [Xk]
        lbw += ini_state_for_nlp; ubw += ini_state_for_nlp; w0 += ini_state_for_nlp

        aux_arg = auxvar_value_np if self.n_auxvar > 0 else SX()

        for k in range(horizon):
            Uk = MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.control_lb; ubw += self.control_ub
            w0 += [0.0] * self.n_control

            Xk_next_dyn = self.dyn_fn(Xk, Uk, aux_arg)
            J += self.path_cost_fn(Xk, Uk, aux_arg)

            Xk_next_var = MX.sym('X_' + str(k+1), current_n_state_nlp)
            w += [Xk_next_var]
            lbw += state_lb_nlp; ubw += state_ub_nlp
            w0 += ini_state_for_nlp  # simple guess

            g += [Xk_next_dyn - Xk_next_var]
            lbg += current_n_state_nlp * [0]; ubg += current_n_state_nlp * [0]
            Xk = Xk_next_var

        J += self.final_cost_fn(Xk, aux_arg)

        nlp_prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        ipopt_opts = {
            'ipopt.print_level': print_level,
            'ipopt.sb': 'yes',
            'print_time': bool(print_level > 0),
            'ipopt.max_iter': 80,
            'ipopt.tol': 1e-5
        }
        solver = nlpsol('solver', 'ipopt', nlp_prob, ipopt_opts)

        try:
            sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            w_opt_flat = sol['x'].full().flatten()
            cost_opt = float(sol['f'])
            lam_g_flat = sol['lam_g'].full().flatten()
            solver_stats = solver.stats()
            success = bool(solver_stats.get('success', False))
        except Exception as e:
            logger.error(f"NLP solve error: {e}")
            return {"success": False, "state_traj_opt": None, "control_traj_opt": None,
                    "lifted_state_traj_opt": None, "costate_traj_opt": None, "cost": float('inf'),
                    "auxvar_value": auxvar_value_np, "solver_stats": {}}

        state_traj_raw = np.zeros((horizon + 1, current_n_state_nlp))
        control_traj_opt = np.zeros((horizon, self.n_control))
        state_traj_raw[0, :] = w_opt_flat[:current_n_state_nlp]
        offset = current_n_state_nlp
        for k in range(horizon):
            control_traj_opt[k, :] = w_opt_flat[offset:offset+self.n_control]
            offset += self.n_control
            state_traj_raw[k+1, :] = w_opt_flat[offset:offset+current_n_state_nlp]
            offset += current_n_state_nlp

        opt_sol = {
            "control_traj_opt": control_traj_opt if success else None,
            "auxvar_value": auxvar_value_np,
            "time": np.arange(horizon+1) if success else None,
            "horizon": horizon,
            "cost": cost_opt if success else float('inf'),
            "success": success,
            "solver_stats": solver_stats,
            "costate_traj_opt": None
        }

        if success:
            if self.using_koopman:
                opt_sol["lifted_state_traj_opt"] = state_traj_raw
                opt_sol["state_traj_opt"] = (self.C_np @ state_traj_raw.T).T
            else:
                opt_sol["state_traj_opt"] = state_traj_raw
                opt_sol["lifted_state_traj_opt"] = None

            # lam_g corresponds to defect constraints, reshape to (horizon, n_state_nlp) -> lambda_1..lambda_H
            opt_sol["costate_traj_opt"] = np.reshape(lam_g_flat, (horizon, current_n_state_nlp))

        return opt_sol

    def diffPMP(self):
        if self.state is None or self.control is None or self.auxvar is None or \
           self.dyn is None or self.path_cost is None or self.final_cost is None:
            raise RuntimeError("Cannot diff PMP. Set variables/dyn/costs first.")

        self.costate = SX.sym('lambda', self.n_state)

        self.path_Hamil = self.path_cost + dot(self.costate, self.dyn)
        self.final_Hamil = self.final_cost

        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = ca.Function('dfx', [self.state, self.control, self.auxvar], [self.dfx],
                                  ['x', 'u', 'p'], ['dfdx'])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = ca.Function('dfu', [self.state, self.control, self.auxvar], [self.dfu],
                                  ['x', 'u', 'p'], ['dfdu'])
        self.dfe = jacobian(self.dyn, self.auxvar) if self.n_auxvar > 0 else SX.zeros(self.n_state, 0)
        self.dfe_fn = ca.Function('dfe', [self.state, self.control, self.auxvar], [self.dfe],
                                  ['x', 'u', 'p'], ['dfdp'])

        self.dHx = jacobian(self.path_Hamil, self.state).T
        self.dHx_fn = ca.Function('dHx', [self.state, self.control, self.costate, self.auxvar], [self.dHx],
                                  ['x', 'u', 'lam', 'p'], ['dHdx'])
        self.dHu = jacobian(self.path_Hamil, self.control).T
        self.dHu_fn = ca.Function('dHu', [self.state, self.control, self.costate, self.auxvar], [self.dHu],
                                  ['x', 'u', 'lam', 'p'], ['dHdu'])

        self.ddHxx = jacobian(self.dHx, self.state)
        self.ddHxx_fn = ca.Function('ddHxx', [self.state, self.control, self.costate, self.auxvar], [self.ddHxx],
                                    ['x', 'u', 'lam', 'p'], ['d2Hdx2'])
        self.ddHxu = jacobian(self.dHx, self.control)
        self.ddHxu_fn = ca.Function('ddHxu', [self.state, self.control, self.costate, self.auxvar], [self.ddHxu],
                                    ['x', 'u', 'lam', 'p'], ['d2HdxdU'])
        self.ddHxe = jacobian(self.dHx, self.auxvar) if self.n_auxvar > 0 else SX.zeros(self.n_state, self.n_auxvar)
        self.ddHxe_fn = ca.Function('ddHxe', [self.state, self.control, self.costate, self.auxvar], [self.ddHxe],
                                    ['x', 'u', 'lam', 'p'], ['d2Hdxp'])

        self.ddHux = jacobian(self.dHu, self.state)
        self.ddHux_fn = ca.Function('ddHux', [self.state, self.control, self.costate, self.auxvar], [self.ddHux],
                                    ['x', 'u', 'lam', 'p'], ['d2Hdux'])
        self.ddHuu = jacobian(self.dHu, self.control)
        self.ddHuu_fn = ca.Function('ddHuu', [self.state, self.control, self.costate, self.auxvar], [self.ddHuu],
                                    ['x', 'u', 'lam', 'p'], ['d2Hdu2'])
        self.ddHue = jacobian(self.dHu, self.auxvar) if self.n_auxvar > 0 else SX.zeros(self.n_control, self.n_auxvar)
        self.ddHue_fn = ca.Function('ddHue', [self.state, self.control, self.costate, self.auxvar], [self.ddHue],
                                    ['x', 'u', 'lam', 'p'], ['d2Hdup'])

        self.dhx = jacobian(self.final_Hamil, self.state).T
        self.dhx_fn = ca.Function('dhx', [self.state, self.auxvar], [self.dhx],
                                  ['x', 'p'], ['dhdx'])
        self.ddhxx = jacobian(self.dhx, self.state)
        self.ddhxx_fn = ca.Function('ddhxx', [self.state, self.auxvar], [self.ddhxx],
                                    ['x', 'p'], ['d2hdx2'])
        self.ddhxe = jacobian(self.dhx, self.auxvar) if self.n_auxvar > 0 else SX.zeros(self.n_state, self.n_auxvar)
        self.ddhxe_fn = ca.Function('ddhxe', [self.state, self.auxvar], [self.ddhxe],
                                    ['x', 'p'], ['d2hdxp'])

        self._pmp_diff_done = True

    def getAuxSys(self, state_traj_opt, control_traj_opt, costate_traj_opt, auxvar_value=None):
        if not self._pmp_diff_done:
            self.diffPMP()

        horizon = control_traj_opt.shape[0]
        if auxvar_value is None:
            if self.n_auxvar == 0:
                auxvar_value_np = np.array([])
            else:
                raise ValueError("auxvar_value required")
        else:
            auxvar_value_np = np.array(auxvar_value).flatten()

        dynF, dynG, dynE = [], [], []
        Hxx, Hxu, Hxe, Hux, Huu, Hue = [], [], [], [], [], []

        empty_aux = SX()
        for t in range(horizon):
            x_t = state_traj_opt[t, :]
            u_t = control_traj_opt[t, :]
            lam_tp1 = costate_traj_opt[t, :]  # lambda_{t+1}
            aux_arg = auxvar_value_np if self.n_auxvar > 0 else empty_aux

            dynF.append(self.dfx_fn(x_t, u_t, aux_arg).full())
            dynG.append(self.dfu_fn(x_t, u_t, aux_arg).full())
            dynE.append(self.dfe_fn(x_t, u_t, aux_arg).full())

            Hxx.append(self.ddHxx_fn(x_t, u_t, lam_tp1, aux_arg).full())
            Hxu.append(self.ddHxu_fn(x_t, u_t, lam_tp1, aux_arg).full())
            Hxe.append(self.ddHxe_fn(x_t, u_t, lam_tp1, aux_arg).full())
            Hux.append(self.ddHux_fn(x_t, u_t, lam_tp1, aux_arg).full())
            Huu.append(self.ddHuu_fn(x_t, u_t, lam_tp1, aux_arg).full())
            Hue.append(self.ddHue_fn(x_t, u_t, lam_tp1, aux_arg).full())

        xH = state_traj_opt[-1, :]
        aux_arg_final = auxvar_value_np if self.n_auxvar > 0 else empty_aux
        hxx = [self.ddhxx_fn(xH, aux_arg_final).full()]
        hxe = [self.ddhxe_fn(xH, aux_arg_final).full()]

        if self.n_auxvar == 0:
            dynE = [np.zeros((self.n_state, 0)) for _ in range(horizon)]
            Hxe = [np.zeros((self.n_state, 0)) for _ in range(horizon)]
            Hue = [np.zeros((self.n_control, 0)) for _ in range(horizon)]
            hxe = [np.zeros((self.n_state, 0))]

        return {
            "dynF": dynF, "dynG": dynG, "dynE": dynE,
            "Hxx": Hxx, "Hxu": Hxu, "Hxe": Hxe,
            "Hux": Hux, "Huu": Huu, "Hue": Hue,
            "hxx": hxx, "hxe": hxe
        }

# ==============================================================================
# LQR (FIXED Riccati recursion; supports n_batch=0 as "multi-batch initial states")
# ==============================================================================
class LQR:
    def __init__(self, project_name="LQR system"):
        self.project_name = project_name
        self.n_state = None
        self.n_control = None
        self.n_batch = None
        self.horizon = None
        self.dynF = None; self.dynG = None; self.dynE = None
        self.Hxx = None; self.Huu = None; self.Hxu = None; self.Hux = None
        self.Hxe = None; self.Hue = None
        self.hxx = None; self.hxe = None

    def _validate_matrix_list(self, matrix_list, name, expected_len, allow_none=False, is_final=False):
        if matrix_list is None:
            if allow_none:
                return None
            raise TypeError(f"{name} cannot be None")
        if is_final and isinstance(matrix_list, np.ndarray):
            matrix_list = [matrix_list]
        if not isinstance(matrix_list, list):
            raise TypeError(f"{name} must be a list")
        if (is_final and len(matrix_list) != 1) or ((not is_final) and len(matrix_list) != expected_len):
            raise ValueError(f"{name} length mismatch")
        if not all(isinstance(m, np.ndarray) for m in matrix_list):
            raise TypeError(f"{name} elements must be numpy arrays")
        if not all(m.ndim == 2 for m in matrix_list):
            raise ValueError(f"{name} must contain 2D arrays")
        return matrix_list

    def setDyn(self, dynF, dynG, dynE=None):
        self.dynF = self._validate_matrix_list(dynF, "dynF", len(dynF))
        self.dynG = self._validate_matrix_list(dynG, "dynG", len(dynG))
        self.horizon = len(dynF)
        self.n_state = self.dynF[0].shape[0]
        self.n_control = self.dynG[0].shape[1]
        if dynE is None:
            self.dynE = [np.zeros((self.n_state, 0)) for _ in range(self.horizon)]
        else:
            self.dynE = self._validate_matrix_list(dynE, "dynE", self.horizon)
        self.n_batch = self.dynE[0].shape[1]  # parameter dimension

    def setPathCost(self, Hxx, Huu, Hxu=None, Hux=None, Hxe=None, Hue=None):
        self.Hxx = self._validate_matrix_list(Hxx, "Hxx", self.horizon)
        self.Huu = self._validate_matrix_list(Huu, "Huu", self.horizon)
        self.Hxu = self._validate_matrix_list(Hxu, "Hxu", self.horizon, allow_none=True)
        self.Hux = self._validate_matrix_list(Hux, "Hux", self.horizon, allow_none=True)
        self.Hxe = self._validate_matrix_list(Hxe, "Hxe", self.horizon, allow_none=True)
        self.Hue = self._validate_matrix_list(Hue, "Hue", self.horizon, allow_none=True)

        # default zeros
        if self.Hxu is None:
            self.Hxu = [np.zeros((self.n_state, self.n_control)) for _ in range(self.horizon)]
        if self.Hux is None:
            self.Hux = [np.zeros((self.n_control, self.n_state)) for _ in range(self.horizon)]
        if self.Hxe is None:
            self.Hxe = [np.zeros((self.n_state, self.n_batch)) for _ in range(self.horizon)]
        if self.Hue is None:
            self.Hue = [np.zeros((self.n_control, self.n_batch)) for _ in range(self.horizon)]

    def setFinalCost(self, hxx, hxe=None):
        self.hxx = self._validate_matrix_list(hxx, "hxx", 1, is_final=True)
        self.hxe = self._validate_matrix_list(hxe, "hxe", 1, allow_none=True, is_final=True)
        if self.hxe is None:
            self.hxe = [np.zeros((self.n_state, self.n_batch))]

    def lqrSolver(self, ini_state, horizon):
        if horizon != self.horizon:
            raise ValueError("horizon mismatch")

        ini_x = np.array(ini_state, dtype=np.float64)
        if ini_x.ndim == 1:
            ini_x = ini_x.reshape(self.n_state, 1)
        if ini_x.shape[0] != self.n_state:
            raise ValueError("ini_state shape mismatch")
        n_batch_ini = ini_x.shape[1]

        # If parameter dimension is zero, allow any n_batch_ini (treat as multiple initial conditions)
        if self.n_batch > 0 and self.n_batch != n_batch_ini:
            raise ValueError(f"n_batch mismatch (params {self.n_batch} vs ini {n_batch_ini})")

        F, G, E = self.dynF, self.dynG, self.dynE
        Hxx, Huu, Hxu, Hux, Hxe, Hue = self.Hxx, self.Huu, self.Hxu, self.Hux, self.Hxe, self.Hue
        hxx_final = self.hxx[0]
        hxe_final = self.hxe[0]

        PP = [None] * (horizon + 1)
        WW = [None] * (horizon + 1)
        PP[horizon] = hxx_final
        WW[horizon] = hxe_final

        K_list = [None] * horizon
        k_list = [None] * horizon

        for t in range(horizon - 1, -1, -1):
            Ft, Gt, Et = F[t], G[t], E[t]
            Qt, Rt = Hxx[t], Huu[t]
            Nt = Hxu[t]        # ns x nc
            Ntt = Hux[t]       # nc x ns
            qx = Hxe[t]        # ns x nb
            ru = Hue[t]        # nc x nb

            Pn, Wn = PP[t+1], WW[t+1]

            Q_uu = Rt + Gt.T @ Pn @ Gt                    # nc x nc
            Q_ux = Ntt + Gt.T @ Pn @ Ft                   # nc x ns
            # parameter forcing into u
            Q_u  = ru + Gt.T @ (Pn @ Et + Wn)             # nc x nb
            Q_xx = Qt + Ft.T @ Pn @ Ft                    # ns x ns
            Q_x  = qx + Ft.T @ (Pn @ Et + Wn)             # ns x nb

            # solve gains
            try:
                Kt = -np.linalg.solve(Q_uu, Q_ux)
                kt = -np.linalg.solve(Q_uu, Q_u)
            except np.linalg.LinAlgError:
                Quu_pinv = np.linalg.pinv(Q_uu)
                Kt = -Quu_pinv @ Q_ux
                kt = -Quu_pinv @ Q_u

            # Riccati update (DP exact form)
            Pt = Q_xx + Kt.T @ Q_uu @ Kt + Kt.T @ Q_ux + Q_ux.T @ Kt
            Wt = Q_x + Kt.T @ Q_u + Q_ux.T @ kt + Kt.T @ Q_uu @ kt

            # symmetrize Pt for numerical stability
            Pt = 0.5 * (Pt + Pt.T)

            PP[t] = Pt
            WW[t] = Wt
            K_list[t] = Kt
            k_list[t] = kt

        # forward pass
        x_traj = [None] * (horizon + 1)
        u_traj = [None] * horizon
        lam_traj = [None] * (horizon + 1)

        x_traj[0] = ini_x
        lam_traj[0] = PP[0] @ x_traj[0] if self.n_batch == 0 else (PP[0] @ x_traj[0] + WW[0])

        for t in range(horizon):
            Kt, kt = K_list[t], k_list[t]
            if self.n_batch == 0:
                u_t = Kt @ x_traj[t]
                x_next = F[t] @ x_traj[t] + G[t] @ u_t
                lam_next = PP[t+1] @ x_next
            else:
                u_t = Kt @ x_traj[t] + kt
                x_next = F[t] @ x_traj[t] + G[t] @ u_t + E[t]
                lam_next = PP[t+1] @ x_next if self.n_batch == 0 else (PP[t+1] @ x_next + WW[t+1])
            x_traj[t+1] = x_next
            u_traj[t] = u_t
            lam_traj[t+1] = lam_next

        return {
            "state_traj_opt": x_traj,
            "control_traj_opt": u_traj,
            "costate_traj_opt": lam_traj[1:],
            "time": np.arange(horizon+1)
        }

# ==============================================================================
# True system + linearization
# ==============================================================================
class CoupledOscillator:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.omega1 = 2.0
        self.omega2 = 2.5
        self.beta1 = 0.1
        self.beta2 = 0.15
        self.kappa_c0 = 0.5
        self.kappa_c1 = 0.8
        self.h_epsilon = 0.01
        self.h_noise_sigma = 0.0  # default off for reproducible gradients
        self.n_state_full = 5
        self.n_state_obs = 4
        self.n_control = 2

    def kappa(self, h_val):
        return self.kappa_c0 + self.kappa_c1 * np.tanh(h_val)

    def dkappa_dh(self, h_val):
        # d/dh [kappa_c0 + kappa_c1 tanh(h)] = kappa_c1 * (1 - tanh(h)^2)
        th = np.tanh(h_val)
        return self.kappa_c1 * (1.0 - th*th)

    def dynamics_step(self, x_full_k, u_k, h_target_k):
        p1, v1, p2, v2, h_val = x_full_k
        u1, u2 = u_k
        kap = self.kappa(h_val)
        dv1 = (-self.omega1**2 * p1 - self.beta1 * v1 - kap * (p1 - p2) + u1)
        p1_next = p1 + v1 * self.dt
        v1_next = v1 + dv1 * self.dt

        dv2 = (-self.omega2**2 * p2 - self.beta2 * v2 - kap * (p2 - p1) + u2)
        p2_next = p2 + v2 * self.dt
        v2_next = v2 + dv2 * self.dt

        dh = self.h_epsilon * (h_target_k - h_val)
        h_next = h_val + dh * self.dt
        if self.h_noise_sigma > 0:
            h_next += self.h_noise_sigma * np.sqrt(self.dt) * np.random.randn()

        return np.array([p1_next, v1_next, p2_next, v2_next, h_next], dtype=float)

    def linearize_step(self, x_full_k, u_k, h_target_k):
        """
        Returns (Fx, Fu) for x_{k+1} = f(x_k, u_k).
        Noise ignored (derivative 0).
        """
        dt = self.dt
        p1, v1, p2, v2, h = x_full_k
        u1, u2 = u_k
        kap = self.kappa(h)
        dkap = self.dkappa_dh(h)

        # dv1 = -w1^2 p1 - b1 v1 - kap (p1 - p2) + u1
        # dv2 = -w2^2 p2 - b2 v2 - kap (p2 - p1) + u2

        Fx = np.zeros((5, 5), dtype=float)
        Fu = np.zeros((5, 2), dtype=float)

        # p1_next = p1 + v1 dt
        Fx[0, 0] = 1.0
        Fx[0, 1] = dt

        # v1_next = v1 + dt*dv1
        Fx[1, 0] = dt * (-self.omega1**2 - kap)         # ∂v1_next/∂p1
        Fx[1, 1] = 1.0 + dt * (-self.beta1)             # ∂/∂v1
        Fx[1, 2] = dt * (kap)                           # ∂/∂p2
        # ∂dv1/∂h = -(dkap)*(p1 - p2)
        Fx[1, 4] = dt * (-(dkap) * (p1 - p2))

        Fu[1, 0] = dt * 1.0  # ∂v1_next/∂u1

        # p2_next = p2 + v2 dt
        Fx[2, 2] = 1.0
        Fx[2, 3] = dt

        # v2_next
        Fx[3, 0] = dt * (kap)                           # ∂/∂p1 (because -kap(p2-p1) => +kap p1)
        Fx[3, 2] = dt * (-self.omega2**2 - kap)         # ∂/∂p2
        Fx[3, 3] = 1.0 + dt * (-self.beta2)             # ∂/∂v2
        # ∂dv2/∂h = -(dkap)*(p2 - p1)
        Fx[3, 4] = dt * (-(dkap) * (p2 - p1))

        Fu[3, 1] = dt * 1.0  # ∂v2_next/∂u2

        # h_next = h + dt * h_eps (h_target - h)
        Fx[4, 4] = 1.0 + dt * (-self.h_epsilon)
        # other partials 0, Fu row 4 is 0

        return Fx, Fu

    def generate_trajectories(self, num_traj, traj_len):
        state_traj_list_full = []
        control_traj_list = []
        for _ in range(num_traj):
            x_full_0 = np.random.uniform(-1.5, 1.5, self.n_state_full)
            x_full_0[4] = np.random.uniform(-1, 1)
            h_target_current = np.random.uniform(-0.8, 0.8)

            current_states_full = [x_full_0]
            current_controls = []
            x_full_k = x_full_0.copy()
            for _ in range(traj_len):
                u_k = np.random.randn(self.n_control) * 0.8
                x_full_k_next = self.dynamics_step(x_full_k, u_k, h_target_current)
                current_states_full.append(x_full_k_next)
                current_controls.append(u_k)
                x_full_k = x_full_k_next

            state_traj_list_full.append(np.array(current_states_full))
            control_traj_list.append(np.array(current_controls))

        state_traj_list_obs = [traj[:, :self.n_state_obs] for traj in state_traj_list_full]
        return state_traj_list_obs, control_traj_list, state_traj_list_full

# ==============================================================================
# Koopman identification helpers
# ==============================================================================
def identify_koopman_model(system, state_traj_list_obs, control_traj_list):
    def simple_phi_func(x_obs_np):
        p1, v1, p2, v2 = x_obs_np
        return np.array([
            p1, v1, p2, v2,
            p1**2, v1**2, p2**2, v2**2,
            p1*v1, p2*v2,
            p1*p2
        ], dtype=float)

    n_observables = simple_phi_func(np.zeros(system.n_state_obs)).shape[0]
    C_matrix = np.zeros((system.n_state_obs, n_observables))
    C_matrix[:, :system.n_state_obs] = np.eye(system.n_state_obs)

    koopman_learner = Koopman_SysID(
        n_state=system.n_state_obs,
        n_control=system.n_control,
        n_observables=n_observables,
        phi_func=simple_phi_func,
        C_matrix=C_matrix
    )
    koopman_learner.fit(state_traj_list_obs, control_traj_list, lambda_reg=1e-4)
    A_np, B_np, C_np, phi_fn_np, n_lifted = koopman_learner.get_model()
    if A_np is None:
        raise RuntimeError("Koopman fit failed")
    return A_np, B_np, C_np, phi_fn_np, n_lifted

def fd_jacobian_phi(phi_func, x, eps=1e-6):
    x = np.array(x, dtype=float).flatten()
    z0 = phi_func(x)
    nz = z0.size
    nx = x.size
    J = np.zeros((nz, nx), dtype=float)
    for i in range(nx):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        zp = phi_func(xp); zm = phi_func(xm)
        J[:, i] = (zp - zm) / (2*eps)
    return J

# ==============================================================================
# MPC controller (same external interface)
# ==============================================================================
class KoopmanMPC:
    def __init__(self, system_params, koopman_model, mpc_horizon, initial_theta_cost_np):
        self.A_k, self.B_k, self.C_k, self.phi_k, self.n_k_lifted = koopman_model
        self.n_state_obs = system_params['n_state_obs']
        self.n_control = system_params['n_control']
        self.dt = system_params['dt']
        self.mpc_horizon = mpc_horizon

        self.ocs = OCSys(project_name="Koopman_MPC_OCP")
        self.x_obs_sym_ocs = ca.SX.sym('x_obs_ocs', self.n_state_obs)
        self.ocs.setStateVariable(self.x_obs_sym_ocs)

        self.u_sym_ocs = ca.SX.sym('u_ocs', self.n_control)
        control_limit = 3.0
        self.ocs.setControlVariable(self.u_sym_ocs,
                                    control_lb=[-control_limit]*self.n_control,
                                    control_ub=[control_limit]*self.n_control)

        self.n_theta_cost = self.n_state_obs + self.n_control
        self.theta_cost_sym_ocs = ca.SX.sym('theta_cost_ocs', self.n_theta_cost)
        self.ocs.setAuxvarVariable(self.theta_cost_sym_ocs)
        self.current_theta_cost_np = initial_theta_cost_np.copy()

        self.ocs.setKoopmanModel(self.A_k, self.B_k, self.C_k, self.phi_k, self.n_k_lifted)
        self.ocs.setDyn()

        def mpc_path_cost_lambda(x_obs_lambda, u_lambda, theta_cost_lambda):
            Q_diag = [ca.exp(theta_cost_lambda[i]) for i in range(self.n_state_obs)]
            R_diag = [ca.exp(theta_cost_lambda[i + self.n_state_obs]) for i in range(self.n_control)]
            Q_mat = ca.diag(ca.vertcat(*Q_diag))
            R_mat = ca.diag(ca.vertcat(*R_diag))
            target = ca.SX.zeros(self.n_state_obs)
            e = x_obs_lambda - target
            return 0.5 * ca.mtimes([e.T, Q_mat, e]) + 0.5 * ca.mtimes([u_lambda.T, R_mat, u_lambda])

        def mpc_final_cost_lambda(x_obs_lambda, theta_cost_lambda):
            Q_diag = [ca.exp(theta_cost_lambda[i]) * 10.0 for i in range(self.n_state_obs)]
            Q_mat = ca.diag(ca.vertcat(*Q_diag))
            target = ca.SX.zeros(self.n_state_obs)
            e = x_obs_lambda - target
            return 0.5 * ca.mtimes([e.T, Q_mat, e])

        self.ocs.setPathCost(mpc_path_cost_lambda)
        self.ocs.setFinalCost(mpc_final_cost_lambda)

    def solve_mpc_step(self, current_obs_state_np, current_theta_cost_np_val):
        current_theta_cost_np_val = np.array(current_theta_cost_np_val).flatten()
        mpc_sol = self.ocs.ocSolver(
            ini_state_orig=current_obs_state_np,
            horizon=self.mpc_horizon,
            auxvar_value=current_theta_cost_np_val,
            print_level=0,
            costate_option=0
        )
        if mpc_sol["success"] and mpc_sol["control_traj_opt"] is not None and len(mpc_sol["control_traj_opt"]) > 0:
            return mpc_sol["control_traj_opt"][0, :], mpc_sol
        return np.zeros(self.n_control), mpc_sol

    def update_theta_cost(self, new_theta_cost_np):
        self.current_theta_cost_np = new_theta_cost_np.copy()

# ==============================================================================
# Outer bilevel gradient: adjoint for true dynamics + chain pi_x, pi_theta
# ==============================================================================
def learn_mpc_cost_params_pdp_adjoint(
    true_system,
    koopman_mpc_controller,
    sim_horizon_meta,
    num_meta_iterations,
    learning_rate_meta,
    initial_true_state_full_np
):
    logger.info("Starting meta-learning (with full outer-chain adjoint)...")

    theta = koopman_mpc_controller.current_theta_cost_np.copy()
    meta_cost_history = []

    aux_lqr_theta = LQR("AuxLQR_theta")
    aux_lqr_x0 = LQR("AuxLQR_x0")

    for meta_iter in range(num_meta_iterations):
        x = initial_true_state_full_np.copy()
        h_target_episode = 0.0

        # store forward rollout data
        u_list = []
        Fx_list = []
        Fu_list = []
        lx_list = []
        lu_list = []
        pi_theta_list = []
        pi_x_list = []

        meta_cost = 0.0

        for t in range(sim_horizon_meta):
            x_obs = x[:true_system.n_state_obs].copy()

            u, mpc_sol = koopman_mpc_controller.solve_mpc_step(x_obs, theta)
            u_list.append(u.copy())

            # Get pi_theta = du/dtheta via PDP aux system (existing structure)
            # Get pi_x = du/dx_full via: du/dz0 * dz0/dx_obs, embed into full x.
            pi_theta = np.zeros((true_system.n_control, koopman_mpc_controller.n_theta_cost), dtype=float)
            pi_x = np.zeros((true_system.n_control, true_system.n_state_full), dtype=float)

            if mpc_sol["success"] and mpc_sol["lifted_state_traj_opt"] is not None and \
               mpc_sol["control_traj_opt"] is not None and mpc_sol["costate_traj_opt"] is not None:

                if not koopman_mpc_controller.ocs._pmp_diff_done:
                    koopman_mpc_controller.ocs.diffPMP()

                aux_sys = koopman_mpc_controller.ocs.getAuxSys(
                    state_traj_opt=mpc_sol["lifted_state_traj_opt"],
                    control_traj_opt=mpc_sol["control_traj_opt"],
                    costate_traj_opt=mpc_sol["costate_traj_opt"],
                    auxvar_value=theta
                )

                # --- pi_theta via PDP auxiliary LQR ---
                aux_lqr_theta.setDyn(aux_sys["dynF"], aux_sys["dynG"], aux_sys["dynE"])
                aux_lqr_theta.setPathCost(aux_sys["Hxx"], aux_sys["Huu"],
                                          Hxu=aux_sys["Hxu"], Hux=aux_sys["Hux"],
                                          Hxe=aux_sys["Hxe"], Hue=aux_sys["Hue"])
                aux_lqr_theta.setFinalCost(aux_sys["hxx"], aux_sys["hxe"])

                ini_X_theta = np.zeros((koopman_mpc_controller.n_k_lifted, koopman_mpc_controller.n_theta_cost))
                sol_theta = aux_lqr_theta.lqrSolver(ini_X_theta, koopman_mpc_controller.mpc_horizon)
                pi_theta = sol_theta["control_traj_opt"][0] if sol_theta["control_traj_opt"] else pi_theta

                # --- pi_z0 = du/dz0 via a second LQR with n_batch=0 and identity initial batch ---
                # Build "state-sensitivity" auxiliary system by zeroing parameter forcing terms
                dynE0 = [np.zeros((koopman_mpc_controller.n_k_lifted, 0)) for _ in range(koopman_mpc_controller.mpc_horizon)]
                Hxe0 = [np.zeros((koopman_mpc_controller.n_k_lifted, 0)) for _ in range(koopman_mpc_controller.mpc_horizon)]
                Hue0 = [np.zeros((koopman_mpc_controller.n_control, 0)) for _ in range(koopman_mpc_controller.mpc_horizon)]
                hxe0 = [np.zeros((koopman_mpc_controller.n_k_lifted, 0))]

                aux_lqr_x0.setDyn(aux_sys["dynF"], aux_sys["dynG"], dynE0)
                aux_lqr_x0.setPathCost(aux_sys["Hxx"], aux_sys["Huu"],
                                       Hxu=aux_sys["Hxu"], Hux=aux_sys["Hux"],
                                       Hxe=Hxe0, Hue=Hue0)
                aux_lqr_x0.setFinalCost(aux_sys["hxx"], hxe0)

                ini_X_x0 = np.eye(koopman_mpc_controller.n_k_lifted, dtype=float)  # treat as 11 batches
                sol_x0 = aux_lqr_x0.lqrSolver(ini_X_x0, koopman_mpc_controller.mpc_horizon)
                pi_z0 = sol_x0["control_traj_opt"][0] if sol_x0["control_traj_opt"] else np.zeros((true_system.n_control, koopman_mpc_controller.n_k_lifted))

                # chain: du/dx_obs = du/dz0 * dz0/dx_obs
                J_phi = fd_jacobian_phi(koopman_mpc_controller.phi_k, x_obs)
                pi_x_obs = pi_z0 @ J_phi  # (2x11)(11x4)->2x4
                pi_x[:, :true_system.n_state_obs] = pi_x_obs

            pi_theta_list.append(pi_theta)
            pi_x_list.append(pi_x)

            # True dynamics step + linearization
            Fx, Fu = true_system.linearize_step(x, u, h_target_episode)
            x_next = true_system.dynamics_step(x, u, h_target_episode)

            Fx_list.append(Fx)
            Fu_list.append(Fu)

            # meta stage cost uses x_t (full 5D) and u_t
            p1 = x[0]
            v1 = x[1]
            p2 = x[2]
            v2 = x[3]
            h  = x[4]

            # State weights (outer/meta objective) - defined on full 5D state
            w_p1 = 1.0
            w_v1 = 0.02
            w_p2 = 0.1
            w_v2 = 0.02
            w_h  = 0.05
            w_u  = 0.01

            stage_cost = (w_p1 * (p1 ** 2) +
                          w_v1 * (v1 ** 2) +
                          w_p2 * (p2 ** 2) +
                          w_v2 * (v2 ** 2) +
                          w_h  * (h  ** 2) +
                          w_u  * float(np.sum(u * u)))
            meta_cost += stage_cost

            # derivatives wrt u_t and x_t
            lu = (2.0 * w_u) * u  # d/du (w_u ||u||^2)
            lx = np.zeros(5, dtype=float)
            lx[0] = (2.0 * w_p1) * p1
            lx[1] = (2.0 * w_v1) * v1
            lx[2] = (2.0 * w_p2) * p2
            lx[3] = (2.0 * w_v2) * v2
            lx[4] = (2.0 * w_h)  * h

            lu_list.append(lu)
            lx_list.append(lx)

            x = x_next

        # terminal cost on x_T (state-only, defined on full 5D)
        p1_T, v1_T, p2_T, v2_T, h_T = x[0], x[1], x[2], x[3], x[4]
        w_p1, w_v1, w_p2, w_v2, w_h = 1.0, 0.02, 0.1, 0.02, 0.05
        terminal_cost = (w_p1 * (p1_T ** 2) +
                         w_v1 * (v1_T ** 2) +
                         w_p2 * (p2_T ** 2) +
                         w_v2 * (v2_T ** 2) +
                         w_h  * (h_T  ** 2))
        meta_cost += terminal_cost
        meta_cost_history.append(meta_cost)

        # backward adjoint for outer problem (x_t cost)
        lam_next = np.zeros(5, dtype=float)   # lambda_{t+1}
        lam_next[0] = 2.0 * w_p1 * p1_T
        lam_next[1] = 2.0 * w_v1 * v1_T
        lam_next[2] = 2.0 * w_p2 * p2_T
        lam_next[3] = 2.0 * w_v2 * v2_T
        lam_next[4] = 2.0 * w_h  * h_T

        grad = np.zeros_like(theta, dtype=float)

        for t in range(sim_horizon_meta - 1, -1, -1):
            # g_u = ∂l/∂u_t + f_u^T λ_{t+1}
            g_u = lu_list[t] + Fu_list[t].T @ lam_next

            # gradient accumulate: (∂π/∂θ)^T g_u
            grad += pi_theta_list[t].T @ g_u

            # adjoint update: λ_t = ∂l/∂x_t + f_x^T λ_{t+1} + (∂π/∂x)^T g_u
            lam_next = lx_list[t] + Fx_list[t].T @ lam_next + pi_x_list[t].T @ g_u

        # gradient clip
        gn = np.linalg.norm(grad)
        if gn > 20.0:
            grad *= (20.0 / (gn + 1e-12))

        theta = theta - learning_rate_meta * grad
        koopman_mpc_controller.update_theta_cost(theta)

        weights = np.exp(theta)
        logger.info(
            f"MetaIter {meta_iter+1}/{num_meta_iterations} | MetaCost={meta_cost:.3e} | "
            f"GradNorm={gn:.2e} | Weights(exp(theta))={np.round(weights, 4)}"
        )

    return theta, meta_cost_history

# ==============================================================================
# Main (FAST settings for runnable demo)
# ==============================================================================
if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    true_sys = CoupledOscillator(dt=0.05)
    system_params = {"n_state_obs": true_sys.n_state_obs, "n_control": true_sys.n_control, "dt": true_sys.dt}

    # --- FAST Koopman dataset ---
    NUM_TRAJ_KOOPMAN = 8
    TRAJ_LEN_KOOPMAN = 80

    logger.info("Stage 1: generating data for Koopman ID (FAST)...")
    state_trajs_obs, control_trajs, _ = true_sys.generate_trajectories(NUM_TRAJ_KOOPMAN, TRAJ_LEN_KOOPMAN)

    logger.info("Stage 2: identifying Koopman model...")
    A_k, B_k, C_k, phi_k, n_k_lifted = identify_koopman_model(true_sys, state_trajs_obs, control_trajs)
    koopman_model = (A_k, B_k, C_k, phi_k, n_k_lifted)

    # --- FAST MPC/meta settings ---
    MPC_HORIZON = 6
    SIM_HORIZON_META = 10
    NUM_META_ITERATIONS = 4
    LEARNING_RATE_META = 8e-4

    initial_theta = np.array([
        np.log(10.0), np.log(0.1), np.log(1.0), np.log(0.1),
        np.log(0.01), np.log(0.01)
    ], dtype=float)

    logger.info("Stage 3: init Koopman MPC...")
    koopman_mpc = KoopmanMPC(system_params, koopman_model, MPC_HORIZON, initial_theta)

    x0_true = np.array([1.0, 0.0, 0.2, 0.0, 0.1], dtype=float)

    logger.info("Stage 4: meta-learning with full adjoint chain...")
    theta_learned, meta_hist = learn_mpc_cost_params_pdp_adjoint(
        true_system=true_sys,
        koopman_mpc_controller=koopman_mpc,
        sim_horizon_meta=SIM_HORIZON_META,
        num_meta_iterations=NUM_META_ITERATIONS,
        learning_rate_meta=LEARNING_RATE_META,
        initial_true_state_full_np=x0_true
    )

    print("\n===== FINAL (FAST RUN) =====")
    print("learned log theta:", np.round(theta_learned, 6))
    print("learned weights exp(theta):", np.round(np.exp(theta_learned), 6))
    print("meta cost history:", [float(v) for v in meta_hist])

    # plot meta cost history
    plt.figure()
    plt.plot(meta_hist, marker='o')
    plt.xlabel("Meta iteration")
    plt.ylabel("Meta episode cost")
    plt.title("Meta cost history (FAST run)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(output_dir / "meta_cost_history_fast.png"))
    print("Saved plot:", str(output_dir / "meta_cost_history_fast.png"))
