
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

F = 96485.33212
R = 8.314462618

st.set_page_config(page_title="Reversible CA + UV–Vis SEC", layout="wide")

st.title("Reversible one-electron chronoamperometry with UV–Vis absorbance")
st.markdown(
    r"""
Planar 1D diffusion model for the simple reversible electron transfer

\[
\mathrm{O}+e^- \rightleftharpoons \mathrm{R}
\]

following a potential step from \(E_i\) to \(E_f\), with **Nernstian equilibrium
at the electrode** and a **zero-flux boundary** at the outer edge of the finite
spatial domain.
"""
)

with st.expander("Model equations and assumptions", expanded=False):
    st.markdown(
        r"""
For simplicity, O and R are assigned the same diffusion coefficient \(D\), so that

\[
c_\mathrm{O}(x,t)+c_\mathrm{R}(x,t)=c_T
\]

everywhere, and only \(c_\mathrm{R}\) needs to be propagated:

\[
\frac{\partial c_\mathrm{R}}{\partial t}
=
D\frac{\partial^2 c_\mathrm{R}}{\partial x^2}.
\]

At the electrode surface, the reversible electron transfer obeys the Nernst equation,

\[
E=E^{0'}+\frac{RT}{F}\ln\!\left(\frac{c_{\mathrm O}(0,t)}
{c_{\mathrm R}(0,t)}\right),
\]

so that, after the potential step to \(E_f\),

\[
c_{\mathrm R}(0,t)
=
\frac{c_T}{1+\exp[F(E_f-E^{0'})/(RT)]}.
\]

At the outer boundary \(x=L\),

\[
\left.\frac{\partial c_\mathrm{R}}{\partial x}\right|_{x=L}=0.
\]

The cathodic current is plotted as negative and is calculated from the
interfacial flux of R:

\[
j = F D
\left.\frac{\partial c_\mathrm{R}}{\partial x}\right|_{x=0}.
\]

Absorbance is obtained from the local Beer–Lambert contribution of O and R.
No double-layer charging or uncompensated-resistance effects are included.
"""
    )

# ---------------- Sidebar ----------------
st.sidebar.header("Electrochemical parameters")
E0 = st.sidebar.number_input("Formal potential, E0' / V", value=0.000, step=0.010, format="%.3f")
E_initial = st.sidebar.number_input("Initial potential, Ei / V", value=0.300, step=0.010, format="%.3f")
E_final = st.sidebar.number_input("Step potential, Ef / V", value=-0.300, step=0.010, format="%.3f")
t_end = st.sidebar.number_input("Step duration / s", min_value=1e-4, value=20.0, step=1.0, format="%.4f")
T = st.sidebar.number_input("Temperature / K", min_value=200.0, value=298.15, step=1.0)
area_cm2 = st.sidebar.number_input("Electrode area / cm²", min_value=1e-6, value=0.0707, format="%.4f")

st.sidebar.header("Transport and concentration")
D_cm2_s = st.sidebar.number_input("Diffusion coefficient D / cm² s⁻¹", min_value=1e-9, value=1.0e-5, format="%.2e")
cT_mM = st.sidebar.number_input("Total concentration cT / mM", min_value=1e-6, value=1.00, format="%.4f")
initial_condition_mode = st.sidebar.selectbox(
    "Initial solution composition",
    ["Nernst equilibrium at Ei", "User-defined initial fraction of R"]
)

fR0_user = 0.0
if initial_condition_mode == "User-defined initial fraction of R":
    fR0_user = st.sidebar.slider("Initial bulk fraction of R", 0.0, 1.0, 0.0, 0.01)

L_um = st.sidebar.number_input("Domain length L / µm", min_value=1.0, value=1000.0, step=50.0)
N = st.sidebar.slider("Spatial grid points", 60, 600, 220, 20)
n_time = st.sidebar.slider("Output time points", 300, 4000, 1200, 100)

st.sidebar.header("Time-grid display")
log_time_display = st.sidebar.checkbox("Use logarithmic time axis for transient plots", value=True)

st.sidebar.header("Optical parameters")
eps_O = st.sidebar.number_input("ε(O) / M⁻¹ cm⁻¹", min_value=0.0, value=1000.0, step=100.0)
eps_R = st.sidebar.number_input("ε(R) / M⁻¹ cm⁻¹", min_value=0.0, value=10000.0, step=100.0)
optical_mode = st.sidebar.selectbox(
    "Optical geometry",
    ["Normal transmission", "Near-normal reflection", "Parallel (finite sampling layer)"]
)

parallel_path_cm = 1.0
x0_um = 50.0
beam_um = 100.0
if optical_mode == "Parallel (finite sampling layer)":
    parallel_path_cm = st.sidebar.number_input("Parallel optical path / cm", min_value=0.001, value=1.0, step=0.1)
    x0_um = st.sidebar.number_input("Beam lower edge x0 / µm", min_value=0.0, value=50.0, step=10.0)
    beam_um = st.sidebar.number_input("Beam thickness / µm", min_value=1.0, value=100.0, step=10.0)

# ---------------- Helpers ----------------
def surface_reduced_concentration(E, cT, E0, T):
    arg = F * (E - E0) / (R * T)
    arg = np.clip(arg, -700, 700)
    return cT / (1.0 + np.exp(arg))

@st.cache_data(show_spinner=False)
def simulate(E0, E_initial, E_final, T, D_cm2_s, cT_M, fR0,
             L_cm, N, n_time, t_end):
    x = np.linspace(0.0, L_cm, N)
    dx = x[1] - x[0]

    # Output grid includes t=0. For log-style inspection we still integrate normally.
    t_eval = np.linspace(0.0, t_end, n_time)

    # State vector contains nodes 1...N-1. Node 0 is the Nernstian boundary.
    y0 = np.full(N - 1, fR0 * cT_M, dtype=float)

    c0_final = surface_reduced_concentration(E_final, cT_M, E0, T)

    def rhs(t, y):
        c = np.empty(N, dtype=float)
        c[0] = c0_final
        c[1:] = y

        dy = np.empty_like(y)

        # Interior nodes x_1 ... x_{N-2}
        dy[:-1] = D_cm2_s * (c[2:] - 2.0*c[1:-1] + c[:-2]) / dx**2

        # Outer boundary x=L: dc/dx = 0, mirrored ghost point.
        dy[-1] = 2.0 * D_cm2_s * (c[-2] - c[-1]) / dx**2
        return dy

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        t_eval=t_eval,
        method="BDF",
        rtol=2e-6,
        atol=max(1e-12, cT_M * 1e-8),
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    cR = np.empty((N, sol.t.size), dtype=float)
    # At t=0, electrochemical boundary is understood as stepped instantaneously.
    cR[0, :] = c0_final
    cR[1:, :] = sol.y
    cR = np.clip(cR, -1e-12, cT_M + 1e-12)
    cO = cT_M - cR

    # 2nd-order one-sided derivative at electrode.
    dRdx0 = (-3.0*cR[0, :] + 4.0*cR[1, :] - cR[2, :]) / (2.0*dx)
    j_A_cm2 = F * D_cm2_s * dRdx0 * 1e-3

    E = np.full_like(sol.t, E_final, dtype=float)
    return sol.t, E, x, cO, cR, j_A_cm2

def optical_response(x_cm, cO_M, cR_M, eps_O, eps_R, mode,
                     parallel_path_cm=1.0, x0_cm=0.0, beam_cm=0.01):
    alpha = eps_O*cO_M + eps_R*cR_M

    if mode == "Normal transmission":
        A = np.trapezoid(alpha, x_cm, axis=0)

    elif mode == "Near-normal reflection":
        A = 2.0*np.trapezoid(alpha, x_cm, axis=0)

    else:
        lo = max(x_cm[0], x0_cm)
        hi = min(x_cm[-1], x0_cm + beam_cm)
        if hi <= lo:
            raise ValueError("The parallel-mode sampling layer does not overlap the spatial domain.")

        mask = (x_cm >= lo) & (x_cm <= hi)
        xs = x_cm[mask]
        aa = alpha[mask, :]

        if xs.size == 0 or xs[0] > lo:
            a_lo = np.array([np.interp(lo, x_cm, alpha[:, k]) for k in range(alpha.shape[1])])
            xs = np.r_[lo, xs]
            aa = np.vstack([a_lo, aa])

        if xs[-1] < hi:
            a_hi = np.array([np.interp(hi, x_cm, alpha[:, k]) for k in range(alpha.shape[1])])
            xs = np.r_[xs, hi]
            aa = np.vstack([aa, a_hi])

        alpha_mean = np.trapezoid(aa, xs, axis=0) / (hi - lo)
        A = parallel_path_cm * alpha_mean

    return A

# ---------------- Run ----------------
cT_M = cT_mM * 1e-3
L_cm = L_um * 1e-4
x0_cm = x0_um * 1e-4
beam_cm = beam_um * 1e-4

if initial_condition_mode == "Nernst equilibrium at Ei":
    fR0 = float(surface_reduced_concentration(E_initial, cT_M, E0, T) / cT_M)
else:
    fR0 = fR0_user

if optical_mode == "Parallel (finite sampling layer)" and x0_um >= L_um:
    st.warning("x0 lies outside the spatial domain. Reduce x0 or increase L.")

try:
    t, E, x, cO, cR, j = simulate(
        E0, E_initial, E_final, T, D_cm2_s, cT_M, fR0,
        L_cm, N, n_time, t_end
    )

    A = optical_response(
        x, cO, cR, eps_O, eps_R, optical_mode,
        parallel_path_cm, x0_cm, beam_cm
    )

    # Reference absorbance is the initial homogeneous solution before the step.
    cR_initial = fR0 * cT_M
    cO_initial = cT_M - cR_initial

    if optical_mode == "Normal transmission":
        A0 = (eps_O*cO_initial + eps_R*cR_initial) * L_cm
    elif optical_mode == "Near-normal reflection":
        A0 = 2.0 * (eps_O*cO_initial + eps_R*cR_initial) * L_cm
    else:
        A0 = (eps_O*cO_initial + eps_R*cR_initial) * parallel_path_cm

    dA = A - A0
    dA[0] = 0.0  # enforce the physical limit: a zero-thickness interfacial jump has no finite absorbance
    current_A = j * area_cm2

    diffusion_length_um = np.sqrt(D_cm2_s * t_end) * 1e4
    mass_error = np.max(np.abs((cO + cR) - cT_M))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Step duration", f"{t_end:.2f} s")
    m2.metric("√(D t)", f"{diffusion_length_um:.0f} µm")
    m3.metric("|i| at first finite time", f"{abs(current_A[1])*1e6:.2f} µA")
    m4.metric("ΔA final", f"{dA[-1]:.4g}")

    if diffusion_length_um > 0.5*L_um:
        st.info(
            "The diffusion length is a substantial fraction of the finite domain. "
            "The zero-flux outer boundary will therefore influence the chronoamperometric "
            "and optical responses."
        )

    tab1, tab2, tab3 = st.tabs(["Transient responses", "Concentration profiles", "Data"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(6.2, 4.5))
            mask = t > 0
            if log_time_display:
                ax.semilogx(t[mask], current_A[mask]*1e6, lw=1.8)
            else:
                ax.plot(t[mask], current_A[mask]*1e6, lw=1.8)
            ax.axhline(0, lw=0.7)
            ax.set_xlabel("t / s")
            ax.set_ylabel("i / µA")
            ax.set_title("Chronoamperometric response")
            ax.grid(alpha=0.2)
            st.pyplot(fig, clear_figure=True)

        with c2:
            fig, ax = plt.subplots(figsize=(6.2, 4.5))
            mask = t > 0
            if log_time_display:
                ax.semilogx(t[mask], dA[mask], lw=1.8)
            else:
                ax.plot(t[mask], dA[mask], lw=1.8)
            ax.axhline(0, lw=0.7)
            ax.set_xlabel("t / s")
            ax.set_ylabel("ΔA")
            ax.set_title(f"Optical response — {optical_mode}")
            ax.grid(alpha=0.2)
            st.pyplot(fig, clear_figure=True)

        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        if log_time_display:
            mask = t > 0
            ax.semilogx(t[mask], np.full(np.sum(mask), E_final), label="E / V")
            ax2 = ax.twinx()
            ax2.semilogx(t[mask], dA[mask], alpha=0.75)
        else:
            mask = t > 0
            ax.plot(t[mask], E[mask], label="E / V")
            ax2 = ax.twinx()
            ax2.plot(t[mask], dA[mask], alpha=0.75)

        ax.set_xlabel("t / s")
        ax.set_ylabel("E / V")
        ax2.set_ylabel("ΔA")
        ax.set_title("Potential step and optical response")
        ax.grid(alpha=0.2)
        st.pyplot(fig, clear_figure=True)

    with tab2:
        profile_times = [0.01*t_end, 0.05*t_end, 0.2*t_end, 0.5*t_end, 1.0*t_end]
        idxs = [np.argmin(np.abs(t - tp)) for tp in profile_times]

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for idx in idxs:
            ax.plot(x*1e4, cR[:, idx]*1e3, label=f"t={t[idx]:.3g} s")
        ax.set_xlabel("x / µm")
        ax.set_ylabel("c(R) / mM")
        ax.set_title("Reduced-species concentration profiles")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        st.pyplot(fig, clear_figure=True)

        st.caption(
            f"Maximum numerical deviation from c(O)+c(R)=cT: {mass_error:.2e} M."
        )

    with tab3:
        df = pd.DataFrame({
            "time_s": t,
            "E_V": E,
            "current_A": current_A,
            "current_density_A_cm2": j,
            "absorbance": A,
            "delta_absorbance": dA,
        })
        st.dataframe(df, use_container_width=True, height=420)
        st.download_button(
            "Download simulated response (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="reversible_ca_sec_simulation.csv",
            mime="text/csv",
        )

except Exception as exc:
    st.error(f"Simulation failed: {exc}")

st.markdown("---")
st.caption(
    "Numerical model: 1D finite differences in space + BDF time integration. "
    "The outer boundary is strictly reflective (zero flux), not a fixed-bulk-concentration boundary."
)
