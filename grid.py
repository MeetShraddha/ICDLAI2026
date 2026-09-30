"""IEEE 33-bus feeder, reconfigurations, AC power flow, PV scenarios, measurement model."""
import numpy as np
import pandapower.networks as pn

VBASE_KV, SBASE = 12.66, 1.0
ZBASE = VBASE_KV ** 2 / SBASE
NB = 33

_net = pn.case33bw()
LINES = _net.line[["from_bus", "to_bus"]].values.astype(int)          # 37 x 2
ZL = (_net.line.r_ohm_per_km.values + 1j * _net.line.x_ohm_per_km.values) * _net.line.length_km.values / ZBASE
BASE_IN = _net.line.in_service.values.astype(bool)                     # 32 sectionalizing in service
LOAD_P = np.zeros(NB); LOAD_Q = np.zeros(NB)
LOAD_P[_net.load.bus.values] = _net.load.p_mw.values / SBASE
LOAD_Q[_net.load.bus.values] = _net.load.q_mvar.values / SBASE
PEAK = LOAD_P.sum()

# ---------- measurement placement ----------
VM_BUSES = np.array([5, 11, 17, 21, 24, 29, 32])     # voltage-magnitude meters (field)
PMU_BUSES = np.array([17, 32])                         # micro-PMU angle measurements
FLOW_LINES = np.array([1, 17, 21, 24])                 # line indices with P/Q flow meters (field)
SIG_VM, SIG_V0, SIG_TH = 0.003, 0.002, 2e-4            # p.u., p.u., rad
SIG_FLOW_REL, SIG_FLOW_ABS = 0.02, 2e-3
SIG_S0_REL = 0.01
SIG_LOAD_REL = 0.20                                    # pseudo-measurement error of load forecasts
SIG_PV_SITE = 0.20                                     # site-level irradiance deviation from feeder sensor


def is_radial_connected(mask):
    if mask.sum() != NB - 1:
        return False
    parent = list(range(NB))
    def f(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for (u, v) in LINES[mask]:
        ru, rv = f(u), f(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return True


def make_topologies(n, rng):
    """Base radial configuration plus n-1 distinct radial reconfigurations (1-3 branch exchanges)."""
    topos = [BASE_IN.copy()]
    seen = {BASE_IN.tobytes()}
    ties = np.where(~BASE_IN)[0]
    while len(topos) < n:
        m = BASE_IN.copy()
        for _ in range(rng.integers(1, 4)):
            t = rng.choice(ties[~m[ties]]) if (~m[ties]).any() else None
            if t is None:
                break
            m[t] = True
            cand = [l for l in np.where(m)[0] if l != 0 and l != t]
            rng.shuffle(cand)
            for l in cand:
                m[l] = False
                if is_radial_connected(m):
                    break
                m[l] = True
        if is_radial_connected(m) and m.tobytes() not in seen:
            # reject configurations with extreme voltage drop at nominal peak load
            V = power_flow(np.array([build_ybus(m)]), np.zeros(1, int), -(LOAD_P + 1j * LOAD_Q)[None])
            if V is not None and np.abs(V[0]).min() > 0.88:
                seen.add(m.tobytes()); topos.append(m)
    return np.array(topos)


def build_ybus(mask):
    Y = np.zeros((NB, NB), complex)
    for (u, v), z, on in zip(LINES, ZL, mask):
        if on:
            y = 1 / z
            Y[u, u] += y; Y[v, v] += y; Y[u, v] -= y; Y[v, u] -= y
    return Y


def power_flow(Ybus_stack, topo_idx, S, tol=1e-10, iters=200):
    """Batched Z-bus fixed-point AC power flow. S: complex net injections (gen - load), slack bus 0 at 1.0 pu."""
    B = S.shape[0]
    V = np.ones((B, NB), complex)
    Zll = np.stack([np.linalg.inv(Y[1:, 1:]) for Y in Ybus_stack])
    Yl0 = np.stack([Y[1:, 0] for Y in Ybus_stack])
    Zt, Yt = Zll[topo_idx], Yl0[topo_idx]
    for it in range(iters):
        I = np.conj(S[:, 1:] / V[:, 1:]) - Yt * V[:, :1]
        Vn = np.einsum("bij,bj->bi", Zt, I)
        err = np.abs(Vn - V[:, 1:]).max()
        V[:, 1:] = Vn
        if err < tol:
            return V
    return V if err < 1e-6 else None


def generate(n, topo_ids, Ybus_stack, rng, pen_range=(0.0, 1.0), pen_fixed=None):
    """Sample operating points. Returns dict with true states and scenario variables."""
    topo = rng.choice(topo_ids, n)
    pen = np.full(n, pen_fixed) if pen_fixed is not None else rng.uniform(*pen_range, n)
    lam = rng.uniform(0.3, 1.1, n)[:, None] * np.clip(1 + 0.15 * rng.standard_normal((n, NB)), 0.4, 1.6)
    pf_scale = np.clip(1 + 0.1 * rng.standard_normal((n, NB)), 0.7, 1.3)
    Pl, Ql = LOAD_P * lam, LOAD_Q * lam * pf_scale
    # PV sites: random 30-60% of load buses, capacity share random, total = pen * peak load
    cap = np.zeros((n, NB))
    for i in range(n):
        k = rng.integers(10, 20)
        sites = rng.choice(np.arange(1, NB), k, replace=False)
        w = rng.gamma(2.0, 1.0, k)
        cap[i, sites] = pen[i] * PEAK * w / w.sum()
    g = rng.beta(2.0, 1.5, n)                                      # feeder clear-sky/irradiance index
    site = np.clip(g[:, None] * (1 + SIG_PV_SITE * rng.standard_normal((n, NB))), 0, 1.0)
    Ppv = cap * site
    S = (Ppv - Pl) - 1j * Ql
    S[:, 0] = 0
    V = power_flow(Ybus_stack, topo, S)
    Ytopo = Ybus_stack[topo]
    Sinj = V * np.conj(np.einsum("bij,bj->bi", Ytopo, V))          # includes slack injection
    return dict(topo=topo, pen=pen, V=V, Sinj=Sinj, cap=cap, g=g, Pl=Pl, Ql=Ql, Ppv=Ppv)


def line_flows(V, topo_masks, lines_idx):
    """Complex power flow from->to on given lines (in-service lines only; zero otherwise)."""
    f, t = LINES[lines_idx, 0], LINES[lines_idx, 1]
    y = 1 / ZL[lines_idx]
    Sft = V[:, f] * np.conj((V[:, f] - V[:, t]) * y)
    return Sft * topo_masks[:, lines_idx]


def measure(d, topos, rng, noise_scale=1.0, drop=0.0):
    """Noisy, partially missing measurements + pseudo-measurements (same inputs for every estimator)."""
    n = len(d["topo"])
    V, Sinj = d["V"], d["Sinj"]
    ns = noise_scale
    z = {}
    z["v0"] = np.abs(V[:, 0]) + ns * SIG_V0 * rng.standard_normal(n)
    s0 = Sinj[:, 0]
    z["p0"] = s0.real + ns * SIG_S0_REL * np.abs(s0.real) * rng.standard_normal(n) + ns * 1e-3 * rng.standard_normal(n)
    z["q0"] = s0.imag + ns * SIG_S0_REL * np.abs(s0.imag) * rng.standard_normal(n) + ns * 1e-3 * rng.standard_normal(n)
    z["vm"] = np.abs(V[:, VM_BUSES]) + ns * SIG_VM * rng.standard_normal((n, len(VM_BUSES)))
    z["th"] = np.angle(V[:, PMU_BUSES]) + ns * SIG_TH * rng.standard_normal((n, len(PMU_BUSES)))
    tm = topos[d["topo"]]
    F = line_flows(V, tm, FLOW_LINES)
    sf = lambda x: ns * (SIG_FLOW_REL * np.abs(x) + SIG_FLOW_ABS)
    z["pf"] = F.real + sf(F.real) * rng.standard_normal(F.shape)
    z["qf"] = F.imag + sf(F.imag) * rng.standard_normal(F.shape)
    # availability masks for field measurements (substation always available); lines out of service have no meter
    dr = np.asarray(drop, float).reshape(-1, 1) if np.ndim(drop) else drop
    z["m_vm"] = (rng.random((n, len(VM_BUSES))) >= dr).astype(float)
    z["m_th"] = (rng.random((n, len(PMU_BUSES))) >= dr).astype(float)
    z["m_f"] = (rng.random((n, len(FLOW_LINES))) >= dr).astype(float) * tm[:, FLOW_LINES]
    # pseudo-measurements: load forecast (20% error) and PV estimate = capacity x feeder irradiance sensor
    pl = d["Pl"] * (1 + SIG_LOAD_REL * rng.standard_normal(d["Pl"].shape))
    ql = d["Ql"] * (1 + SIG_LOAD_REL * rng.standard_normal(d["Ql"].shape))
    g_meas = np.clip(d["g"] + 0.02 * rng.standard_normal(n), 0, 1)
    ppv = d["cap"] * g_meas[:, None]
    z["pp"] = ppv - pl
    z["qp"] = -ql
    z["sig_pp"] = np.sqrt((SIG_LOAD_REL * pl) ** 2 + (SIG_PV_SITE * ppv) ** 2 + 1e-8)
    z["sig_qp"] = np.sqrt((SIG_LOAD_REL * ql) ** 2 + 1e-8)
    z["cap"] = d["cap"]; z["g"] = g_meas
    for k in ["pp", "qp", "sig_pp", "sig_qp"]:
        z[k][:, 0] = 0
    return z
