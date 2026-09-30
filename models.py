"""Measurement model h(x) in torch, WLS estimator, MLP / GNN / PI-GNN estimators."""
import numpy as np
import torch
import torch.nn as nn
import grid as G

NB = G.NB
VMB = torch.tensor(G.VM_BUSES); PMB = torch.tensor(G.PMU_BUSES)
FL = torch.tensor(G.FLOW_LINES)
LF = torch.tensor(G.LINES[G.FLOW_LINES, 0]); LT = torch.tensor(G.LINES[G.FLOW_LINES, 1])
YF = 1 / G.ZL[G.FLOW_LINES]


class Batch:
    """Torch view of a set of samples: measurements, weights, topology admittances, targets."""
    def __init__(self, d, z, topos, Ystack, dtype=torch.float32):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype)
        n = len(d["topo"])
        self.n = n
        self.topo = torch.as_tensor(d["topo"])
        self.Gs, self.Bs = t(Ystack.real), t(Ystack.imag)
        self.tmask = t(topos.astype(float))
        # measurement vector z and weights w = mask / sigma (nominal sigma, as assumed by the operator)
        zs, ws = [], []
        def add(val, sig, mask=None):
            val = t(val); sig = t(sig)
            if val.dim() == 1: val = val[:, None]
            if sig.dim() == 1: sig = sig[:, None]
            sig = sig.expand_as(val)
            m = torch.ones_like(val) if mask is None else t(mask)
            zs.append(val); ws.append(m / sig)
        add(z["v0"], torch.tensor(G.SIG_V0))
        add(z["p0"], G.SIG_S0_REL * np.abs(z["p0"]) + 1e-3)
        add(z["q0"], G.SIG_S0_REL * np.abs(z["q0"]) + 1e-3)
        add(z["vm"], torch.full((n, len(G.VM_BUSES)), G.SIG_VM), z["m_vm"])
        add(z["th"], torch.full((n, len(G.PMU_BUSES)), G.SIG_TH), z["m_th"])
        add(z["pf"], G.SIG_FLOW_REL * np.abs(z["pf"]) + G.SIG_FLOW_ABS, z["m_f"])
        add(z["qf"], G.SIG_FLOW_REL * np.abs(z["qf"]) + G.SIG_FLOW_ABS, z["m_f"])
        add(z["pp"][:, 1:], z["sig_pp"][:, 1:])
        add(z["qp"][:, 1:], z["sig_qp"][:, 1:])
        self.z = torch.cat(zs, 1) * 1.0
        self.w = torch.cat(ws, 1)
        self.z = torch.where(self.w > 0, self.z, torch.zeros_like(self.z))
        self.Vm = t(np.abs(d["V"])); self.th = t(np.angle(d["V"]))
        self.raw = z
        self.dtype = dtype
        self.idx = torch.arange(n)
        self.nf = node_features(self)
        self.ei, self.ef = edge_features(self)
        fm = np.zeros((n, 12), np.float32)
        fm[:, 0:4] = z["pf"] * 5 * z["m_f"]; fm[:, 4:8] = z["qf"] * 5 * z["m_f"]; fm[:, 8:12] = z["m_f"]
        self.fm = torch.from_numpy(fm)

    def sub(self, idx):
        b = object.__new__(Batch)
        for k, v in self.__dict__.items():
            if k in ("Gs", "Bs", "tmask", "dtype", "raw", "n"):
                setattr(b, k, v)
            else:
                setattr(b, k, v[idx])
        b.n = len(idx)
        return b


def h_meas(Vm, th, Gm, Bm, tmask_row):
    """Measurement function for a batch. Vm, th: [B, NB]; Gm, Bm: [B, NB, NB]; tmask_row: [B, 37]."""
    Vr, Vi = Vm * torch.cos(th), Vm * torch.sin(th)
    Ir = torch.einsum("bij,bj->bi", Gm, Vr) - torch.einsum("bij,bj->bi", Bm, Vi)
    Ii = torch.einsum("bij,bj->bi", Gm, Vi) + torch.einsum("bij,bj->bi", Bm, Vr)
    P = Vr * Ir + Vi * Ii
    Q = Vi * Ir - Vr * Ii
    yr = torch.as_tensor(YF.real, dtype=Vm.dtype); yi = torch.as_tensor(YF.imag, dtype=Vm.dtype)
    dr, di = Vr[:, LF] - Vr[:, LT], Vi[:, LF] - Vi[:, LT]
    Ifr, Ifi = dr * yr - di * yi, dr * yi + di * yr
    Pf = (Vr[:, LF] * Ifr + Vi[:, LF] * Ifi) * tmask_row[:, FL]
    Qf = (Vi[:, LF] * Ifr - Vr[:, LF] * Ifi) * tmask_row[:, FL]
    return torch.cat([Vm[:, :1], P[:, :1], Q[:, :1], Vm[:, VMB], th[:, PMB], Pf, Qf, P[:, 1:], Q[:, 1:]], 1)


def residual(b, Vm, th):
    h = h_meas(Vm, th, b.Gs[b.topo], b.Bs[b.topo], b.tmask[b.topo])
    return (b.z - h) * b.w


# ---------------------------------------------------------------- learned estimators
VSCALE, TSCALE = 20.0, 30.0          # standardisation of outputs: (Vm-1)*20, th*30


def node_features(b):
    z = b.raw
    n = b.n
    idx = b.idx.numpy()
    f = np.zeros((n, NB, 11), np.float32)
    f[:, 0, 0] = (z["v0"][idx] - 1) * 20; f[:, 0, 1] = 1
    f[:, G.VM_BUSES, 0] = (z["vm"][idx] - 1) * 20 * z["m_vm"][idx]; f[:, G.VM_BUSES, 1] = z["m_vm"][idx]
    f[:, 0, 3] = 1                                                   # slack angle reference known
    f[:, G.PMU_BUSES, 2] = z["th"][idx] * 30 * z["m_th"][idx]; f[:, G.PMU_BUSES, 3] = z["m_th"][idx]
    f[:, :, 4] = z["pp"][idx] * 5; f[:, :, 5] = z["qp"][idx] * 5
    f[:, :, 6] = z["cap"][idx] * 5
    f[:, :, 7] = z["g"][idx][:, None]
    f[:, 0, 8] = 1
    f[:, 0, 9] = z["p0"][idx] / 2; f[:, 0, 10] = z["q0"][idx] / 2
    return torch.from_numpy(f)


def edge_features(b):
    """Directed edges of in-service lines (32 per sample -> 64 directed), features [r, x, dir, Pf, Qf, mask]."""
    z = b.raw
    idx = b.idx.numpy()
    tm = b.tmask[b.topo].numpy().astype(bool)
    lines = np.stack([np.where(r)[0] for r in tm])                  # [n, 32]
    src, dst = G.LINES[lines, 0], G.LINES[lines, 1]
    ei = np.concatenate([np.stack([src, dst], -1), np.stack([dst, src], -1)], 1)  # [n, 64, 2]
    zl = G.ZL[lines] * 100
    fm = np.zeros((b.n, 37, 3), np.float32)
    fm[:, G.FLOW_LINES, 0] = z["pf"][idx] * 5 * z["m_f"][idx]
    fm[:, G.FLOW_LINES, 1] = z["qf"][idx] * 5 * z["m_f"][idx]
    fm[:, G.FLOW_LINES, 2] = z["m_f"][idx]
    fl = np.take_along_axis(fm, lines[..., None], 1)
    fwd = np.concatenate([zl.real[..., None], zl.imag[..., None], np.ones_like(zl.real)[..., None], fl], -1)
    bwd = fwd.copy(); bwd[..., 2] = -1; bwd[..., 3:5] *= -1
    ef = np.concatenate([fwd, bwd], 1).astype(np.float32)
    return torch.from_numpy(ei).long(), torch.from_numpy(ef)


def mlp(i, h, o, nl=2):
    layers, d = [], i
    for _ in range(nl):
        layers += [nn.Linear(d, h), nn.SiLU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class MLPEstimator(nn.Module):
    def __init__(self, hid=256):
        super().__init__()
        self.net = nn.Sequential(mlp(NB * 11 + 37 + 12, hid, 2 * NB, nl=3))

    def forward(self, b):
        x = torch.cat([b.nf.flatten(1), b.tmask[b.topo], b.fm], 1)
        out = self.net(x)
        return out[:, :NB], out[:, NB:]


class GNNEstimator(nn.Module):
    """Edge-conditioned residual message-passing network on the feeder graph.
    Message m_st = W2 SiLU(A h_s + B h_t + C e_st); update h_t <- LN(h_t + U([h_t, sum_s m_st]))."""
    def __init__(self, hid=64, layers=12):
        super().__init__()
        self.enc = mlp(11, hid, hid, 1)
        self.eenc = mlp(6, hid, hid, 1)
        self.A = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers)])
        self.Bm = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(layers)])
        self.C = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(layers)])
        self.W2 = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers)])
        self.upd = nn.ModuleList([mlp(2 * hid, hid, hid, 1) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.dec = mlp(hid, hid, 2, 1)

    def forward(self, b):
        nf, ei, ef = b.nf, b.ei, b.ef
        B = b.n
        h = self.enc(nf).reshape(B * NB, -1)
        off = (torch.arange(B) * NB)[:, None]
        s = (ei[..., 0] + off).reshape(-1); t = (ei[..., 1] + off).reshape(-1)
        e = self.eenc(ef).reshape(B * 64, -1)
        for A, Bm, C, W2, upd, nm in zip(self.A, self.Bm, self.C, self.W2, self.upd, self.norm):
            m = W2(nn.functional.silu(A(h)[s] + Bm(h)[t] + C(e)))
            agg = torch.zeros_like(h).index_add_(0, t, m)
            h = nm(h + upd(torch.cat([h, agg], -1)))
        out = self.dec(h).reshape(B, NB, 2)
        return out[..., 0], out[..., 1]


def to_state(vo, to):
    Vm = 1 + vo / VSCALE
    th = to / TSCALE
    th = th - th[:, :1]                                              # slack angle reference
    return Vm, th


def prior_sigma(Vm, th, b, inflate=1.0):
    """Per-state prior std from validation errors of a learned estimator (x = [th_1..th_32, Vm_0..Vm_32])."""
    sv = (Vm - b.Vm).pow(2).mean(0).sqrt()
    st = (th - b.th).pow(2).mean(0).sqrt()[1:]
    return torch.cat([st, sv]).clamp(min=1e-5) * inflate


def map_refine(b, Vm, th, sigma, iters=10):
    """Physics-consistent inference: MAP estimate combining AC measurement model (likelihood) with the
    learned estimate as a Gaussian prior; Gauss-Newton warm-started at the network output."""
    x0 = torch.cat([th[:, 1:], Vm], 1)
    return wls_fast(b, iters=iters, x0=x0, prior=(x0.double(), sigma[None]))


def h_and_jac(x, Yc, tm):
    """Analytic measurement function and Jacobian (complex128). x: [B, 65]; Yc: [B, N, N] complex; tm: [B, 37]."""
    B = x.shape[0]
    th = torch.cat([x.new_zeros(B, 1), x[:, :NB - 1]], 1)
    Vm = x[:, NB - 1:]
    V = torch.polar(Vm, th)
    I = torch.einsum("bij,bj->bi", Yc, V)
    S = V * I.conj()
    Vn = V / Vm
    dVa = 1j * V[:, :, None] * (torch.diag_embed(I) - Yc * V[:, None, :]).conj()
    dVm = V[:, :, None] * (Yc * Vn[:, None, :]).conj() + torch.diag_embed(I.conj() * Vn)
    nl = len(G.FLOW_LINES)
    f, t = LF, LT
    y = torch.as_tensor(YF, dtype=torch.complex128)
    Vf, Vt = V[:, f], V[:, t]
    Sf = Vf * (y * (Vf - Vt)).conj() * tm[:, FL]
    M = 3 + len(G.VM_BUSES) + len(G.PMU_BUSES) + 2 * nl + 2 * (NB - 1)
    J = x.new_zeros(B, M, 2 * NB - 1)
    h = torch.cat([Vm[:, :1], S.real[:, :1], S.imag[:, :1], Vm[:, VMB], th[:, PMB], Sf.real, Sf.imag,
                   S.real[:, 1:], S.imag[:, 1:]], 1)
    J[:, 0, NB - 1] = 1
    J[:, 1, :NB - 1] = dVa[:, 0, 1:].real; J[:, 1, NB - 1:] = dVm[:, 0].real
    J[:, 2, :NB - 1] = dVa[:, 0, 1:].imag; J[:, 2, NB - 1:] = dVm[:, 0].imag
    r = 3
    for k, bb in enumerate(G.VM_BUSES):
        J[:, r + k, NB - 1 + bb] = 1
    r += len(G.VM_BUSES)
    for k, bb in enumerate(G.PMU_BUSES):
        J[:, r + k, bb - 1] = 1
    r += len(G.PMU_BUSES)
    yc = y.conj()
    dSf_dthf = -1j * Vf * yc * Vt.conj()
    dSf_dtht = 1j * Vf * yc * Vt.conj()
    dSf_dVf = (Vf / Vm[:, f]) * yc * (2 * Vf.conj() - Vt.conj())
    dSf_dVt = -Vf * yc * Vt.conj() / Vm[:, t]
    for k in range(nl):
        fk, tk = int(f[k]), int(t[k]); on = tm[:, FL[k]]
        for part, row in ((lambda z: z.real, r + k), (lambda z: z.imag, r + nl + k)):
            if fk > 0: J[:, row, fk - 1] += part(dSf_dthf[:, k]) * on
            if tk > 0: J[:, row, tk - 1] += part(dSf_dtht[:, k]) * on
            J[:, row, NB - 1 + fk] += part(dSf_dVf[:, k]) * on
            J[:, row, NB - 1 + tk] += part(dSf_dVt[:, k]) * on
    r += 2 * nl
    J[:, r:r + NB - 1, :NB - 1] = dVa[:, 1:, 1:].real; J[:, r:r + NB - 1, NB - 1:] = dVm[:, 1:].real
    r += NB - 1
    J[:, r:r + NB - 1, :NB - 1] = dVa[:, 1:, 1:].imag; J[:, r:r + NB - 1, NB - 1:] = dVm[:, 1:].imag
    return h, J


def wls_fast(b, iters=30, tol=1e-7, x0=None, prior=None, chunk=2000):
    outs = []
    for i in range(0, b.n, chunk):
        idx = torch.arange(i, min(b.n, i + chunk))
        s = b.sub(idx)
        x0c = None if x0 is None else x0[idx]
        pc = None if prior is None else (prior[0][idx], prior[1])
        outs.append(_wls_fast(s, iters, tol, x0c, pc))
    return (torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs]),
            torch.cat([o[2] for o in outs]), max(o[3] for o in outs))


def _wls_fast(b, iters, tol, x0, prior):
    B = b.n
    x = torch.cat([torch.zeros(B, NB - 1), torch.ones(B, NB)], 1).double() if x0 is None else x0.clone().double()
    Yc = torch.complex(b.Gs[b.topo].double(), b.Bs[b.topo].double())
    tm = b.tmask[b.topo].double()
    z, w2 = b.z.double(), b.w.double() ** 2
    conv = torch.zeros(B, dtype=torch.bool)
    for it in range(iters):
        h, J = h_and_jac(x, Yc, tm)
        JW = J.transpose(1, 2) * w2[:, None, :]
        A = JW @ J
        g = (JW @ (z - h)[..., None])[..., 0]
        if prior is not None:
            xp, sp = prior
            ip = (1 / sp.double() ** 2).expand_as(x)
            A = A + torch.diag_embed(ip)
            g = g - ip * (x - xp.double())
        dx = torch.linalg.solve(A + 1e-10 * torch.eye(A.shape[-1], dtype=A.dtype), g[..., None])[..., 0]
        dx = torch.nan_to_num(dx, nan=0.0)
        x = torch.where(conv[:, None], x, x + dx)
        conv = conv | (dx.abs().max(1).values < tol)
        if conv.all():
            break
    th = torch.cat([torch.zeros(B, 1, dtype=x.dtype), x[:, :NB - 1]], 1)
    return x[:, NB - 1:].float(), th.float(), conv, it + 1
