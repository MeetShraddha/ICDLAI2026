import numpy as np, torch, time, math
import grid as G, models as M



def train_model(kind, d_tr, d_va, T, Ys, seed, lam=0.0, label_frac=1.0, epochs=40, bs=128,
                lr=2e-3, resample_every=4, log=None, verbose=False):
    """kind in {'mlp','gnn'}; lam>0 adds the physics (measurement-consistency) loss -> PI-GNN / PI-MLP.
    label_frac<1: only that fraction of training states has ground-truth labels (physics loss uses all)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = M.MLPEstimator() if kind == "mlp" else M.GNNEstimator()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = len(d_tr["topo"])
    lab = torch.zeros(n, dtype=torch.bool)
    lab[torch.from_numpy(rng.permutation(n)[: max(1, int(round(label_frac * n)))])] = True
    steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    zva = G.measure(d_va, T, np.random.default_rng(999), drop=np.random.default_rng(998).uniform(0, 0.6, len(d_va["topo"])))
    bva = M.Batch(d_va, zva, T, Ys)
    best, best_state = 1e9, None
    for ep in range(epochs):
        if ep % resample_every == 0:        # fresh noise + random meter outages (augmentation)
            z = G.measure(d_tr, T, rng, drop=rng.uniform(0, 0.6, n))
            btr = M.Batch(d_tr, z, T, Ys)
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b = btr.sub(idx)
            vo, to = model(b)
            Vm, th = M.to_state(vo, to)
            b_lab = lab[idx]
            loss = torch.zeros(())
            if b_lab.any():
                sup = ((Vm - b.Vm) * M.VSCALE) ** 2 + ((th - b.th) * M.TSCALE) ** 2
                loss = loss + sup[b_lab].mean()
            if lam > 0:
                ramp = min(1.0, ep / max(1, int(0.3 * epochs)))        # curriculum: physics term ramps in
                r = M.residual(b, Vm, th)
                loss = loss + ramp * lam * torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=3.0, reduction="mean")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            Vm, th = predict(model, bva)
            val = ((Vm - bva.Vm).abs().mean() * M.VSCALE + (th - bva.th).abs().mean() * M.TSCALE).item()
        if val < best:
            best = val; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"{kind} lam={lam} ep{ep} loss={loss.item():.4f} val={val:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, b, bs=1024):
    model.eval()
    outs = []
    for i in range(0, b.n, bs):
        s = b.sub(torch.arange(i, min(b.n, i + bs)))
        outs.append(M.to_state(*model(s)))
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])


def metrics(Vm, th, b, conv=None):
    ev = (Vm - b.Vm).abs(); et = (th - b.th).abs()
    r = M.residual(b, Vm, th)
    rt = r[:, :3 + len(G.VM_BUSES) + len(G.PMU_BUSES) + 2 * len(G.FLOW_LINES)]
    wt = b.w[:, :rt.shape[1]] > 0
    out = dict(
        mae_v=ev.mean().item() * 1e3, rmse_v=ev.pow(2).mean().sqrt().item() * 1e3,
        mae_t=et.mean().item() * 1e3, rmse_t=et.pow(2).mean().sqrt().item() * 1e3,
        p99_maxv=torch.quantile(ev.max(1).values, 0.99).item() * 1e3,
        viol=(ev.max(1).values > 0.01).float().mean().item() * 100,   # % snapshots with any |dV| > 0.01 pu
        chi_rt=((rt ** 2).sum(1) / wt.sum(1).clamp(min=1)).mean().item(),  # normalised real-time residual
    )
    if conv is not None:
        out["fail"] = (1 - conv.float().mean().item()) * 100
    return out
