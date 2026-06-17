def mk(Mv, Kv):
    parts = []; cnt = np.empty(Mv, np.uint32)
    for m in range(Mv):
        f = np.sort(rng.choice(NI, Kv, replace=False)).astype(np.uint32)
        parts.append(f); cnt[m] = Kv
    idx = np.concatenate(parts); off = np.zeros(Mv, np.uint32)
    np.cumsum(cnt[:-1], out=off[1:])
    return idx.astype(np.uint32), off, cnt, np.ones(idx.size, np.float32)

def tm(Mv, args, n=20, w=3):
    idx, off, cnt, vals = args
    fn = lambda: g.spike_propagate_batch(dev, idx.view(np.float32), off.view(np.float32),
                                         cnt.view(np.float32), W, vals, NI, NO, Mv)
    for _ in range(w): fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1e3

for (Mv, Kv) in [(512, 1), (512, 40), (64, 40), (16, 40), (512, 200)]:
    print("M=%4d K=%3d  ms=%.3f" % (Mv, Kv, tm(Mv, mk(Mv, Kv))))
