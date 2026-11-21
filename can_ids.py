"""CAN intrusion detector prototype. Traffic here is synthetic (20 periodic IDs) with the three
HCRL Car-Hacking attack types injected: DoS flood, fuzzing, spoofing. Train and test captures use
different seeds and attack timings. The HCRL recordings themselves are the next step."""
import numpy as np, time, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

IDS = np.array([0x0A0, 0x0C4, 0x130, 0x18F, 0x1A0, 0x260, 0x2A0, 0x2C0, 0x316, 0x329, 0x370, 0x430, 0x43F, 0x440, 0x4B1, 0x4F0, 0x545, 0x5A0, 0x5F0, 0x690])
PERIOD = np.array([10, 10, 10, 20, 20, 20, 20, 20, 10, 10, 50, 50, 10, 10, 20, 50, 100, 100, 200, 100]) / 1000.0

def capture(seconds, seed):
    r = np.random.default_rng(seed); t, i, lab = [], [], []
    for cid, p in zip(IDS, PERIOD):
        ts = np.arange(r.random() * p, seconds, p); ts = ts + r.normal(0, p * 0.03, len(ts))
        t.append(ts); i.append(np.full(len(ts), cid)); lab.append(np.zeros(len(ts), int))
    s = 5.0
    while s < seconds - 6:
        kind = r.integers(1, 4); dur = r.uniform(2, 5)
        if kind == 1:   ts = np.arange(s, s + dur, 0.0003); ids = np.zeros(len(ts), int)                    # DoS: ID 0x000 flood
        elif kind == 2: ts = np.arange(s, s + dur, 0.0005); ids = r.integers(0, 0x7FF, len(ts))             # fuzzing: random IDs
        else:           ts = np.arange(s, s + dur, 0.001);  ids = np.full(len(ts), r.choice([0x316, 0x43F])) # spoof RPM / gear ID
        t.append(ts); i.append(ids); lab.append(np.full(len(ts), kind)); s += dur + r.uniform(4, 10)
    t, i, lab = np.concatenate(t), np.concatenate(i), np.concatenate(lab); o = np.argsort(t)
    return t[o], i[o], lab[o]

KNOWN = set(IDS.tolist()); NOMINAL = dict(zip(IDS.tolist(), (0.1 / PERIOD).tolist()))
def features(t, ids, lab, win=0.1):
    X, y = [], []; edges = np.searchsorted(t, np.arange(0, t[-1], win))
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2: continue
        w = ids[a:b]; u, c = np.unique(w, return_counts=True); p = c / c.sum(); gaps = np.diff(t[a:b])
        ratio = max(cc / NOMINAL.get(int(uu), 1.0) for uu, cc in zip(u, c))
        X.append([b - a, len(u), -(p * np.log2(p)).sum(), np.mean([int(x) not in KNOWN for x in u]), gaps.min(), gaps.mean(), ratio, p.max()])
        y.append(int(lab[a:b].max() > 0))
    return np.array(X), np.array(y)

if __name__ == '__main__':
    Xtr, ytr = features(*capture(300, 1)); Xte, yte = features(*capture(300, 2))
    clf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=0, n_jobs=1).fit(Xtr, ytr)
    t0 = time.perf_counter(); p = clf.predict(Xte); dt = (time.perf_counter() - t0) / len(Xte) * 1e6
    tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
    res = dict(train_windows=len(ytr), test_windows=len(yte), attack_share_test=round(float(yte.mean()), 3),
               precision=round(precision_score(yte, p), 4), recall=round(recall_score(yte, p), 4), f1=round(f1_score(yte, p), 4),
               false_alarms=int(fp), missed=int(fn), microsec_per_window=round(dt, 1))
    print(json.dumps(res)); json.dump(res, open('ids_results.json', 'w'))
