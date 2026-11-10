#!/usr/bin/env python3
"""Clasificatori bayesieni naiv si nenaiv: ghicim cat de bune sunt cartile unui jucator de poker.

Date: 40 de meciuri ACPC 2011 din setul PHH (Zenodo, DOI 10.5281/zenodo.10796885).
Clasa: scorul Chen al celor doua carti, in 3 grupe (slaba, medie, puternica).
Caracteristici: s = 1 daca e small blind, r0..r3 = actiunea din fiecare runda
(0 nimic, 1 fold, 2 check/call, 3 bet/raise).

naiv:    P(c) P(s|c) P(r0|c)   P(r1|c)    P(r2|c)    P(r3|c)
nenaiv:  P(c) P(s|c) P(r0|c,s) P(r1|c,r0) P(r2|c,r1) P(r3|c,r2)

Rulare: python3 bayes_poker.py (Python 3.9+, numpy, matplotlib).
Prima rulare descarca circa 45 MB in ./data. Seed fixat: SEED = 0.
"""

import io
import json
import pathlib
import re
import urllib.request
import zipfile

import numpy as np

SEED = 0
N_MATCHES = 40
TEST_FRACTION = 0.3
LAPLACE = 1.0

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE = HERE / "figures" / "calibration.pdf"
ZENODO_RECORD = "https://zenodo.org/api/records/10796885"
MATCH_FILES = re.compile(r".*acpc_2011_2p_limit.*\.phhs$")

CLASSES = ["slabă", "medie", "puternică"]
ACTIONS = ["-", "fold", "check/call", "bet/raise"]
RANKS = "23456789TJQKA"

FEATURES = ["s", "r0", "r1", "r2", "r3"]
N_VALUES = {"s": 2, "r0": 4, "r1": 4, "r2": 4, "r3": 4}
NAIVE_PARENTS = {"s": None, "r0": None, "r1": None, "r2": None, "r3": None}
NON_NAIVE_PARENTS = {"s": None, "r0": "s", "r1": "r0", "r2": "r1", "r3": "r2"}


# 1. Descarcarea datelor: din arhiva de 20 GB citim doar fisierele necesare (HTTP Range).


class RemoteFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = max(0, [0, self.pos, self.size][whence] + offset)
        return self.pos

    def _get(self, start, end):
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(request) as r:
            return r.read()

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * self.block:
            data = self._get(self.pos, self.pos + n)
            self.pos += len(data)
            return data
        out = bytearray()
        while n > 0:
            index, offset = divmod(self.pos, self.block)
            if index not in self.cache:
                if len(self.cache) > 64:
                    self.cache.clear()
                start = index * self.block
                self.cache[index] = self._get(start, min(start + self.block, self.size))
            chunk = self.cache[index][offset : offset + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def download():
    DATA_DIR.mkdir(exist_ok=True)
    have = sorted(DATA_DIR.glob("*.phhs"))
    if len(have) >= N_MATCHES:
        return have[:N_MATCHES]
    print("Descarc meciurile ACPC 2011 de pe Zenodo (circa 45 MB)...", flush=True)
    with urllib.request.urlopen(ZENODO_RECORD) as r:
        files = json.load(r)["files"]
    url = next(f["links"]["self"] for f in files if f["key"].endswith(".zip"))
    archive = zipfile.ZipFile(RemoteFile(url))
    members = [m for m in archive.infolist() if MATCH_FILES.match(m.filename)][:N_MATCHES]
    for k, member in enumerate(members, 1):
        target = DATA_DIR / pathlib.PurePosixPath(member.filename).name
        if not (target.exists() and target.stat().st_size == member.file_size):
            target.write_bytes(archive.read(member))
        print(f"  [{k:2d}/{len(members)}] {target.name}", flush=True)
    return sorted(DATA_DIR.glob("*.phhs"))[:N_MATCHES]


# 2. Clasa: scorul Chen (A-A = 20, 7-2 de culori diferite = -1.5).


def chen_score(cards):
    high, low = sorted((RANKS.index(cards[0]), RANKS.index(cards[2])), reverse=True)
    score = {12: 10, 11: 8, 10: 7, 9: 6}.get(high, (high + 2) / 2)
    if high == low:
        return max(2 * score, 5)
    if cards[1] == cards[3]:
        score += 2
    gap = high - low - 1
    score -= [0, 1, 2, 4, 5][min(gap, 4)]
    if gap <= 1 and high < 10:
        score += 1
    return score


def class_thresholds():
    deck = [rank + suit for rank in RANKS for suit in "cdhs"]
    scores = [chen_score(a + b) for i, a in enumerate(deck) for b in deck[i + 1 :]]
    return np.quantile(scores, [1 / 3, 2 / 3])


# 3. Citirea mainilor: 'd dh p1 4c2h' carti, 'p1 f' fold, 'p1 cc' check/call,
# 'p2 cbr 20' bet/raise, 'd db ...' carti pe masa, adica runda noua.


def read_hands(path):
    return [re.findall(r"'([^']*)'", line) for line in path.read_text().splitlines()
            if line.startswith("actions = ")]


def hand_to_records(actions):
    cards = {}
    rounds = {1: [0, 0, 0, 0], 2: [0, 0, 0, 0]}
    current_round = 0
    code = {"f": 1, "cc": 2, "cbr": 3}
    for action in actions:
        parts = action.split()
        if parts[:2] == ["d", "dh"]:
            cards[int(parts[2][1])] = parts[3]
        elif parts[:2] == ["d", "db"]:
            current_round += 1
        elif parts[1] in code:
            seat = int(parts[0][1])
            rounds[seat][current_round] = max(rounds[seat][current_round], code[parts[1]])
    return [(chen_score(cards[seat]), int(seat == 2), *rounds[seat]) for seat in (1, 2)]


def load(files):
    rows, deal_seeds = [], []
    for path in files:
        seed = int(re.search(r"run-(\d+)", path.name).group(1))
        for actions in read_hands(path):
            for record in hand_to_records(actions):
                rows.append(record)
                deal_seeds.append(seed)
    rows = np.array(rows, dtype=float)
    classes = np.digitize(rows[:, 0], class_thresholds(), right=True)
    features = rows[:, 1:].astype(int)
    return classes, features, np.array(deal_seeds)


# 4. Clasificatorul: acelasi cod pentru ambele modele, difera doar parintii.
# Probabilitatile se invata prin numarare, cu +1 (Laplace) ca niciuna sa nu fie 0.


class BayesClassifier:
    def __init__(self, parents):
        self.parents = parents

    def _parent_values(self, X, feature):
        parent = self.parents[feature]
        return X[:, FEATURES.index(parent)] if parent else np.zeros(len(X), int)

    def fit(self, classes, X):
        k = len(CLASSES)
        class_counts = np.bincount(classes, minlength=k) + LAPLACE
        self.log_prior = np.log(class_counts / class_counts.sum())
        self.log_tables = {}
        for j, feature in enumerate(FEATURES):
            parent = self.parents[feature]
            n_parent = N_VALUES[parent] if parent else 1
            counts = np.full((k, n_parent, N_VALUES[feature]), LAPLACE)
            np.add.at(counts, (classes, self._parent_values(X, feature), X[:, j]), 1)
            self.log_tables[feature] = np.log(counts / counts.sum(axis=2, keepdims=True))
        return self

    def probability(self, feature, value, parent_value=0):
        return np.exp(self.log_tables[feature][:, parent_value, value])

    def predict_proba(self, X):
        log_joint = np.tile(self.log_prior, (len(X), 1))
        for j, feature in enumerate(FEATURES):
            table = self.log_tables[feature]
            log_joint += table[:, self._parent_values(X, feature), X[:, j]].T
        log_joint -= log_joint.max(axis=1, keepdims=True)
        p = np.exp(log_joint)
        return p / p.sum(axis=1, keepdims=True)


# 5. Masuri: acuratete, log-loss, Brier, incredere, calibrare, bootstrap.


def evaluate(p, classes):
    predicted = p.argmax(axis=1)
    correct = p[np.arange(len(classes)), classes]
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
    np.add.at(confusion, (classes, predicted), 1)
    return {
        "acuratețe": float((predicted == classes).mean()),
        "log-loss": float(-np.log(correct).mean()),
        "Brier": float(((p - np.eye(len(CLASSES))[classes]) ** 2).sum(axis=1).mean()),
        "încredere": float(p.max(axis=1).mean()),
        "confuzie": confusion,
    }


def bootstrap_interval(p_naive, p_non_naive, classes, rng, repeats=1000):
    rows = np.arange(len(classes))
    loss_naive = -np.log(p_naive[rows, classes])
    loss_non_naive = -np.log(p_non_naive[rows, classes])
    hit_naive = (p_naive.argmax(axis=1) == classes).astype(float)
    hit_non_naive = (p_non_naive.argmax(axis=1) == classes).astype(float)
    samples = rng.integers(0, len(classes), (repeats, len(classes)))
    loss_gain = (loss_naive[samples] - loss_non_naive[samples]).mean(axis=1)
    accuracy_gain = (hit_non_naive[samples] - hit_naive[samples]).mean(axis=1)
    return np.percentile(accuracy_gain, [2.5, 97.5]), np.percentile(loss_gain, [2.5, 97.5])


def calibration(p, classes, bins=10):
    confidence = p.max(axis=1)
    hit = p.argmax(axis=1) == classes
    edges = np.linspace(1 / len(CLASSES), 1, bins + 1)
    which = np.clip(np.digitize(confidence, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        in_bin = which == b
        if in_bin.sum() >= 50:
            rows.append((confidence[in_bin].mean(), hit[in_bin].mean(), int(in_bin.sum())))
    return np.array(rows)


def plot_calibration(probabilities, classes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot([1 / 3, 1], [1 / 3, 1], "k--", lw=0.8, label="calibrare perfectă")
    for name, color in (("naiv", "tab:red"), ("nenaiv", "tab:blue")):
        points = calibration(probabilities[name], classes)
        ax.plot(points[:, 0], points[:, 1], "o-", color=color, ms=4, label=name)
    ax.set_xlabel("încrederea modelului în clasa prezisă")
    ax.set_ylabel("cât de des are dreptate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)


def show(label, values):
    return f"{label}: " + "  ".join(f"{c}={v:.3f}" for c, v in zip(CLASSES, values))


# 6. Experimentul: impartim train/test dupa seed-ul cartilor, ca o mana si
# geamana ei (aceleasi carti, locuri schimbate) sa ramana de aceeasi parte.


def main():
    rng = np.random.default_rng(SEED)
    files = download()
    classes, X, deal_seeds = load(files)

    seeds = np.unique(deal_seeds)
    test_seeds = rng.choice(seeds, round(TEST_FRACTION * len(seeds)), replace=False)
    test = np.isin(deal_seeds, test_seeds)
    train = ~test

    print(f"\n{len(files)} meciuri, {len(classes):,} înregistrări "
          f"({len(classes) // 2:,} mâini, câte una pentru fiecare jucător)")
    print("ponderea claselor: " + ", ".join(
        f"{c} {share:.0%}" for c, share in zip(CLASSES, np.bincount(classes) / len(classes))))
    print(f"antrenare: {train.sum():,} înregistrări, test: {test.sum():,} "
          f"({len(test_seeds)} din {len(seeds)} seed-uri de împărțire)\n")

    models = {
        "naiv": BayesClassifier(NAIVE_PARENTS).fit(classes[train], X[train]),
        "nenaiv": BayesClassifier(NON_NAIVE_PARENTS).fit(classes[train], X[train]),
    }
    prior = np.bincount(classes[train], minlength=len(CLASSES)) / train.sum()
    probabilities = {"doar a priori": np.tile(prior, (test.sum(), 1))}
    for name, model in models.items():
        probabilities[name] = model.predict_proba(X[test])
    y = classes[test]

    print("REZULTATE PE DATELE DE TEST")
    print(f"{'model':>14} | acuratețe | log-loss | Brier | încredere")
    for name, p in probabilities.items():
        s = evaluate(p, y)
        print(f"{name:>14} | {s['acuratețe']:9.3f} | {s['log-loss']:8.3f} | "
              f"{s['Brier']:5.3f} | {s['încredere']:9.3f}")

    accuracy_gain, loss_gain = bootstrap_interval(
        probabilities["naiv"], probabilities["nenaiv"], y, rng)
    print(f"\nnenaiv față de naiv, intervale de încredere 95% (bootstrap, 1000 de repetări):")
    print(f"  câștig de acuratețe: [{accuracy_gain[0]:+.3f}, {accuracy_gain[1]:+.3f}]")
    print(f"  scădere a log-loss:  [{loss_gain[0]:+.3f}, {loss_gain[1]:+.3f}]")

    for name in models:
        print(f"\nmatricea de confuzie, {name} (rânduri: clasa reală, coloane: clasa prezisă)")
        print(evaluate(probabilities[name], y)["confuzie"])

    folded = X[test][:, FEATURES.index("r0")] == 1
    print(f"\nDOAR ÎNREGISTRĂRILE CU FOLD ÎNAINTE DE FLOP ({folded.sum():,})")
    for name in models:
        s = evaluate(probabilities[name][folded], y[folded])
        print(f"  {name:>6}: acuratețe {s['acuratețe']:.3f}, log-loss {s['log-loss']:.3f}, "
              f"încredere {s['încredere']:.3f}")

    naive, non_naive = models["naiv"], models["nenaiv"]
    print("\nPROBABILITĂȚI ÎNVĂȚATE (folosite în raport)")
    print(show("  naiv, P(small blind | clasă)", naive.probability("s", 1)))
    print(show("  naiv, P(fold preflop | clasă)", naive.probability("r0", 1)))
    print(show("  naiv, P(nicio acțiune pe flop | clasă)", naive.probability("r1", 0)))
    print(show("  naiv, P(bet pe flop | clasă)", naive.probability("r1", 3)))
    print(show("  nenaiv, P(nicio acțiune pe flop | clasă, fold preflop)",
               non_naive.probability("r1", 0, parent_value=1)))
    print(show("  nenaiv, P(bet pe flop | clasă, check/call preflop)",
               non_naive.probability("r1", 3, parent_value=2)))
    print(show("  nenaiv, P(bet pe flop | clasă, bet/raise preflop)",
               non_naive.probability("r1", 3, parent_value=3)))
    print(show("  nenaiv, P(fold preflop | clasă, small blind)",
               non_naive.probability("r0", 1, parent_value=1)))
    print(show("  nenaiv, P(fold preflop | clasă, big blind)",
               non_naive.probability("r0", 1, parent_value=0)))

    print("\nCALIBRARE (încredere -> cât de des are dreptate, număr de înregistrări)")
    for name in models:
        print(f"  {name}:")
        for confidence, hit, n in calibration(probabilities[name], y):
            print(f"    {confidence:.2f} -> {hit:.2f}  ({int(n):,})")

    plot_calibration(probabilities, y)
    print(f"\nfigura: {FIGURE}")


if __name__ == "__main__":
    main()
