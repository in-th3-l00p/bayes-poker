"""Load the opponent model exported by notebook.ipynb and play heads-up limit Hold'em with it.

    model = holdem_model.load()                      # models/hulhe_opponent_model.pt
    agent = holdem_model.EquityAgent(model.reader("population", learn=True))
    replica = holdem_model.Replica(model, "bot:Hyperborean-2011-2p-limit-tbr")

Agents talk to a game through four calls: start_hand(seat, cards, button), act(state),
observe(state, action) for every action of the other player, and end_hand(opponent_cards).
Actions are 0 (fold), 1 (check or call) and 2 (bet or raise). A state holds the street (0-3),
the board, whether the player faces a bet, the number of bets on the street, the pot, the
amount to call, whether a raise is allowed, whether the player is the small blind and whether
it acts second on this street. Cards are integers 0-51: rank * 4 + suit, ranks 2..A, suits cdhs.
"""

import itertools
import math
import pathlib

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "models" / "hulhe_opponent_model.pt"
LUT_PATH = HERE / "data" / "cache" / "rank5.npy"

RANKS = "23456789TJQKA"
SUITS = "cdhs"
ACTIONS = "fcr"


def parse_cards(s):
    return [RANKS.index(s[i]) * 4 + SUITS.index(s[i + 1]) for i in range(0, len(s), 2)]


def card_str(c):
    return RANKS[c // 4] + SUITS[c % 4]


COMBOS = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)])
NC = len(COMBOS)
COMBO_ID = np.full((52, 52), -1)
COMBO_ID[COMBOS[:, 0], COMBOS[:, 1]] = np.arange(NC)
COMBO_ID[COMBOS[:, 1], COMBOS[:, 0]] = np.arange(NC)
COMBO_HAS = np.zeros((NC, 52), dtype=bool)
COMBO_HAS[np.arange(NC)[:, None], COMBOS] = True
COMBOS_T = torch.from_numpy(COMBOS)
COMBO_HAS_T = torch.from_numpy(COMBO_HAS)


def consistent(visible):
    visible = [c for c in visible if c >= 0]
    return ~COMBO_HAS[:, visible].any(1)


R13 = np.arange(13)
POW13 = 13 ** np.arange(4, -1, -1)


def _top(mask, k):
    v = np.where(mask, R13, -1)
    return -np.sort(-v, axis=1)[:, :k]


def _code(cat, ranks):
    r = np.maximum(ranks, 0)
    r = np.concatenate([r, np.zeros((r.shape[0], 5 - r.shape[1]), dtype=np.int64)], 1)
    return cat * 13**5 + r @ POW13


def _straight_high(bits):
    m = (bits << 1) | ((bits >> 12) & 1)
    s = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4)
    hi = np.full(m.shape, -1, dtype=np.int64)
    for i in range(10):
        hi = np.where((s >> i) & 1 == 1, i + 3, hi)
    return hi


def rank_direct(cards):
    cards = np.asarray(cards, dtype=np.int64)
    n = cards.shape[0]
    onehot = np.zeros((n, 52), dtype=np.int64)
    onehot[np.arange(n)[:, None], cards] = 1
    rs = onehot.reshape(n, 13, 4).transpose(0, 2, 1)
    rc, sc = rs.sum(1), rs.sum(2)
    present = rc > 0
    bits = (present * (1 << R13)).sum(1)
    has_flush = sc.max(1) >= 5
    fmask = rs[np.arange(n), sc.argmax(1)] > 0
    sf_hi = np.where(has_flush, _straight_high((fmask * (1 << R13)).sum(1)), -1)
    st_hi = _straight_high(bits)
    quad = rc == 4
    qr = np.where(quad, R13, -1).max(1)
    trip = rc >= 3
    tr = np.where(trip, R13, -1).max(1)
    fh_pair = np.where((rc >= 2) & (R13 != tr[:, None]), R13, -1).max(1)
    pairs = rc == 2
    npairs = pairs.sum(1)
    top2 = _top(pairs, 2)
    out = _code(0, _top(present, 5))
    p1 = top2[:, :1]
    out = np.where(
        npairs >= 1, _code(1, np.concatenate([p1, _top(present & (R13 != p1), 3)], 1)), out
    )
    rest = present & (R13 != top2[:, :1]) & (R13 != top2[:, 1:2])
    out = np.where(npairs >= 2, _code(2, np.concatenate([top2, _top(rest, 1)], 1)), out)
    trips = np.concatenate([tr[:, None], _top(present & (R13 != tr[:, None]), 2)], 1)
    out = np.where(trip.any(1), _code(3, trips), out)
    out = np.where(st_hi >= 0, _code(4, st_hi[:, None]), out)
    out = np.where(has_flush, _code(5, _top(fmask, 5)), out)
    out = np.where(trip.any(1) & (fh_pair >= 0), _code(6, np.stack([tr, fh_pair], 1)), out)
    quads = np.concatenate([qr[:, None], _top(present & (R13 != qr[:, None]), 1)], 1)
    out = np.where(quad.any(1), _code(7, quads), out)
    out = np.where(sf_hi >= 0, _code(8, sf_hi[:, None]), out)
    return out


BINOM = torch.tensor([[math.comb(n, k) for k in range(6)] for n in range(53)])
SUBSETS = {k: torch.tensor(list(itertools.combinations(range(k), 5))) for k in (5, 6, 7)}


def _colex(sorted5):
    return sum(BINOM[sorted5[..., j], j + 1] for j in range(5))


def _load_lut():
    if LUT_PATH.exists():
        return torch.from_numpy(np.load(LUT_PATH))
    five = torch.tensor(list(itertools.combinations(range(52), 5)))
    lut = torch.empty(len(five), dtype=torch.int64)
    for i in range(0, len(five), 200_000):
        chunk = five[i : i + 200_000]
        lut[_colex(chunk)] = torch.from_numpy(rank_direct(chunk.numpy()))
    LUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(LUT_PATH, lut.numpy())
    return lut


LUT = _load_lut()


def hand_rank(cards, chunk=1 << 17):
    cards = torch.as_tensor(cards)
    sub = SUBSETS[cards.shape[1]]
    out = torch.empty(cards.shape[0], dtype=torch.int64)
    for i in range(0, cards.shape[0], chunk):
        c = torch.sort(cards[i : i + chunk], dim=1).values[:, sub]
        out[i : i + chunk] = LUT[_colex(c)].max(1).values
    return out


def strength(boards):
    boards = torch.as_tensor(boards).reshape(-1, 5)
    nb = boards.shape[0]
    invalid = COMBO_HAS_T[:, boards].any(-1).T
    taken = torch.zeros(nb, 52, dtype=torch.bool)
    taken[torch.arange(nb)[:, None], boards] = True
    spare = torch.sort(taken.double(), dim=1, stable=True).indices[:, :2]
    pairs = torch.where(invalid[..., None], spare[:, None, :], COMBOS_T[None])
    cards = torch.cat([pairs, boards[:, None, :].expand(nb, NC, 5)], 2).reshape(-1, 7)
    score = hand_rank(cards).reshape(nb, NC).double()
    score = torch.where(invalid, torch.full_like(score, 1e18), score).contiguous()
    srt = torch.sort(score, 1).values
    lo = torch.searchsorted(srt, score, right=False)
    hi = torch.searchsorted(srt, score, right=True)
    n = (~invalid).sum(1, keepdim=True)
    hs = (lo.double() + 0.5 * (hi - lo - 1).double()) / (n - 1)
    return torch.where(invalid, torch.full_like(hs, float("nan")), hs)


def exact_turn_hs2(boards):
    nb = len(boards)
    taken = torch.zeros(nb, 52, dtype=torch.bool)
    taken[torch.arange(nb)[:, None], boards] = True
    rivers = torch.sort(taken.double(), dim=1, stable=True).indices[:, :48]
    full = torch.cat([boards[:, None, :].expand(nb, 48, 4), rivers[:, :, None]], 2)
    hs = strength(full.reshape(-1, 5)).reshape(nb, 48, NC)
    return torch.nanmean(hs**2, dim=1)


def to_buckets(value, k):
    value = torch.as_tensor(value)
    invalid = torch.isnan(value)
    v = torch.where(invalid, torch.full_like(value, 1e18), value).contiguous()
    srt = torch.sort(v, -1).values
    lo = torch.searchsorted(srt, v, right=False)
    hi = torch.searchsorted(srt, v, right=True)
    n = (~invalid).sum(-1, keepdim=True)
    pct = (lo.double() + 0.5 * (hi - lo).double()) / n
    b = torch.clamp((pct * k).floor().long(), max=k - 1)
    return torch.where(invalid, torch.full_like(b, -1), b)


SUIT_PERMS = torch.tensor(list(itertools.permutations(range(4))))
CARD_PERM = (torch.arange(52) // 4 * 4)[None, :] + SUIT_PERMS[:, torch.arange(52) % 4]
COMBO_PERM = torch.from_numpy(COMBO_ID)[CARD_PERM[:, COMBOS_T[:, 0]], CARD_PERM[:, COMBOS_T[:, 1]]]
BOARD_CODE = 52 ** torch.arange(5)


def canonical_code(boards):
    permuted = torch.sort(CARD_PERM[:, boards], dim=-1).values
    code = (permuted * BOARD_CODE[: boards.shape[1]]).sum(-1)
    best = code.argmin(0)
    canon = permuted[best, torch.arange(len(boards))]
    return (canon * BOARD_CODE[: boards.shape[1]]).sum(-1), best


class Buckets:
    def __init__(self, artifact):
        self.k = artifact["K"]
        self.preflop = artifact["preflop_bucket"].long()
        self.flop_codes = artifact["flop_codes"]
        self.flop_table = artifact["flop_table"]
        self.cache = {}

    def __call__(self, board):
        key = tuple(board)
        if key not in self.cache:
            self.cache[key] = self._compute(list(board))
        return self.cache[key]

    def _compute(self, board):
        if len(board) == 0:
            return self.preflop
        code, perm = canonical_code(torch.tensor([board]))
        if len(board) == 3:
            row = self.flop_table[torch.searchsorted(self.flop_codes, code)[0]].long()
        else:
            canon = torch.stack([(code // 52**j) % 52 for j in range(len(board))], 1)
            value = exact_turn_hs2(canon) if len(board) == 4 else strength(canon)
            row = to_buckets(value, self.k)[0]
        return row[COMBO_PERM[perm[0]]]


def context(street, facing, bets, position):
    return ((street * 2 + int(facing)) * 5 + min(bets, 4)) * 2 + int(position)


def position_bit(convention, state):
    if convention == "irc" or state["street"] > 0:
        return int(state["acts_second"])
    return int(state["is_sb"])


def transition(eps, m):
    if m == 1:
        return torch.ones(1, 1)
    return torch.full((m, m), eps / (m - 1)) + torch.eye(m) * (1 - eps - eps / (m - 1))


class Model:
    def __init__(self, artifact):
        self.artifact = artifact
        self.legal = artifact["legal"]
        self.k = artifact["K"]
        self.buckets = Buckets(artifact)
        self.names = list(artifact["models"])

    def reader(self, name="population", learn=False, prior_strength=None):
        return CardReader(self, name, learn, prior_strength)


def load(path=DEFAULT_PATH):
    return Model(torch.load(path, weights_only=False))


class CardReader:
    def __init__(self, model, name=None, learn=False, prior_strength=None):
        self.model, self.learn = model, learn
        if name is None:
            self.theta = None
            return
        m = model.artifact["models"][name]
        self.convention = m["convention"]
        self.theta = m["theta"].double()
        if prior_strength is None:
            prior_strength = model.artifact.get("online_prior_strength", 1.0)
        self.alpha = prior_strength * self.theta.clone() if learn else None
        self.eps = m["eps"]
        self.log_w = torch.log(m["rho"].double()) if self.theta.shape[0] > 1 else None

    def start_hand(self):
        self.events = []

    def record(self, state, action):
        if self.theta is None:
            return
        x = context(
            state["street"], state["facing"], state["bets"], position_bit(self.convention, state)
        )
        if self.model.legal[x, action] and state["bets"] <= 4:
            self.events.append((x, action, tuple(state["board"])))

    def loglik(self, visible):
        valid = torch.from_numpy(consistent(visible))
        if self.theta is None:
            return torch.where(valid, 0.0, -torch.inf)[None].double()
        out = torch.zeros(self.theta.shape[0], NC, dtype=torch.float64)
        for x, a, board in self.events:
            out += torch.log(self.theta[:, x, :, a].clamp(min=1e-300))[:, self.model.buckets(board)]
        if self.log_w is not None:
            out = out + self.log_w[:, None]
        return torch.where(valid[None], out, -torch.inf)

    def pmf(self, visible):
        return torch.softmax(torch.logsumexp(self.loglik(visible), 0), 0)

    def end_hand(self, visible, opponent_cards):
        if self.theta is None:
            return
        L = self.loglik(visible)
        if opponent_cards is not None:
            true = int(COMBO_ID[opponent_cards[0], opponent_cards[1]])
            keep = torch.full_like(L, -torch.inf)
            keep[:, true] = L[:, true]
            L = keep
        post = torch.softmax(L.reshape(-1), 0).reshape(L.shape)
        if self.log_w is not None:
            filt = torch.log(post.sum(1).clamp(min=1e-300))
            log_g = torch.log(transition(self.eps, self.theta.shape[0]))
            self.log_w = torch.logsumexp(filt[:, None] + log_g, 0)
        if self.learn:
            for x, a, board in self.events:
                b = self.model.buckets(board).clamp(min=0)
                add = torch.zeros(post.shape[0], self.model.k, dtype=torch.float64)
                self.alpha[:, x, :, a] += add.scatter_add_(1, b[None].expand_as(post), post)
            self.theta = self.alpha / self.alpha.sum(-1, keepdim=True)


def equity(cards, board, pmf, rollouts, rng):
    if len(board) == 5:
        runouts = torch.tensor([board])
    else:
        rest = np.setdiff1d(np.arange(52), cards + board)
        extra = np.stack([rng.choice(rest, 5 - len(board), replace=False) for _ in range(rollouts)])
        known = torch.tensor(board, dtype=torch.long).expand(rollouts, len(board))
        runouts = torch.cat([known, torch.from_numpy(extra)], 1)
    n = len(runouts)
    mine = torch.tensor(cards).expand(n, 2)
    ours = hand_rank(torch.cat([mine, runouts], 1))
    clash = COMBO_HAS_T[:, runouts].any(-1).T
    taken = torch.zeros(n, 52, dtype=torch.bool)
    taken[torch.arange(n)[:, None], torch.cat([runouts, mine], 1)] = True
    spare = torch.sort(taken.double(), dim=1, stable=True).indices[:, :2]
    pairs = torch.where(clash[..., None], spare[:, None, :], COMBOS_T[None])
    theirs = hand_rank(torch.cat([pairs, runouts[:, None, :].expand(n, NC, 5)], 2).reshape(-1, 7))
    theirs = theirs.reshape(n, NC)
    w = pmf[None] * ~clash
    score = (ours[:, None] > theirs).double() + 0.5 * (ours[:, None] == theirs).double()
    return float((w * score).sum() / w.sum())


class EquityAgent:
    def __init__(self, reader, rollouts=32, bet_q=0.6, raise_q=0.7, seed=0):
        self.reader, self.rollouts, self.bet_q, self.raise_q = reader, rollouts, bet_q, raise_q
        self.rng = np.random.default_rng(seed)
        self.last_q = None

    def start_hand(self, seat, cards, button):
        self.cards = list(cards)
        self.reader.start_hand()

    def act(self, state):
        board = list(state["board"])
        pmf = self.reader.pmf(self.cards + board)
        q = equity(self.cards, board, pmf, self.rollouts, self.rng)
        self.last_q, self.last_pmf = q, pmf
        if state["facing"]:
            if q < state["to_call"] / (state["pot"] + state["to_call"]):
                return 0
            return 2 if state["can_raise"] and q > self.raise_q else 1
        return 2 if state["can_raise"] and q > self.bet_q else 1

    def observe(self, state, action):
        self.reader.record(state, action)

    def end_hand(self, opponent_cards):
        self.reader.end_hand(self.cards, opponent_cards)


class Replica:
    def __init__(self, model, name, seed=0):
        m = model.artifact["models"][name]
        self.model, self.convention = model, m["convention"]
        self.theta = m["theta"][0].double()
        self.rng = np.random.default_rng(seed)

    def start_hand(self, seat, cards, button):
        self.combo = int(COMBO_ID[cards[0], cards[1]])

    def act(self, state):
        x = context(
            state["street"], state["facing"], state["bets"], position_bit(self.convention, state)
        )
        s = int(self.model.buckets(list(state["board"]))[self.combo])
        p = (self.theta[x, s] * self.model.legal[x]).numpy()
        if not state["can_raise"]:
            p[2] = 0.0
        if not state["facing"]:
            p[0] = 0.0
        return int(self.rng.choice(3, p=p / p.sum()))

    def observe(self, state, action):
        pass

    def end_hand(self, opponent_cards):
        pass
