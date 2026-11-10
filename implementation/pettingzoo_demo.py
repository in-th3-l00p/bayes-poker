"""The exported opponent model playing heads-up limit Hold'em in PettingZoo (texas_holdem_v4).

    python pettingzoo_demo.py [--deals 150] [--out illustrations]      (or: devbox run demo)

Writes to the output folder:
    winnings.png   cumulative big blinds of three agents against four opponents
    hand.gif       one hand, the table as PettingZoo draws it next to the model's beliefs
    hand.png       the same hand, one panel per decision
    summary.txt    the table printed at the end
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from pettingzoo.classic import texas_holdem_v4

import holdem_model as hm

HERE = pathlib.Path(__file__).resolve().parent
CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3


def card_index(card):
    return hm.RANKS.index(card.rank) * 4 + hm.SUITS.index(card.suit.lower())


def to_env(code, mask, facing):
    if code == 2 and mask[RAISE]:
        return RAISE
    if code == 0 and facing and mask[FOLD]:
        return FOLD
    return CHECK if mask[CHECK] else CALL


def from_env(action):
    return {CALL: 1, CHECK: 1, RAISE: 2, FOLD: 0}[action]


class Passive:
    def start_hand(self, seat, cards, button):
        pass

    def act(self, state):
        return 1

    def observe(self, state, action):
        pass

    def end_hand(self, opponent_cards):
        pass


class Maniac(Passive):
    def act(self, state):
        return 2 if state["can_raise"] else 1


class Random(Passive):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        choices = [1] + [2] * state["can_raise"] + [0] * state["facing"]
        return int(self.rng.choice(choices))


def play_hand(env, agents, seed, recorder=None):
    env.reset(seed=seed)
    game = env.unwrapped.env.game
    index = {name: i for i, name in enumerate(env.possible_agents)}
    sb = next(i for i, p in enumerate(game.players) if p.in_chips == game.small_blind)
    hands = [[card_index(c) for c in p.hand] for p in game.players]
    for i, agent in enumerate(agents):
        agent.start_hand(i, hands[i], sb)
    first, folded, rewards = {}, False, {}
    for name in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if term or trunc:
            rewards[index[name]] = reward
            env.step(None)
            continue
        i = index[name]
        street = game.round_counter
        first.setdefault(street, i)
        to_call = max(game.round.raised) - game.round.raised[i]
        bets = game.round.have_raised + (1 if street == 0 else 0)
        mask = obs["action_mask"]
        state = {
            "street": street,
            "board": [card_index(c) for c in game.public_cards],
            "facing": to_call > 0,
            "bets": bets,
            "pot": float(sum(p.in_chips for p in game.players)),
            "to_call": float(to_call),
            "can_raise": bool(mask[RAISE]) and bets < 4,
            "is_sb": i == sb,
            "acts_second": i != first[street],
        }
        action = to_env(agents[i].act(state), mask, state["facing"])
        folded |= action == FOLD
        agents[1 - i].observe(state, from_env(action))
        if recorder is not None:
            recorder(env, agents, i, state, action, hands)
        env.step(action)
    for i, agent in enumerate(agents):
        agent.end_hand(None if folded else hands[1 - i])
    if recorder is not None:
        recorder(env, agents, None, None, None, hands)
    return rewards[0]


def play_match(make_agent, make_opponent, deals, seed):
    env = texas_holdem_v4.env()
    agent, opponent = make_agent(), make_opponent()
    gains = np.zeros((deals, 2))
    for d in range(deals):
        for seat in (0, 1):
            agent.rng = np.random.default_rng([seed, d, seat, 1])
            opponent.rng = np.random.default_rng([seed, d, seat, 2])
            players = [agent, opponent] if seat == 0 else [opponent, agent]
            g = play_hand(env, players, seed=seed * 100_000 + d)
            gains[d, seat] = g if seat == 0 else -g
    return gains


def raise_share(model, name):
    theta = model.artifact["models"][name]["theta"][0]
    return float(
        (theta[..., 2] * model.legal[:, None, 2]).sum() / (model.legal[:, 2].sum() * theta.shape[1])
    )


def class_grid(p):
    hi = np.maximum(hm.COMBOS[:, 0] // 4, hm.COMBOS[:, 1] // 4)
    lo = np.minimum(hm.COMBOS[:, 0] // 4, hm.COMBOS[:, 1] // 4)
    suited = hm.COMBOS[:, 0] % 4 == hm.COMBOS[:, 1] % 4
    grid = np.zeros((13, 13))
    np.add.at(grid, (np.where(suited, 12 - hi, 12 - lo), np.where(suited, 12 - lo, 12 - hi)), p)
    return grid


def illustrate_hand(model, bot, seed, out):
    env = texas_holdem_v4.env(render_mode="rgb_array")
    hero = hm.EquityAgent(model.reader("population", learn=True), seed=seed)
    villain = hm.Replica(model, bot, seed=seed)
    frames = []
    names = {0: "Player 1 (model)", 1: f"Player 2 ({bot.split(':')[1][:18]})"}
    labels = {CALL: "calls", CHECK: "checks", RAISE: "bets/raises", FOLD: "folds"}

    def record(env, agents, actor, state, action, hands):
        table = env.render()
        if actor is None:
            text = "hand over: opponent had " + "".join(hm.card_str(c) for c in hands[1])
            frames.append((table, frames[-1][1] if frames else None, text))
            return
        text = f"{names[actor]} {labels[action]}"
        belief = None
        if actor == 0 and hero.last_q is not None:
            belief = (class_grid(hero.last_pmf.numpy()), hero.last_q, state)
            text += f"   (q = {hero.last_q:.2f})"
        frames.append((table, belief if belief else (frames[-1][1] if frames else None), text))

    gain = play_hand(env, [hero, villain], seed=seed, recorder=record)
    fig, (ax_t, ax_b) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    def draw(k):
        table, belief, text = frames[k]
        ax_t.clear()
        ax_t.imshow(table)
        ax_t.set_axis_off()
        ax_t.set_title(text, fontsize=9)
        ax_b.clear()
        if belief is not None:
            grid, q, state = belief
            ax_b.imshow(grid, cmap="magma")
            ax_b.set_title(
                f"model: opponent's pair PMF (street {state['street']}), q = {q:.2f}", fontsize=9
            )
        ax_b.set_xticks(range(13), hm.RANKS[::-1], fontsize=7)
        ax_b.set_yticks(range(13), hm.RANKS[::-1], fontsize=7)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1400)
    anim.save(out / "hand.gif", writer=animation.PillowWriter(fps=0.8))
    plt.close(fig)
    decisions = [f for f in frames if f[1] is not None]
    keep = [decisions[0]] + [b for a, b in zip(decisions, decisions[1:]) if b[1] is not a[1]]
    fig, axes = plt.subplots(2, len(keep), figsize=(3.6 * len(keep), 6.4), squeeze=False)
    for j, (table, belief, text) in enumerate(keep):
        axes[0, j].imshow(table)
        axes[0, j].set_axis_off()
        axes[0, j].set_title(text, fontsize=8)
        grid, q, state = belief
        axes[1, j].imshow(grid, cmap="magma")
        axes[1, j].set_title(f"street {state['street']}: q = {q:.2f}", fontsize=8)
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
    fig.suptitle(
        f"one hand against a replica of {bot.split(':')[1]}: model wins {gain:+.1f} big blinds"
    )
    fig.tight_layout()
    fig.savefig(out / "hand.png", dpi=110)
    plt.close(fig)
    return gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=150)
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "illustrations")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = hm.load()
    bots = [n for n in model.names if n.startswith("bot:")]
    aggressive = max(bots, key=lambda n: raise_share(model, n))
    print(
        f"loaded {hm.DEFAULT_PATH.name}: {', '.join(n for n in model.names if n not in bots)} "
        f"and {len(bots)} bot models"
    )

    agents = {
        "no model": lambda: hm.EquityAgent(model.reader(None)),
        "population model": lambda: hm.EquityAgent(model.reader("population")),
        "learning online": lambda: hm.EquityAgent(model.reader("population", learn=True)),
    }
    opponents = {
        "random": lambda: Random(),
        "calling station": lambda: Passive(),
        "maniac": lambda: Maniac(),
        f"replica of {aggressive[4:]}": lambda: hm.Replica(model, aggressive),
    }
    results = {}
    for o_name, make_o in opponents.items():
        for a_name, make_a in agents.items():
            results[a_name, o_name] = play_match(make_a, make_o, args.deals, args.seed)
            print(
                f"{a_name:>17} vs {o_name:<34} {results[a_name, o_name].mean():+7.3f} "
                "big blinds per hand"
            )

    lines = [
        f"{2 * args.deals} hands per match (duplicate deals); big blinds per hand, "
        "difference to the agent with no model in brackets (± 2 sd of the mean over deals)",
        "",
    ]
    lines.append(f"{'':>17} | " + " | ".join(f"{o[:26]:>26}" for o in opponents))
    for a_name in agents:
        cells = []
        for o_name in opponents:
            g = results[a_name, o_name].mean(1)
            diff = g - results["no model", o_name].mean(1)
            se = 2 * diff.std(ddof=1) / np.sqrt(len(diff)) if a_name != "no model" else 0.0
            cells.append(f"{g.mean():+6.2f} ({diff.mean():+5.2f} ± {se:4.2f})")
        lines.append(f"{a_name:>17} | " + " | ".join(f"{c:>26}" for c in cells))
    summary = "\n".join(lines)
    print("\n" + summary)
    (args.out / "summary.txt").write_text(summary + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
    for ax, o_name in zip(axes.ravel(), opponents):
        for a_name in agents:
            ax.plot(np.cumsum(results[a_name, o_name].reshape(-1)), label=a_name)
        ax.set_title(f"against {o_name}", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative big blinds")
    for ax in axes[1]:
        ax.set_xlabel("hand")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("The exported opponent model in PettingZoo texas_holdem_v4")
    fig.tight_layout()
    fig.savefig(args.out / "winnings.png", dpi=110)
    plt.close(fig)

    gain = illustrate_hand(model, aggressive, seed=args.seed + 3, out=args.out)
    print(
        f"\nillustrated one hand (model {gain:+.1f} big blinds); "
        f"wrote {', '.join(sorted(p.name for p in args.out.iterdir()))}"
    )


if __name__ == "__main__":
    main()
