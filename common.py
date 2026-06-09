import numpy as np
import gymnasium as gym


def make_env(is_slippery=True):
    """Construct the canonical 4x4 FrozenLake env."""
    return gym.make("FrozenLake-v1", is_slippery=is_slippery)


def extract_mdp(env):
    """
    Pull the model out of env.unwrapped.P into plain Python/NumPy structures.

    Returns:
        nS    : number of states (16 for 4x4)
        nA    : number of actions (4)
        P     : nested list P[s][a] = list of (prob, next_state, reward, done) tuples
                (this is just env.unwrapped.P, returned for convenience)
        ncol  : grid width (for rendering)
    """
    P = env.unwrapped.P
    nS = env.observation_space.n
    nA = env.action_space.n
    ncol = env.unwrapped.ncol
    return nS, nA, P, ncol


def build_arrays(P, nS, nA):
    T = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for prob, s_next, reward, _done in P[s][a]:
                T[s, a, s_next] += prob
                R[s, a] += prob * reward
    return T, R


# Action indices for FrozenLake: 0=Left, 1=Down, 2=Right, 3=Up
ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}


def render_policy(policy, ncol, holes=None, goal=None):
    """
    Render a greedy policy as a grid of arrows.
    """
    holes = holes or set()
    nS = len(policy)
    nrow = nS // ncol
    lines = []
    for r in range(nrow):
        row = []
        for c in range(ncol):
            s = r * ncol + c
            if s == goal:
                row.append("G")
            elif s in holes:
                row.append("H")
            else:
                row.append(ARROWS[int(policy[s])])
        lines.append(" ".join(row))
    out = "\n".join(lines)
    print(out)
    return out


def render_values(V, ncol):
    """Pretty-print the value function as a grid."""
    nS = len(V)
    nrow = nS // ncol
    lines = []
    for r in range(nrow):
        cells = [f"{V[r * ncol + c]:6.3f}" for c in range(ncol)]
        lines.append(" ".join(cells))
    out = "\n".join(lines)
    print(out)
    return out


def frozenlake_4x4_layout():
    """
    Hole and goal locations for the default 4x4 map ("SFFF/FHFH/FFFH/HFFG").
    """
    desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
    holes, goal = set(), None
    for r, line in enumerate(desc):
        for c, ch in enumerate(line):
            s = r * len(line) + c
            if ch == "H":
                holes.add(s)
            elif ch == "G":
                goal = s
    return holes, goal
