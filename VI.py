"""
Part (a): Value iteration.
"""

import numpy as np
from common import (
    make_env,
    extract_mdp,
    render_policy,
    render_values,
    frozenlake_4x4_layout,
)


def termination_check(V_prev, V_new, theta, gamma):
    threshold = theta * (1 - gamma) / gamma
    return np.max(np.abs(V_new - V_prev)) < threshold


def calculate_Q(P, V_prev, gamma, nS, nA):
    Q = np.zeros((nS, nA))
    for state in range(nS):
        for action in range(nA):
            total = 0.0
            for prob, s_next, reward, done in P[state][action]:
                total += prob * (reward + gamma * V_prev[s_next])
            Q[state][action] = total
    return Q


def update_values(V_prev, Q):
    V = V_prev.copy()
    for state in range(len(V_prev)):
        V[state] = max(Q[state])

    return V


def extract_policy(Q):
    policy = np.zeros(len(Q), dtype=int)

    for state in range(len(Q)):
        policy[state] = np.argmax(Q[state])

    return policy


def value_iteration(P, gamma, theta, nS, nA):
    Q = np.zeros((nS, nA))
    V_prev = np.zeros(nS)
    V_new = np.zeros(nS)
    iters = 0

    while True:
        iters += 1
        Q = calculate_Q(P, V_prev, gamma, nS, nA)
        V_new = update_values(V_prev, Q)

        if termination_check(V_prev, V_new, theta, gamma):
            break

        V_prev = V_new

    policy = extract_policy(Q)

    return V_new, policy, iters


def value_iteration_with_history(P, gamma, theta, nS, nA):
    V = np.zeros(nS)
    V_history = []  # V at each iteration
    policy_history = []  # greedy policy at each iteration
    iters = 0

    while True:
        iters += 1
        Q = calculate_Q(P, V, gamma, nS, nA)
        V_new = update_values(V, Q)
        policy_history.append(extract_policy(Q))
        V_history.append(V_new.copy())

        if termination_check(V, V_new, theta, gamma):
            break
        V = V_new

    return V_new, policy_history, V_history, iters


def find_policy_emergence(P, gamma, theta, nS, nA):
    V_star, policy_history, V_history, iters = value_iteration_with_history(
        P, gamma, theta, nS, nA
    )
    pi_star = policy_history[-1]

    k_star = None
    for k, pi_k in enumerate(policy_history, start=1):
        if np.array_equal(pi_k, pi_star):
            k_star = k
            break

    # value gap at the emergence iteration (k_star 1-indexed)
    gap = np.max(np.abs(V_history[k_star - 1] - V_star))

    print(f"k* (first iter where pi_k == pi*): {k_star}")
    print(f"Total VI iterations to value convergence: {iters}")
    print(f"||V_k* - V*||_inf at emergence: {gap:.6e}")
    return k_star, gap


def main():
    env = make_env(is_slippery=True)
    nS, nA, P, ncol = extract_mdp(env)
    holes, goal = frozenlake_4x4_layout()

    gamma, theta = 0.99, 1e-4

    # Used originally for a
    V, policy, iters = value_iteration(P, gamma, theta, nS, nA)

    print(f"Converged in {iters} iterations.\n")
    print("V* table:")
    render_values(V, ncol)
    print("\nGreedy policy pi*:")
    render_policy(policy, ncol, holes=holes, goal=goal)
    # find_policy_emergence(P, gamma, theta, nS, nA)


if __name__ == "__main__":
    main()
