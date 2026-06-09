"""
Part (b): Policy iteration.
"""

import numpy as np
from common import (
    make_env,
    extract_mdp,
    build_arrays,
    render_policy,
    render_values,
    frozenlake_4x4_layout,
)


def policy_evaluation(policy, T, R, gamma, nS):
    """
    Exact policy evaluation via the linear system.
    For a fixed policy pi, the value function satisfies:
        V_pi = R_pi + gamma * P_pi @ V_pi
    =>  (I - gamma * P_pi) V_pi = R_pi
    """
    P_pi = np.zeros((nS, nS))
    R_pi = np.zeros(nS)
    for s in range(nS):
        a = policy[s]
        P_pi[s] = T[s, a]  # row of next-state probs under the policy
        R_pi[s] = R[s, a]  # expected reward under the policy

    # Form A = I - gamma * P_pi.
    A = np.eye(nS) - gamma * P_pi

    # Solve A V = R_pi
    return np.linalg.solve(A, R_pi)


def policy_improvement(V, T, R, gamma, nS, nA):
    """
    Greedy one-step improvement.
    """
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            Q[s, a] = R[s, a] + gamma * (T[s, a] @ V)
    return np.argmax(Q, axis=1).astype(int)


def termination_check(policy, new_policy, V, new_V, theta):
    # At points the policy's weren't terminating because of ties, so I added a
    # forced policy stabilization in the event of ties
    policy_stable = np.array_equal(new_policy, policy)
    value_stable = np.max(np.abs(new_V - V)) < theta
    return policy_stable or value_stable


def policy_iteration(P, gamma, nS, nA, theta=1e-10):
    T, R = build_arrays(P, nS, nA)

    policy = np.zeros(nS, dtype=int)
    V = policy_evaluation(policy, T, R, gamma, nS)
    iters = 0

    while True:
        iters += 1
        new_policy = policy_improvement(V, T, R, gamma, nS, nA)
        new_V = policy_evaluation(new_policy, T, R, gamma, nS)

        if termination_check(policy, new_policy, V, new_V, theta):
            policy, V = new_policy, new_V
            break

        policy, V = new_policy, new_V

    return V, policy, iters


def main():

    env = make_env(is_slippery=True)
    nS, nA, P, ncol = extract_mdp(env)
    holes, goal = frozenlake_4x4_layout()
    gamma = 0.99
    V, policy, iters = policy_iteration(P, gamma, nS, nA)

    print(f"Converged in {iters} iterations.\n")
    print("V* table:")
    render_values(V, ncol)
    print("\nOptimal policy pi*:")
    render_policy(policy, ncol, holes=holes, goal=goal)


if __name__ == "__main__":
    main()
