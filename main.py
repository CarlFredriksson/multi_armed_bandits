import numpy as np
import matplotlib.pyplot as plt
import math

def my_argmax(Q_values):
    """
    Returns the index of the maximum value in Q_values.
    In case of ties, one of the tying indices are selected at uniform-random and returned.
    """
    max_val = -math.inf
    ties = []
    for i in range(len(Q_values)):
        Q_val = Q_values[i]
        if Q_val > max_val:
            max_val = Q_val
            ties = [i]
        elif Q_val == max_val:
            ties.append(i)

    return np.random.choice(ties)

def run_epsilon_greedy(q, num_steps, epsilon, const_alpha=None, Q_0=0, stationary=True):
    R = np.zeros((num_steps,))
    optimal = np.zeros((num_steps,))
    Q = np.ones(np.shape(q)) * Q_0
    N = np.zeros(np.shape(q))

    # Run steps
    for t in range(0, num_steps):
        if not stationary:
            q = q + np.random.normal(scale=0.01, size=np.shape(q))

        # Choose action greedily by default and randomly by probability epsilon
        a = my_argmax(Q)
        if np.random.uniform(low=0.0, high=1.0) < epsilon:
            a = np.random.randint(low=0, high=len(Q) - 1)
        N[a] += 1
        if a == np.argmax(q):
            optimal[t] = 1

        # Get reward
        R[t] = np.random.normal(loc=q[a])

        # Update Q
        alpha = const_alpha
        if const_alpha is None:
            alpha = 1 / N[a]
        Q[a] = Q[a] + alpha * (R[t] - Q[a])

    return R, optimal

def run_experiments(parameters, num_runs, num_steps, k, stationary=True):
    bandits = np.random.normal(loc=0, scale=1, size=(num_runs, k))
    R = np.zeros((len(parameters), num_runs, num_steps))
    optimal = np.zeros(np.shape(R))

    for i in range(len(parameters)):
        for j in range(num_runs):
            if j % 100 == 0:
                print("Parameter set", i, "- Starting run", j)
            q = np.copy(bandits[j])
            epsilon = parameters[i]["epsilon"]
            const_alpha = parameters[i]["const_alpha"]
            Q_0 = parameters[i]["Q_0"]
            R[i, j], optimal[i, j] = run_epsilon_greedy(q, num_steps, epsilon, const_alpha, Q_0, stationary)

    return np.mean(R, axis=1), np.mean(optimal, axis=1)

def plot_subplot(y, y_label, title=None):
    num_plots = np.shape(y)[0]
    x = np.arange(0, np.shape(y)[1])

    for i in range(num_plots):
        epsilon = parameters[i]["epsilon"]
        const_alpha = parameters[i]["const_alpha"]
        Q_0 = parameters[i]["Q_0"]
        alpha = const_alpha if const_alpha is not None else "1/n"
        label = (
            "epsilon=" + str(epsilon) +
            ", alpha=" + str(alpha) +
            ", Q_0=" + str(Q_0)
        )
        plt.plot(x, y[i], label=label)

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)

def plot_results(file_name, R_avg, optimal_avg, parameters, title):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plot_subplot(R_avg, "Average reward", title)
    plt.subplot(2, 1, 2)
    plot_subplot(optimal_avg * 100, "Optimal action %")
    plt.yticks([0, 20, 40, 60, 80, 100], labels=["0%", "20%", "40%", "60%", "80%", "100%"])
    plt.tight_layout()
    plt.savefig(file_name)

if __name__ == "__main__":
    parameters = [
        { "epsilon": 0.1, "const_alpha": None, "Q_0": 0 },
        { "epsilon": 0.1, "const_alpha": 0.1, "Q_0": 0 },
        { "epsilon": 0.01, "const_alpha": 0.1, "Q_0": 0 },
        { "epsilon": 0, "const_alpha": 0.1, "Q_0": 0 },
        { "epsilon": 0, "const_alpha": 0.1, "Q_0": 5 },
    ]
    num_runs = 1000
    num_steps = 1000
    k = 10
    R_avg, optimal_avg = run_experiments(parameters, num_runs, num_steps, k)
    plot_results("results_stationary.png", R_avg, optimal_avg, parameters, "Stationary")
    R_avg, optimal_avg = run_experiments(parameters, num_runs, num_steps, k)
    plot_results("results_non-stationary.png", R_avg, optimal_avg, parameters, "Non-stationary")
