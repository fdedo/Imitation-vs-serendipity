import numpy as np
from scipy import stats
import time
import os

# ------------------------ Configurable Parameters ------------------------ #
T = 500                      # Time steps
N = 25                       # Number of agents
M = 10                       # Number of actions
q_vec = np.logspace(-3, 0, 30)
Niter = 500                  # Number of independent simulations
l = 0.7                      # Some model-specific constant

# Create output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# ------------------------ Output Containers ------------------------ #
total_wealth = np.zeros((Niter, len(q_vec)))
gini = np.zeros((Niter, len(q_vec)))
corr_phi_max = np.zeros((Niter, len(q_vec)))
corr_phi_avg = np.zeros((Niter, len(q_vec)))
corr_phi_pi = np.zeros((Niter, len(q_vec)))
diversity = np.zeros((Niter, len(q_vec)))
utility = np.zeros((Niter, len(q_vec)))
corr_payoff_alpha = np.zeros((Niter, len(q_vec)))
total_benefit = np.zeros((Niter, len(q_vec)))
total_benefit2 = np.zeros((Niter, len(q_vec)))
Ipr = np.zeros((Niter, len(q_vec)))
Ipr_num = np.zeros((Niter, len(q_vec)))
I = np.zeros((Niter, len(q_vec)))
p_win = []

# ------------------------ Simulation Loop ------------------------ #
for qq, q in enumerate(q_vec):
    for ni in range(Niter):
        p = np.random.rand(M)
        p_aux = np.cumsum(p / np.sum(p))
        alpha = np.random.rand(N, M)
        actions = np.zeros((N, 1))

        # Initial action assignment
        for i in range(N):
            r = np.random.rand()
            actions[i] = np.argmax(p_aux > r)

        binc = np.arange(M + 1)
        counts, _ = np.histogram(actions, bins=binc)
        actions_freq = counts.reshape((M, 1)) / N

        # Compute fitness
        phi_avg = np.mean(alpha, axis=1, keepdims=True)
        phi_max = np.max(alpha, axis=1, keepdims=True)
        phi_pi = np.dot(alpha, p.reshape(M, 1))

        # Initial wealth and benefit
        wealth = np.array([
            [alpha[i, int(actions[i])] * actions_freq[int(actions[i])]]
            for i in range(N)
        ])
        benefit = np.array([
            [alpha[i, int(actions[i])] * p[int(actions[i])]]
            for i in range(N)
        ])
        benefit2 = benefit.copy()

        # Simulation loop
        ind = np.argwhere(actions_freq > 0)
        c = np.cumsum(actions_freq[ind[:, 0]])

        for t in range(T - 1):
            actions_new = actions[:, -1].copy()
            wealth_new = wealth[:, -1].copy()
            benefit_new = benefit[:, -1].copy()
            benefit_new2 = benefit2[:, -1].copy()

            aux_freq = actions_freq[:, -1] if t > 0 else actions_freq[:, 0]

            for i in range(N):
                stay = np.random.rand() < alpha[i, int(actions[i, -1])] * aux_freq[int(actions[i, -1])]

                if stay:
                    wealth_new[i] += alpha[i, int(actions[i, -1])] * aux_freq[int(actions[i, -1])]
                    benefit_new[i] += alpha[i, int(actions[i, -1])] * p[int(actions[i, -1])]
                    benefit_new2[i] = alpha[i, int(actions[i, -1])] * p[int(actions[i, -1])]
                else:
                    if np.random.rand() > q:
                        aux = np.random.rand()
                        f = np.argwhere(c > aux)[0]
                        new_action = ind[f, 0]
                    else:
                        new_action = np.random.randint(0, M)

                    actions_new[i] = new_action
                    wealth_new[i] += alpha[i, new_action] * aux_freq[new_action]
                    benefit_new[i] += alpha[i, int(actions[i, -1])] * p[int(actions[i, -1])]
                    benefit_new2[i] = alpha[i, int(actions[i, -1])] * p[int(actions[i, -1])]

            # Update records
            actions = np.hstack((actions, actions_new[:, np.newaxis]))
            wealth = np.hstack((wealth, wealth_new[:, np.newaxis]))
            benefit = np.hstack((benefit, benefit_new[:, np.newaxis]))
            counts, _ = np.histogram(actions[:, -1], bins=binc)
            actions_freq2 = counts.reshape(M, 1) / N
            actions_freq = np.hstack((actions_freq, actions_freq2))
            ind = np.argwhere(actions_freq[:, -1] > 0)
            c = np.cumsum(actions_freq[ind[:, 0], -1])

        # Metrics
        total_benefit2[ni, qq] = np.sum(benefit2)
        total_wealth[ni, qq] = np.sum(wealth[:, -1])
        total_benefit[ni, qq] = np.sum(benefit[:, -1])
        Ipr[ni, qq] = np.sum(actions_freq[-1, :])
        Ipr_num[ni, qq] = 1 / M * np.linalg.norm(actions_freq[-1, :])
        diversity[ni, qq] = len(np.unique(actions[:, -1])) / M

        # Gini coefficient
        G = np.sum([
            abs(wealth[i, -1] - wealth[j, -1])
            for i in range(N) for j in range(i + 1, N)
        ])
        gini[ni, qq] = G / (N * np.sum(wealth[:, -1]))

        # Correlations
        corr_phi_avg[ni, qq] = stats.kendalltau(phi_avg, wealth[:, -1])[0]
        corr_phi_max[ni, qq] = stats.kendalltau(phi_max, wealth[:, -1])[0]
        corr_phi_pi[ni, qq] = stats.kendalltau(phi_pi, wealth[:, -1])[0]

        alpha_aux = np.array([[alpha[i, int(actions[i, -1])]] for i in range(N)])
        utility[ni, qq] = np.sum(alpha_aux * p[int(actions[i, -1])])
        p_fin = np.array([[p[int(actions[i, -1])]] for i in range(N)])
        corr_payoff_alpha[ni, qq] = np.corrcoef(np.hstack((alpha_aux, p_fin)).T)[0, 1]

        # Winner stats
        f = np.argwhere(actions_freq[:, -1] == np.max(actions_freq[:, -1]))[0]
        p_win.append(p[f])
        counts_i, _ = np.histogram(actions[:, -1], bins=binc)
        counts_norm = counts_i / N
        den = sum((counts_norm / np.linalg.norm(counts_norm))**4)
        I[ni, qq] = 1 / (M * den)

        print(f'q = {q:.4f}, iteration = {ni}')

# ------------------------ Save Results ------------------------ #
timestamp = time.strftime("%Y%m%d-%H%M%S")

def save_array(data, name):
    np.savetxt(os.path.join(output_dir, f'{name}_{timestamp}.txt'), data)

save_array(np.mean(total_wealth, axis=0), "total_wealth_mean")
save_array(np.std(total_wealth, axis=0), "total_wealth_std")
save_array(np.mean(gini, axis=0), "gini_mean")
save_array(np.std(gini, axis=0), "gini_std")
save_array(np.mean(diversity, axis=0), "diversity_mean")
save_array(np.mean(total_benefit, axis=0), "benefit_mean")
save_array(np.std(total_benefit, axis=0), "benefit_std")
save_array(np.mean(total_benefit2, axis=0), "benefit2_mean")
save_array(np.std(total_benefit2, axis=0), "benefit2_std")
save_array(np.mean(I, axis=0), "ipr_mean")
save_array(np.std(I, axis=0), "ipr_std")
save_array(np.mean(utility, axis=0), "utility_mean")
save_array(np.mean(corr_phi_avg, axis=0), "corr_avg_mean")
save_array(np.mean(corr_phi_max, axis=0), "corr_max_mean")
save_array(np.mean(corr_phi_pi, axis=0), "corr_pi_mean")
save_array(np.std(corr_phi_avg, axis=0), "corr_avg_std")
save_array(np.std(corr_phi_max, axis=0), "corr_max_std")
save_array(np.std(corr_phi_pi, axis=0), "corr_pi_std")
