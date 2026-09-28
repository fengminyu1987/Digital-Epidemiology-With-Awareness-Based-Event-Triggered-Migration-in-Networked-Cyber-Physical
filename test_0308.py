import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. 辅助函数
# ---------------------------
def compute_epsilon(states, alpha, epsilon0):
    """
    计算激活因子 epsilon_i:
      如果居住点 i 的局部有意识比例 p_A = (AS + AI) 超过阈值 alpha,
      则 epsilon_i = epsilon0;否则为 0。
    参数:
      states: (N,5) 数组，每行分别为 [US, AS, AI, UR, AR] 的比例
      alpha: 意识阈值
      epsilon0: 激活常数
    返回:
      epsilon: (N,) 数组
    """
    p_A = states[:, 1] + states[:, 2]  # 取 AS + AI 作为有意识水平
    epsilon = np.where(p_A > alpha, epsilon0, 0.0)
    return epsilon

def compute_effective_awareness(n, states, g_i, R):
    """
    计算每个中转站 j 的过客有效意识比例 tilde_p_j^A(t):
    tilde_p_j^A = (sum_i (n_i * g_i * R[i,j] * p_A^(i))) / (sum_i (n_i * g_i * R[i,j]))
    其中 p_A^(i) = states[i,1] + states[i,2]
    参数:
      n: (N,) 各居住点人口数
      states: (N,5) 每个居住点状态比例
      g_i: (N,) 居住点迁移率
      R: (N, M) 居住点到中转站权重矩阵
    返回:
      tilde_p_A: (M,) 数组
    """
    N = len(n)
    M = R.shape[1]
    n_out = n * g_i  
    p_A_res = states[:, 1] + states[:, 2]
    tilde_p_A = np.zeros(M)
    for j in range(M):
        numerator = 0.0
        denominator = 0.0
        for i in range(N):
            flow = n_out[i] * R[i, j]
            numerator += flow * p_A_res[i]
            denominator += flow
        tilde_p_A[j] = numerator / denominator if denominator > 0 else 0.0
    return tilde_p_A


def compute_awareness_conversion(A, beta1, tilde_p_A):
    """
    计算每个居住点 i 的意识转换概率 r_i:
      r_i = 1 - ∏_{j=1}^{M} [1 - beta1 * A[j, i] * tilde_p_A[j]]
    参数:
      A: (M, N) 意识层的二值边权矩阵，
         如果中转站 j 与居住点 i 相连，则 A[j, i] = 1,否则为 0。
      beta1: 信息接收率
      tilde_p_A: (M,) 中转站的过客有效意识比例
    返回:
      r: (N,) 数组
    """
    M, N = A.shape
    r = np.zeros(N)
    for i in range(N):
        prod = 1.0
        for j in range(M):
            prod *= (1 - beta1 * A[j, i] * tilde_p_A[j])
        r[i] = 1 - prod
    return r


def compute_infection_residence(n, states, g_i, lambda_U, lambda_A):
    """
    计算居住点内部的感染概率：
      n_remain = n_i * (1 - g_i) 为留在居住点的人数
      p_I = AI 状态比例
      N_U = 1 - [1 - lambda_U * p_I]^(n_remain)
      N_A = 1 - [1 - lambda_A * p_I]^(n_remain)
    参数:
      n: (N,) 人口数
      states: (N,5) 状态比例数组；使用 states[:,2] 作为 AI 比例
      g_i: (N,) 迁移率
      lambda_U, lambda_A: 感染敏感度参数
    返回:
      N_U, N_A: (N,) 数组
    """
    n_remain = n * (1 - g_i)
    p_I = states[:, 2]
    # 采用 np.power 计算 (1 - lambda * p_I)^(n_remain)
    N_U = 1 - np.power(1 - lambda_U * p_I, n_remain)
    N_A = 1 - np.power(1 - lambda_A * p_I, n_remain)
    return N_U, N_A

def compute_infection_transit(n, states, g_i, R, lambda_U, lambda_A):
    """
    计算中转站层面的感染概率：
      对于中转站 j,计算每个居住点 k 流向 j的人数 n_{k->j} = n_k * g_k * R[k, j]
      然后 M_j^U = 1 - sum_k [1 - lambda_U * p_k^I]^(n_{k->j})
           M_j^A = 1 - sum_k [1 - lambda_A * p_k^I]^(n_{k->j})
    参数:
      n, states, g_i, R, lambda_U, lambda_A(含义同上)
    返回:
      M_U, M_A: (M,) 数组
    """
    N, M = R.shape
    M_U = np.zeros(M)
    M_A = np.zeros(M)
    for j in range(M):
        sum_U = 0.0
        sum_A = 0.0
        for k in range(N):
            n_to_j = n[k] * g_i[k] * R[k, j]
            p_I = states[k, 2]
            sum_U += np.power(1 - lambda_U * p_I, n_to_j)
            sum_A += np.power(1 - lambda_A * p_I, n_to_j)
        M_U[j] = 1 - sum_U
        M_A[j] = 1 - sum_A
    return M_U, M_A

def compute_overall_infection(n, states, g_i, R, N_U, N_A, M_U, M_A):
    """
    计算每个居住点 i 的总体感染概率：
      Q_i^U = (1 - g_i)*N_U[i] + g_i * sum_{j} R[i,j]*M_U[j]
      Q_i^A = (1 - g_i)*N_A[i] + g_i * sum_{j} R[i,j]*M_A[j]
    参数:
      n, states, g_i, R, N_U, N_A, M_U, M_A(含义如前)
    返回:
      Q_U, Q_A: (N,) 数组
    """
    N, M = R.shape
    Q_U = np.zeros(N)
    Q_A = np.zeros(N)
    for i in range(N):
        sum_MU = np.sum(R[i, :] * M_U)
        sum_MA = np.sum(R[i, :] * M_A)
        Q_U[i] = (1 - g_i[i]) * N_U[i] + g_i[i] * sum_MU
        Q_A[i] = (1 - g_i[i]) * N_A[i] + g_i[i] * sum_MA
    return Q_U, Q_A

def update_states(states, r, Q_U, Q_A, mu1, mu2):
    """
    根据状态更新公式更新居住点内各状态比例：
      p_US(t+1) = p_US*(1-r)*(1-Q_U) + p_AS*mu1*(1-Q_U)
      p_AS(t+1) = p_AS*(1-mu1)*(1-Q_A) + p_US*r*(1-Q_A)
      p_AI(t+1) = p_AI*(1-mu2) + p_AS*(1-mu1)*Q_A + p_AS*mu1*Q_U +
                  p_US*(1-r)*Q_U + p_US*r*Q_A
      p_UR(t+1) = p_AI*mu1*mu2 + p_AR*mu1 + p_UR*(1-r)
      p_AR(t+1) = p_AR*(1-mu1) + p_AI*(1-mu1)*mu2 + p_UR*r
    参数:
      states: (N,5) 当前状态比例
      r: (N,) 意识转换概率
      Q_U, Q_A: (N,) 感染概率
      mu1, mu2: 遗忘率、康复率
    返回:
      new_states: (N,5) 更新后的状态比例（每行归一化）
    """
    new_states = np.zeros_like(states)
    US = states[:, 0]
    AS = states[:, 1]
    AI = states[:, 2]
    UR = states[:, 3]
    AR = states[:, 4]
    
    new_states[:, 0] = US * (1 - r) * (1 - Q_U) + AS * mu1 * (1 - Q_U)
    new_states[:, 1] = AS * (1 - mu1) * (1 - Q_A) + US * r * (1 - Q_A)
    new_states[:, 2] = AI * (1 - mu2) + AS * (1 - mu1) * Q_A + AS * mu1 * Q_U + US * (1 - r) * Q_U + US * r * Q_A
    new_states[:, 3] = AI * mu1 * mu2 + AR * mu1 + UR * (1 - r)
    new_states[:, 4] = AR * (1 - mu1) + AI * (1 - mu1) * mu2 + UR * r

    # 归一化每个居住点的状态比例
    new_states = new_states / new_states.sum(axis=1, keepdims=True)
    return new_states

def update_population(n, states, g_i, epsilon, R, gamma_i):
    """
    按照 MIR 流动更新居住点人口及状态：
      (1) 每个居住点 i 的流出人口 F_i = n_i * g_i * epsilon_i;
      (2) 流出时各状态按原比例分离；
      (3) 各流出人口按 R 分布到中转站 j;
      (4) 中转站中人口在一个时间步内返回到各居住点，
          返回人口 = m_j * R[i,j] * gamma_i;
      (5) 居住点新人口 = 原人口 - 流出 + 返回。
    参数:
      n: (N,) 居住点人口数
      states: (N,5) 各居住点状态比例
      g_i: (N,) 居住点迁移率
      epsilon: (N,) 激活因子
      R: (N, M) 居住点到中转站权重矩阵
      gamma_i: (N,) 居住点返回率
    返回:
      n_new: (N,) 更新后人口数
      new_states: (N,5) 更新后状态比例
    """
    N, M = R.shape
    # 计算流出人口
    F = n * g_i * epsilon

    # 计算流出后各居住点剩余人口与状态数量
    X = states * n[:, None]  # 各居住点状态人数
    X_remain = X - (X.T * (g_i * epsilon)).T     # g_i * epsilon形状为 (N,),n[:, None]形状为 (N,1)
    n_remain = n - F

    # 计算各中转站 j 的流入人口及各状态
    m = np.zeros(M)
    m_X = np.zeros((M, 5))
    for j in range(M):
        for i in range(N):
            flow = n[i] * g_i[i] * epsilon[i] * R[i, j]
            m[j] += flow
            m_X[j] += X[i] * g_i[i] * epsilon[i] * R[i, j]
    
    # 返回：各中转站 j 人口返回到居住点 i，按权重 R 和返回率 gamma_i
    flow_return = np.zeros(N)
    flow_return_X = np.zeros((N, 5))
    for i in range(N):
        for j in range(M):
            flow_return[i] += m[j] * R[i, j] * gamma_i[i]
            flow_return_X[i] += m_X[j] * R[i, j] * gamma_i[i]
    
    n_new = n_remain + flow_return
    X_new = X_remain + flow_return_X
    new_states = X_new / n_new[:, None]
    return n_new, new_states

# ---------------------------
# 2. 主仿真流程
# ---------------------------
def main_simulation(T, N, M, params):
    """
    主仿真流程
    参数:
      T: 总时间步数
      N: 居住点数量
      M: 中转站数量
      params: 参数字典，包含 beta1, mu1, mu2, lambda_U, lambda_A, g, gamma, epsilon0, alpha, theta, kappa
    返回:
      state_history: (T, N, 5) 各时间步各居住点状态比例历史记录
      pop_history: (T, N) 各时间步各居住点人口数
    """
    beta1 = params['beta1']
    mu1 = params['mu1']
    mu2 = params['mu2']
    lambda_U = params['lambda_U']
    lambda_A = params['lambda_A']
    g_global = params['g']
    gamma_global = params['gamma']
    epsilon0 = params['epsilon0']
    alpha = params['alpha']
    theta = params['theta']
    kappa = params['kappa']

    # 居住点特异性迁移率和返回率（这里使用常数，可扩展为数组）
    g_i = np.full(N, theta * g_global)
    gamma_i = np.full(N, kappa * gamma_global)
    
    # 初始化各居住点人口（例如每个 1000 人）和状态比例（US, AS, AI, UR, AR）
    n = np.full(N, 500.0)
    states = np.zeros((N, 5))
    states[:, 0] = 0.95  # US
    states[:, 2] = 0.05  # AI
    states = states / states.sum(axis=1, keepdims=True)
    
    # 初始化居住点与中转站之间的权重矩阵 R (N x M)，对称归一化：每个居住点行和为 1
    R = np.random.rand(N, M)
    R = R / R.sum(axis=1, keepdims=True)
    
    # 用于记录历史数据
    state_history = np.zeros((T, N, 5))
    pop_history = np.zeros((T, N))
    
    for t in range(T):
        state_history[t] = states
        pop_history[t] = n
        
        # 1. 计算当前居住点局部有意识比例 p^A = AS + AI
        # 2. 计算激活因子 epsilon: 若 p^A > alpha 则 epsilon = epsilon0，否则 0
        epsilon = compute_epsilon(states, alpha, epsilon0)
        
        # 3. 计算中转站的过客有效意识比例 tilde_p^A
        tilde_p_A = compute_effective_awareness(n, states, g_i, epsilon, R)
        
        # 4. 计算居住点的意识转换概率 r
        r = compute_awareness_conversion(R, beta1, tilde_p_A)
        
        # 5. 计算居住点内部感染风险
        N_U, N_A = compute_infection_residence(n, states, g_i, lambda_U, lambda_A)
        # 6. 计算中转站层面的感染风险
        M_U, M_A = compute_infection_transit(n, states, g_i, R, lambda_U, lambda_A)
        # 7. 综合计算每个居住点的总体感染概率 Q
        Q_U, Q_A = compute_overall_infection(n, states, g_i, R, N_U, N_A, M_U, M_A)
        
        # 8. 更新居住点内部的疾病/意识状态
        states = update_states(states, r, Q_U, Q_A, mu1, mu2)
        
        # 9. 更新居住点的人口及状态（基于 MIR 过程）
        n, states = update_population(n, states, g_i, epsilon, R, gamma_i)
        
    return state_history, pop_history

# ---------------------------
# 3. 主函数入口
# ---------------------------
def main():
    # 参数设置
    T = 200               # 总时间步数
    N = 10                # 居住点数量
    M = 5                 # 中转站数量
    params = {
        'beta1': 0.2,         # 信息接收率
        'mu1': 0.15,          # 遗忘率
        'mu2': 0.1,           # 康复率
        'lambda_U': 0.03,     # 无意识易感感染敏感度
        'lambda_A': 0.015,    # 有意识易感感染敏感度
        'g': 0.5,             # 全局迁移率
        'gamma': 0.8,         # 全局返回率
        'epsilon0': 0.3,      # 激活参数
        'alpha': 0.2,         # 局部意识阈值
        'theta': 1,           # 居住点特异性因子（用于计算 g_i）
        'kappa': 1            # 居住点特异性因子（用于计算 gamma_i）
    }
    
    # 执行仿真
    state_history, pop_history = main_simulation(T, N, M, params)
    
    # 计算全局平均状态比例（按各居住点人口加权平均）
    global_states = np.zeros((T, 5))
    for t in range(T):
        total_state = np.sum(state_history[t] * pop_history[t][:, None], axis=0)
        global_states[t] = total_state / np.sum(pop_history[t])
    
    # 绘制全局状态比例随时间变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(global_states[:, 0], label='US')
    plt.plot(global_states[:, 1], label='AS')
    plt.plot(global_states[:, 2], label='AI')
    plt.plot(global_states[:, 3], label='UR')
    plt.plot(global_states[:, 4], label='AR')
    plt.xlabel('Time Steps')
    plt.ylabel('Global Average Proportion')
    plt.title('Dynamics of States in UAU-SIR Dual-layer Model')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
