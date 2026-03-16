"""
全面改进版过夜实验脚本 v2。

核心改进:
1. MCTS 树搜索推理 (AlphaGo 式策略+价值引导)
2. 电路死门消除 + 验证 (后处理)
3. 1-step Lookahead 贪心专家 (更强训练数据)
4. 结构化邻接 (parent-child 图结构)
5. Beam Search + Best-of-N 采样
6. 自模仿学习 (PPO 期间回放最佳轨迹)
7. Cosine LR 调度
8. 向量化加速

预计总时长: GPU ~8-10h, CPU ~14-18h

用法:
    python run_overnight.py
    python run_overnight.py --gpu
    python run_overnight.py --phase 2
    python run_overnight.py --quick        # ~1-2h 快速验证
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import json
import os
import sys
import math


# ==========================================================
# 1. 向量化快速贪心专家
# ==========================================================
class FastGreedyExpert:
    """numpy 全向量化的贪心专家"""

    def __init__(self, temperature=0.0):
        self.temperature = temperature

    def generate_trajectory(self, env):
        obs = env.reset()
        trajectory = []
        while not np.all(env.achieved) and env.next_idx < env.max_nodes:
            u, v = self._find_best_pair_fast(env)
            if u is None:
                break
            v_mask = env.get_v_mask(u)
            trajectory.append({
                'features': obs['node_features'].copy(),
                'adj': obs['adj'].copy(),
                'valid_mask': obs['valid_mask'].copy(),
                'v_mask': v_mask.copy(),
                'u': u, 'v': v,
            })
            obs, reward, done, info = env.step(u, v)
            if done:
                break
        return trajectory, bool(np.all(env.achieved)), len(env.circuit)

    def _find_best_pair_fast(self, env):
        if getattr(self, '_time_limit', None) and time.time() > self._time_limit:
            return None, None
        valid_idx = np.where(env.valid)[0]
        n = len(valid_idx)
        if n < 2:
            return None, None

        ii, jj = np.triu_indices(n, k=1)
        u_arr = valid_idx[ii]
        v_arr = valid_idx[jj]

        ok = np.maximum(env.depth[u_arr], env.depth[v_arr]) + 1 <= env.max_depth
        u_arr, v_arr = u_arr[ok], v_arr[ok]
        if len(u_arr) == 0:
            return None, None

        xors = (env.nodes[u_arr].astype(np.int16) + env.nodes[v_arr].astype(np.int16)) % 2
        xors = xors.astype(np.int8)

        nonzero = np.any(xors, axis=1)
        u_arr, v_arr, xors = u_arr[nonzero], v_arr[nonzero], xors[nonzero]
        if len(u_arr) == 0:
            return None, None

        valid_set = set(map(lambda v: v.tobytes(), env.nodes[env.valid]))
        not_dup = np.array([x.tobytes() not in valid_set for x in xors])
        u_arr, v_arr, xors = u_arr[not_dup], v_arr[not_dup], xors[not_dup]
        if len(u_arr) == 0:
            return None, None

        scores = self._batch_score(xors, env)

        if self.temperature > 0 and len(scores) > 1:
            s = scores - scores.max()
            probs = np.exp(s / self.temperature)
            probs /= probs.sum()
            idx = np.random.choice(len(scores), p=probs)
        else:
            idx = int(np.argmax(scores))
        return int(u_arr[idx]), int(v_arr[idx])

    def _batch_score(self, xors, env):
        scores = np.zeros(len(xors), dtype=np.float64)
        for t in range(env.num_targets):
            if env.achieved[t]:
                continue
            dists = np.sum(xors != env.target[t], axis=1).astype(np.float64)
            scores[dists == 0] += 1000.0
            improve = env.min_dist[t] - dists
            scores += np.maximum(improve, 0) * 5.0
            scores += np.maximum(0, env.num_inputs / 2.0 - dists) * 0.5
        return scores


# ==========================================================
# 2. Lookahead 贪心专家 (dim<=16 时启用 1-step lookahead)
# ==========================================================
class LookaheadGreedyExpert(FastGreedyExpert):
    """1-step lookahead: 对 top-30 候选模拟下一步，选综合最优"""

    def __init__(self, temperature=0.0, lookahead_dim_limit=16):
        super().__init__(temperature)
        self.lookahead_dim_limit = lookahead_dim_limit

    def _find_best_pair_fast(self, env):
        if env.num_inputs > self.lookahead_dim_limit:
            return super()._find_best_pair_fast(env)

        valid_idx = np.where(env.valid)[0]
        n = len(valid_idx)
        if n < 2:
            return None, None

        ii, jj = np.triu_indices(n, k=1)
        u_arr = valid_idx[ii]
        v_arr = valid_idx[jj]

        ok = np.maximum(env.depth[u_arr], env.depth[v_arr]) + 1 <= env.max_depth
        u_arr, v_arr = u_arr[ok], v_arr[ok]
        if len(u_arr) == 0:
            return None, None

        xors = (env.nodes[u_arr].astype(np.int16) + env.nodes[v_arr].astype(np.int16)) % 2
        xors = xors.astype(np.int8)

        nonzero = np.any(xors, axis=1)
        u_arr, v_arr, xors = u_arr[nonzero], v_arr[nonzero], xors[nonzero]
        if len(u_arr) == 0:
            return None, None

        valid_set = set(map(lambda v: v.tobytes(), env.nodes[env.valid]))
        not_dup = np.array([x.tobytes() not in valid_set for x in xors])
        u_arr, v_arr, xors = u_arr[not_dup], v_arr[not_dup], xors[not_dup]
        if len(u_arr) == 0:
            return None, None

        # 立即得分
        imm_scores = self._batch_score(xors, env)

        # 对 top-30 做 lookahead
        top_k = min(30, len(imm_scores))
        top_idx = np.argsort(imm_scores)[-top_k:]
        scores = imm_scores.copy()

        for idx in top_idx:
            env_copy = env.copy()
            reward, done, info = env_copy.step_fast(int(u_arr[idx]), int(v_arr[idx]))
            if info:
                continue
            if np.all(env_copy.achieved):
                scores[idx] = imm_scores[idx] + 500.0
                continue
            # 下一步最佳得分
            next_u, next_v = super()._find_best_pair_fast(env_copy)
            if next_u is not None:
                next_xor = (env_copy.nodes[next_u].astype(np.int16) +
                            env_copy.nodes[next_v].astype(np.int16)) % 2
                la_score = self._score_single(next_xor.astype(np.int8), env_copy)
                scores[idx] = imm_scores[idx] + 0.5 * la_score

        if self.temperature > 0 and len(scores) > 1:
            s = scores - scores.max()
            probs = np.exp(s / self.temperature)
            probs /= probs.sum()
            idx = np.random.choice(len(scores), p=probs)
        else:
            idx = int(np.argmax(scores))
        return int(u_arr[idx]), int(v_arr[idx])

    def _score_single(self, xor_vec, env):
        score = 0.0
        for t in range(env.num_targets):
            if env.achieved[t]:
                continue
            d = int(np.sum(xor_vec != env.target[t]))
            if d == 0:
                score += 1000.0
            else:
                imp = env.min_dist[t] - d
                if imp > 0:
                    score += imp * 5.0
                score += max(0, env.num_inputs / 2.0 - d) * 0.5
        return score


# ==========================================================
# 3. 结构化邻接环境 (parent-child 图结构)
# ==========================================================
class StructuralSLPEnv:
    """包装 SLPGraphEnv, 替换全连接邻接为 parent-child 结构邻接"""

    def __init__(self, base_env):
        self.base = base_env
        # 代理所有属性
        for attr in ('target', 'num_targets', 'num_inputs', 'max_extra', 'max_depth',
                      'max_nodes', 'feature_dim', 'nodes', 'depth', 'valid',
                      'next_idx', 'achieved', 'circuit', 'parents', 'min_dist'):
            pass  # 通过 __getattr__ 代理

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_obs(self):
        obs = self.base.get_obs()
        # 替换邻接矩阵
        adj = np.zeros((self.base.max_nodes, self.base.max_nodes), dtype=np.float32)
        valid_idx = np.where(self.base.valid)[0]
        # 自环
        adj[valid_idx, valid_idx] = 1.0
        # 所有 valid 节点都能看到输入节点 (basis 快捷连接)
        for i in valid_idx:
            for j in range(self.base.num_inputs):
                if self.base.valid[j]:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        # Parent-child 边
        for child, (u, v) in self.base.parents.items():
            if child < self.base.max_nodes:
                adj[child, u] = 1.0
                adj[u, child] = 1.0
                adj[child, v] = 1.0
                adj[v, child] = 1.0
        obs['adj'] = adj
        return obs


# ==========================================================
# 4. 快速 baselines
# ==========================================================
def fast_paar(target_matrix):
    n_gates, solved, _ = paar_with_circuit(target_matrix)
    return n_gates, solved


def paar_with_circuit(target_matrix):
    """Paar 算法，返回 (n_gates, solved, circuit_list)"""
    T = np.array(target_matrix, dtype=np.int8).copy()
    n_targets, n_inputs = T.shape
    D = T.copy()
    circuit = []
    for _ in range(n_inputs * 5):
        row_sums = np.sum(D, axis=1)
        if np.all(row_sums <= 1):
            break
        n_cols = D.shape[1]
        best_count, best_i, best_j = -1, 0, 1
        for i in range(n_cols):
            if not np.any(D[:, i]):
                continue
            counts = np.sum(D[:, i:i+1] & D[:, i+1:], axis=0)
            if len(counts) > 0 and counts.max() > best_count:
                best_count = counts.max()
                best_j = i + 1 + int(np.argmax(counts))
                best_i = i
        if best_count <= 0:
            break
        new_col = (D[:, best_i] & D[:, best_j]).reshape(-1, 1)
        D[new_col.ravel() == 1, best_i] = 0
        D[new_col.ravel() == 1, best_j] = 0
        D = np.hstack([D, new_col])
        circuit.append((best_i, best_j))
    row_sums = np.sum(D, axis=1)
    solved = bool(np.all(row_sums <= 1))
    return len(circuit), solved, circuit


def replay_circuit_in_env(circuit, target_matrix, max_extra, max_depth):
    """将电路 (u,v) 列表在环境中重放，生成训练轨迹"""
    from gnn_env import SLPGraphEnv
    env = SLPGraphEnv(target_matrix, max_extra, max_depth)
    obs = env.reset()
    if np.all(env.achieved):
        return [], True, 0

    trajectory = []
    for u, v in circuit:
        if u >= env.max_nodes or v >= env.max_nodes:
            break
        if not env.valid[u] or not env.valid[v]:
            break
        # 检查深度约束，必要时交换 u,v
        v_mask = env.get_v_mask(u)
        if v_mask[v] == 0:
            v_mask2 = env.get_v_mask(v)
            if v_mask2[u] > 0:
                u, v = v, u
                v_mask = v_mask2
            else:
                continue  # 跳过此步

        trajectory.append({
            'features': obs['node_features'].copy(),
            'adj': obs['adj'].copy(),
            'valid_mask': obs['valid_mask'].copy(),
            'v_mask': v_mask.copy(),
            'u': u, 'v': v,
        })
        obs, reward, done, info = env.step(u, v)
        if info:  # 错误(重复/零向量等)
            trajectory.pop()
            continue
        if done:
            break

    solved = bool(np.all(env.achieved))
    return trajectory, solved, len(env.circuit)


def fast_bp(target_matrix, time_limit=300):
    """Boyar-Peralta 式贪心搜索，预计算 min_dist 加速"""
    T = np.array(target_matrix, dtype=np.int8)
    n_targets, n_inputs = T.shape
    basis = [np.zeros(n_inputs, dtype=np.int8) for i in range(n_inputs)]
    for i in range(n_inputs):
        basis[i][i] = 1
    circuit = []
    achieved = np.zeros(n_targets, dtype=bool)
    for t in range(n_targets):
        for b in basis:
            if np.array_equal(b, T[t]):
                achieved[t] = True
    t0 = time.time()
    while not np.all(achieved):
        if time.time() - t0 > time_limit:
            break
        n = len(basis)
        basis_arr = np.array(basis)

        # 预计算每个未达成 target 的当前最小距离
        unachieved = np.where(~achieved)[0]
        if len(unachieved) == 0:
            break
        # (n_unachieved, n_basis) 距离矩阵
        dist_matrix = np.array([
            np.sum(basis_arr != T[t], axis=1) for t in unachieved
        ])  # shape: (n_unachieved, n)
        min_dists = dist_matrix.min(axis=1)  # 每个 target 的当前最小距离

        # 预计算所有 basis 向量的哈希用于快速重复检查
        basis_set = set()
        for b in basis:
            basis_set.add(b.tobytes())

        best_score, best_pair = -1, None
        for i in range(n):
            if time.time() - t0 > time_limit:
                break
            # 向量化: 一次计算 i 与 i+1..n-1 的所有 XOR
            xors = (basis_arr[i] + basis_arr[i+1:]) % 2  # (n-i-1, n_inputs)
            if len(xors) == 0:
                continue
            # 过滤零向量
            nonzero = np.any(xors, axis=1)
            valid_indices = np.where(nonzero)[0]
            for jj in valid_indices:
                xor = xors[jj]
                # 快速重复检查
                if xor.tobytes() in basis_set:
                    continue
                # 向量化评分
                dists_to_targets = np.sum(xor != T[unachieved], axis=1)  # (n_unachieved,)
                score = 0
                exact_match = dists_to_targets == 0
                score += int(np.sum(exact_match)) * 100
                improve = min_dists - dists_to_targets
                score += int(np.sum(np.maximum(improve[~exact_match], 0)))
                if score > best_score:
                    best_score = score
                    best_pair = (i, i + 1 + jj)

        if best_pair is None or best_score <= 0:
            break
        i, j = best_pair
        new_vec = (basis[i] + basis[j]) % 2
        circuit.append((i, j))
        basis.append(new_vec.copy())
        for t in range(n_targets):
            if not achieved[t] and np.array_equal(new_vec, T[t]):
                achieved[t] = True
        if len(circuit) > n_inputs * 8:
            break
    return len(circuit), bool(np.all(achieved)), time.time() - t0


def fast_greedy_solve(target_matrix, time_limit=300):
    from gnn_env import SLPGraphEnv
    dim = len(target_matrix[0])
    max_extra = dim * 5
    env = SLPGraphEnv(target_matrix, max_extra, max_depth=max(dim, 20))
    expert = FastGreedyExpert()
    t0 = time.time()
    expert._time_limit = t0 + time_limit
    _, solved, n_gates = expert.generate_trajectory(env)
    expert._time_limit = None
    return n_gates, solved, time.time() - t0


# ==========================================================
# 5. 电路后处理: 死门消除 + 验证
# ==========================================================
def verify_circuit(circuit, target_matrix, num_inputs):
    """验证电路是否正确计算所有 target"""
    T = np.array(target_matrix, dtype=np.int8)
    basis = []
    for i in range(num_inputs):
        vec = np.zeros(num_inputs, dtype=np.int8)
        vec[i] = 1
        basis.append(vec)
    for u, v in circuit:
        if u >= len(basis) or v >= len(basis):
            return False
        new_vec = (basis[u].astype(np.int16) + basis[v].astype(np.int16)) % 2
        basis.append(new_vec.astype(np.int8))
    for t in range(len(T)):
        if not any(np.array_equal(b, T[t]) for b in basis):
            return False
    return True


def simplify_circuit(circuit, target_matrix, num_inputs):
    """死门消除: 反向标记从 target 可达的门，删除不可达门"""
    T = np.array(target_matrix, dtype=np.int8)
    n_targets = len(T)

    # 重建所有节点向量
    basis = []
    for i in range(num_inputs):
        vec = np.zeros(num_inputs, dtype=np.int8)
        vec[i] = 1
        basis.append(vec)
    for u, v in circuit:
        new_vec = (basis[u].astype(np.int16) + basis[v].astype(np.int16)) % 2
        basis.append(new_vec.astype(np.int8))

    # 找 target 对应的节点
    target_nodes = set()
    for t in range(n_targets):
        for i in range(len(basis) - 1, -1, -1):  # 优先最后出现的
            if np.array_equal(basis[i], T[t]):
                target_nodes.add(i)
                break

    # 反向标记需要的节点
    needed = set(target_nodes)
    for gate_idx in range(len(circuit) - 1, -1, -1):
        node_idx = num_inputs + gate_idx
        if node_idx in needed:
            u, v = circuit[gate_idx]
            needed.add(u)
            needed.add(v)

    # 构建简化电路 (重映射索引)
    index_map = {i: i for i in range(num_inputs)}
    new_circuit = []
    next_free = num_inputs

    for gate_idx, (u, v) in enumerate(circuit):
        node_idx = num_inputs + gate_idx
        if node_idx in needed:
            new_u = index_map.get(u, u)
            new_v = index_map.get(v, v)
            new_circuit.append((new_u, new_v))
            index_map[node_idx] = next_free
            next_free += 1

    # 验证简化后电路
    if verify_circuit(new_circuit, target_matrix, num_inputs):
        return new_circuit
    return circuit  # 验证失败则返回原电路


# ==========================================================
# 6. MCTS 树搜索推理
# ==========================================================
class MCTSNode:
    __slots__ = ['env', 'parent', 'action', 'children', 'visit_count',
                 'value_sum', 'prior', 'expanded']

    def __init__(self, env, parent=None, action=None, prior=0.0):
        self.env = env
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.expanded = False

    @property
    def is_terminal(self):
        return np.all(self.env.achieved) or self.env.next_idx >= self.env.max_nodes

    @property
    def is_solved(self):
        return np.all(self.env.achieved)

    @property
    def q_value(self):
        return self.value_sum / max(self.visit_count, 1)


class MCTSSolver:
    """
    AlphaGo 式 MCTS:
    - Policy network 提供先验概率
    - Value network 估计叶节点价值
    - UCB 选择 → 展开 → 回传
    """

    def __init__(self, model, device, c_puct=2.0, n_simulations=400,
                 max_children=30):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.max_children = max_children

    def solve(self, target_matrix, max_extra, max_depth, n_restarts=3):
        """多次重启 MCTS, 取最优"""
        best_gates = float('inf')
        best_circuit = None
        for _ in range(n_restarts):
            g, c = self._solve_once(target_matrix, max_extra, max_depth)
            if g is not None and g < best_gates:
                best_gates = g
                best_circuit = c
        return (best_gates if best_gates < float('inf') else None), best_circuit

    def solve_with_data(self, target_matrix, max_extra, max_depth, temperature=1.0,
                        temp_threshold=30):
        """
        AlphaZero 自博弈：运行 MCTS 并返回训练数据。
        返回 (n_gates, circuit, training_data)
        training_data: [{'obs', 'visit_counts', 'total_visits', 'step', 'outcome'}]
        """
        from gnn_env import SLPGraphEnv
        env = SLPGraphEnv(target_matrix, max_extra, max_depth)
        env.reset()
        if np.all(env.achieved):
            return 0, [], []

        self.model.eval()
        training_data = []

        for step in range(max_extra):
            if np.all(env.achieved):
                break
            if env.next_idx >= env.max_nodes:
                break

            obs = env.get_obs()
            root = MCTSNode(env.copy())

            for _ in range(self.n_simulations):
                node = root
                while node.expanded and node.children and not node.is_terminal:
                    node = self._ucb_select(node)
                if node.is_terminal:
                    value = max(0.0, 1.0 - len(node.env.circuit) / (max_extra * 0.8)) if node.is_solved else -0.5
                else:
                    value = self._expand(node)
                cur = node
                while cur is not None:
                    cur.visit_count += 1
                    cur.value_sum += value
                    cur = cur.parent

            if not root.children:
                break

            visit_counts = {}
            total_visits = 0
            for action, child in root.children.items():
                visit_counts[action] = child.visit_count
                total_visits += child.visit_count

            training_data.append({
                'obs': obs,
                'visit_counts': visit_counts,
                'total_visits': total_visits,
                'step': step,
            })

            if step < temp_threshold and temperature > 0:
                actions = list(visit_counts.keys())
                counts = np.array([visit_counts[a] for a in actions], dtype=np.float64)
                counts = counts ** (1.0 / temperature)
                probs = counts / counts.sum()
                chosen_idx = np.random.choice(len(actions), p=probs)
                best_action = actions[chosen_idx]
            else:
                best_action = max(root.children.keys(),
                                  key=lambda a: root.children[a].visit_count)

            reward, done, info = env.step_fast(*best_action)
            if info or done:
                break

        n_gates = len(env.circuit) if np.all(env.achieved) else None
        for td in training_data:
            td['outcome'] = n_gates
        return n_gates, env.circuit if n_gates else [], training_data

    def _solve_once(self, target_matrix, max_extra, max_depth):
        from gnn_env import SLPGraphEnv
        env = SLPGraphEnv(target_matrix, max_extra, max_depth)
        env.reset()
        if np.all(env.achieved):
            return 0, []

        self.model.eval()
        best_gates = float('inf')
        best_circuit = None

        for step in range(max_extra):
            if np.all(env.achieved):
                ng = len(env.circuit)
                if ng < best_gates:
                    best_gates = ng
                    best_circuit = env.circuit[:]
                break
            if env.next_idx >= env.max_nodes:
                break

            # 从当前状态运行 MCTS
            root = MCTSNode(env.copy())

            for _ in range(self.n_simulations):
                # Selection
                node = root
                while node.expanded and node.children and not node.is_terminal:
                    node = self._ucb_select(node)

                # Expansion + Evaluation
                if node.is_terminal:
                    if node.is_solved:
                        ng = len(node.env.circuit)
                        value = max(0.0, 1.0 - ng / (max_extra * 0.8))
                        if ng < best_gates:
                            best_gates = ng
                            best_circuit = node.env.circuit[:]
                    else:
                        value = -0.5
                else:
                    value = self._expand(node)

                # Backpropagation
                cur = node
                while cur is not None:
                    cur.visit_count += 1
                    cur.value_sum += value
                    cur = cur.parent

            if not root.children:
                break

            # 选访问次数最多的动作
            best_action = max(root.children.keys(),
                              key=lambda a: root.children[a].visit_count)

            # 执行动作
            reward, done, info = env.step_fast(*best_action)
            if info:
                break
            if done:
                if np.all(env.achieved):
                    ng = len(env.circuit)
                    if ng < best_gates:
                        best_gates = ng
                        best_circuit = env.circuit[:]
                break

        return (best_gates if best_gates < float('inf') else None), best_circuit

    def _ucb_select(self, node):
        sqrt_n = math.sqrt(max(node.visit_count, 1))
        best, best_score = None, -1e18
        for child in node.children.values():
            q = child.q_value
            explore = self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            score = q + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def _expand(self, node):
        """展开节点: 用 policy 生成子节点, 用 value 估计价值"""
        env = node.env
        obs = env.get_obs()

        feat = torch.FloatTensor(obs['node_features']).unsqueeze(0).to(self.device)
        adj_t = torch.FloatTensor(obs['adj']).unsqueeze(0).to(self.device)
        vmask = torch.FloatTensor(obs['valid_mask']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            h = self.model.encode(feat, adj_t)
            u_logits = self.model.get_u_logits(h, vmask).squeeze(0)
            value = self.model.get_value(h, vmask).item()

        u_probs = F.softmax(u_logits, dim=-1)
        k_u = min(10, (u_probs > 1e-6).sum().item())
        if k_u == 0:
            node.expanded = True
            return value

        top_u_p, top_u = torch.topk(u_probs, k_u)

        actions = []
        for ui in range(k_u):
            u = top_u[ui].item()
            u_p = top_u_p[ui].item()
            v_mask_np = env.get_v_mask(u)
            if np.sum(v_mask_np) == 0:
                continue
            v_mask_t = torch.FloatTensor(v_mask_np).unsqueeze(0).to(self.device)
            with torch.no_grad():
                v_logits = self.model.get_v_logits(
                    h, torch.LongTensor([u]).to(self.device), v_mask_t, vmask
                ).squeeze(0)
            v_probs = F.softmax(v_logits, dim=-1)
            k_v = min(5, (v_probs > 1e-6).sum().item())
            if k_v == 0:
                continue
            top_v_p, top_v = torch.topk(v_probs, k_v)
            for vi in range(k_v):
                actions.append(((u, top_v[vi].item()), u_p * top_v_p[vi].item()))

        if not actions:
            node.expanded = True
            return value

        actions.sort(key=lambda x: x[1], reverse=True)
        actions = actions[:self.max_children]
        total_p = sum(p for _, p in actions) + 1e-8

        for (u, v), p in actions:
            child_env = env.copy()
            reward, done, info = child_env.step_fast(u, v)
            if info:
                continue
            child = MCTSNode(child_env, parent=node, action=(u, v),
                             prior=p / total_p)
            node.children[(u, v)] = child

        node.expanded = True
        return value


# ==========================================================
# 7. Beam Search + Best-of-N
# ==========================================================
def beam_search_solve(model, target_matrix, max_extra, max_depth, device,
                      beam_width=15):
    from gnn_env import SLPGraphEnv
    env = SLPGraphEnv(target_matrix, max_extra, max_depth)
    env.reset()
    if np.all(env.achieved):
        return 0

    model.eval()
    beams = [(env, 0.0)]
    best_gates = float('inf')

    for step in range(max_extra):
        if not beams:
            break

        beam_obs = [b[0].get_obs() for b in beams]
        feats = torch.FloatTensor(np.array([o['node_features'] for o in beam_obs])).to(device)
        adjs = torch.FloatTensor(np.array([o['adj'] for o in beam_obs])).to(device)
        vmasks = torch.FloatTensor(np.array([o['valid_mask'] for o in beam_obs])).to(device)

        with torch.no_grad():
            h = model.encode(feats, adjs)
            u_logits = model.get_u_logits(h, vmasks)

        candidates = []
        for bi, (env_state, beam_logp) in enumerate(beams):
            if np.all(env_state.achieved):
                ng = len(env_state.circuit)
                if ng < best_gates:
                    best_gates = ng
                continue
            if env_state.next_idx >= env_state.max_nodes:
                continue

            u_lp = F.log_softmax(u_logits[bi], dim=-1)
            n_valid = (u_lp > -1e8).sum().item()
            if n_valid == 0:
                continue
            k_u = min(beam_width, n_valid)
            top_u_logp, top_u = torch.topk(u_lp, k_u)

            for ui in range(k_u):
                u = top_u[ui].item()
                v_mask_np = env_state.get_v_mask(u)
                if np.sum(v_mask_np) == 0:
                    continue
                v_mask_t = torch.FloatTensor(v_mask_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    v_logits = model.get_v_logits(
                        h[bi:bi+1], torch.LongTensor([u]).to(device),
                        v_mask_t, vmasks[bi:bi+1]
                    ).squeeze(0)
                v_lp = F.log_softmax(v_logits, dim=-1)
                n_valid_v = (v_lp > -1e8).sum().item()
                if n_valid_v == 0:
                    continue
                k_v = min(3, n_valid_v)
                top_v_logp, top_v = torch.topk(v_lp, k_v)
                for vi in range(k_v):
                    total = beam_logp + top_u_logp[ui].item() + top_v_logp[vi].item()
                    candidates.append((bi, u, top_v[vi].item(), total))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[3], reverse=True)
        candidates = candidates[:beam_width * 3]

        new_beams = []
        seen = set()
        for bi, u, v, logp in candidates:
            env_copy = beams[bi][0].copy()
            reward, done, info = env_copy.step_fast(u, v)
            if info:
                continue
            if np.all(env_copy.achieved):
                ng = len(env_copy.circuit)
                if ng < best_gates:
                    best_gates = ng
                continue
            sig = tuple(env_copy.circuit)
            if sig in seen:
                continue
            seen.add(sig)
            new_beams.append((env_copy, logp))
            if len(new_beams) >= beam_width:
                break

        beams = new_beams

    return best_gates if best_gates < float('inf') else None


def best_of_n_evaluate(model, target_matrix, max_extra, max_depth, device,
                       n_deterministic=5, n_sample=100, n_high_temp=30):
    from gnn_env import SLPGraphEnv
    from train_gnn_slp import collect_episode

    model.eval()
    gate_counts = []

    for _ in range(n_deterministic):
        env = SLPGraphEnv(target_matrix, max_extra, max_depth)
        _, _, solved, ng = collect_episode(env, model, device, deterministic=True)
        if solved:
            gate_counts.append(ng)

    for _ in range(n_sample):
        env = SLPGraphEnv(target_matrix, max_extra, max_depth)
        _, _, solved, ng = collect_episode(env, model, device, deterministic=False)
        if solved:
            gate_counts.append(ng)

    for _ in range(n_high_temp):
        env = SLPGraphEnv(target_matrix, max_extra, max_depth)
        solved, ng = _sample_temperature(env, model, device, 1.5)
        if solved:
            gate_counts.append(ng)

    total = n_deterministic + n_sample + n_high_temp
    if not gate_counts:
        return {'solved': False, 'min_gates': None, 'avg_gates': None,
                'solve_rate': 0, 'n_tried': total}
    return {
        'solved': True,
        'min_gates': min(gate_counts),
        'avg_gates': float(np.mean(gate_counts)),
        'solve_rate': len(gate_counts) / total,
        'n_tried': total,
    }


def _sample_temperature(env, model, device, temperature):
    obs = env.reset()
    for _ in range(env.max_extra):
        feat = torch.FloatTensor(obs['node_features']).unsqueeze(0).to(device)
        adj_t = torch.FloatTensor(obs['adj']).unsqueeze(0).to(device)
        vmask = torch.FloatTensor(obs['valid_mask']).unsqueeze(0).to(device)
        with torch.no_grad():
            h = model.encode(feat, adj_t)
            u_logits = model.get_u_logits(h, vmask).squeeze(0) / temperature
        u = Categorical(logits=u_logits).sample().item()
        v_mask_np = env.get_v_mask(u)
        if np.sum(v_mask_np) == 0:
            break
        v_mask_t = torch.FloatTensor(v_mask_np).unsqueeze(0).to(device)
        with torch.no_grad():
            v_logits = model.get_v_logits(
                h, torch.LongTensor([u]).to(device), v_mask_t, vmask
            ).squeeze(0) / temperature
        v = Categorical(logits=v_logits).sample().item()
        obs, reward, done, info = env.step(u, v)
        if done:
            break
    return bool(np.all(env.achieved)), len(env.circuit)


# ==========================================================
# 8. 综合评估 (合并所有推理策略)
# ==========================================================
def comprehensive_evaluate(model, target_matrix, max_extra, max_depth, device,
                           quick=False):
    """Best-of-N + Beam Search + MCTS + 电路简化 → 取最优"""
    is_cpu = (device == 'cpu')
    n_sample = 30 if quick else (60 if is_cpu else 100)
    n_high = 10 if quick else (15 if is_cpu else 30)
    bw = 8 if quick else (10 if is_cpu else 15)
    mcts_sims = 150 if quick else (200 if is_cpu else 400)
    mcts_restarts = 2 if quick else (3 if is_cpu else 5)
    num_inputs = len(target_matrix[0])

    print(f"      BoN ({3 + n_sample + n_high} runs)...", end="", flush=True)
    bon = best_of_n_evaluate(model, target_matrix, max_extra, max_depth, device,
                             n_deterministic=3, n_sample=n_sample, n_high_temp=n_high)
    bon_min = bon.get('min_gates')
    print(f" min={bon_min}")

    print(f"      Beam (width={bw})...", end="", flush=True)
    beam_min = beam_search_solve(model, target_matrix, max_extra, max_depth, device,
                                 beam_width=bw)
    print(f" gates={beam_min}")

    print(f"      MCTS ({mcts_sims} sims x {mcts_restarts} restarts)...", end="", flush=True)
    mcts = MCTSSolver(model, device, n_simulations=mcts_sims, max_children=25)
    mcts_gates, mcts_circuit = mcts.solve(target_matrix, max_extra, max_depth,
                                          n_restarts=mcts_restarts)
    print(f" gates={mcts_gates}")

    # 电路简化
    mcts_simplified = mcts_gates
    if mcts_circuit:
        simp = simplify_circuit(mcts_circuit, target_matrix, num_inputs)
        if verify_circuit(simp, target_matrix, num_inputs):
            mcts_simplified = len(simp)
            if mcts_simplified < mcts_gates:
                print(f"      Simplified: {mcts_gates} → {mcts_simplified}")

    all_gates = [g for g in [bon_min, beam_min, mcts_gates, mcts_simplified]
                 if g is not None]

    return {
        'best_gates': min(all_gates) if all_gates else None,
        'bon_min': bon_min,
        'bon_avg': bon.get('avg_gates'),
        'bon_solve_rate': bon.get('solve_rate', 0),
        'beam_gates': beam_min,
        'mcts_gates': mcts_gates,
        'mcts_simplified': mcts_simplified,
    }


# ==========================================================
# 9. 高质量专家数据生成
# ==========================================================
def generate_quality_expert_data(target_matrix, max_extra, max_depth,
                                 n_target_restarts=80, n_random=150,
                                 use_lookahead=True):
    from gnn_env import SLPGraphEnv

    n_inputs = len(target_matrix[0])
    n_targets = len(target_matrix)
    all_data = []
    target_trajs = []

    # === 1) Paar 算法: 最可靠的专家 ===
    paar_ng, paar_solved, paar_circuit = paar_with_circuit(target_matrix)
    if paar_solved and paar_circuit:
        traj, env_solved, env_ng = replay_circuit_in_env(
            paar_circuit, target_matrix, max_extra, max_depth)
        if env_solved and traj:
            target_trajs.append((env_ng, traj))
            print(f"    Paar: {paar_ng} gates (env replay: {env_ng} gates, {len(traj)} trans)")
        else:
            print(f"    Paar: {paar_ng} gates 但 env 重放失败 (solved={env_solved}, traj={len(traj)})")
    else:
        print(f"    Paar: 未解出 (gates={paar_ng})")

    # === 2) 贪心专家 multi-restart ===
    expert_det = FastGreedyExpert(temperature=0.0)
    expert_noisy = FastGreedyExpert(temperature=2.0)
    expert_med = FastGreedyExpert(temperature=1.0)

    if use_lookahead and n_inputs <= 16:
        expert_la = LookaheadGreedyExpert(temperature=0.0)
        expert_la_noisy = LookaheadGreedyExpert(temperature=1.5)
    else:
        expert_la = expert_det
        expert_la_noisy = expert_noisy

    greedy_solved = 0
    for i in range(n_target_restarts):
        env = SLPGraphEnv(target_matrix, max_extra, max_depth)
        if i < n_target_restarts * 0.05:
            exp = expert_la
        elif i < n_target_restarts * 0.15:
            exp = expert_det
        elif i < n_target_restarts * 0.4:
            exp = expert_la_noisy if n_inputs <= 16 else expert_med
        else:
            exp = expert_noisy
        traj, solved, n_gates = exp.generate_trajectory(env)
        if solved and traj:
            target_trajs.append((n_gates, traj))
            greedy_solved += 1

    target_trajs.sort(key=lambda x: x[0])
    best_ng = target_trajs[0][0] if target_trajs else 999
    print(f"    目标矩阵: {greedy_solved}/{n_target_restarts} greedy solved, best={best_ng}")

    # Top 30% 轨迹权重 x3
    top_k = max(1, len(target_trajs) * 3 // 10)
    for _, traj in target_trajs[:top_k]:
        all_data.extend(traj * 3)
    for _, traj in target_trajs[top_k:]:
        all_data.extend(traj)

    # === 3) 随机矩阵 (用 Paar + greedy) ===
    rand_solved = 0
    for i in range(n_random):
        rng = np.random.RandomState(i + 42)
        m = rng.randint(0, 2, (n_targets, n_inputs)).astype(np.int8)
        for r in range(n_targets):
            if not np.any(m[r]):
                m[r, rng.randint(0, n_inputs)] = 1
        m_list = m.tolist()
        # 先尝试 Paar
        p_ng, p_ok, p_circ = paar_with_circuit(m_list)
        if p_ok and p_circ:
            traj, solved, _ = replay_circuit_in_env(p_circ, m_list, max_extra, max_depth)
            if solved and traj:
                all_data.extend(traj)
                rand_solved += 1
                continue
        # Paar 失败则用贪心
        env = SLPGraphEnv(m_list, max_extra, max_depth)
        traj, solved, _ = expert_det.generate_trajectory(env)
        if solved and traj:
            all_data.extend(traj)
            rand_solved += 1

    print(f"    随机矩阵: {rand_solved}/{n_random}, 总计: {len(all_data)} transitions")
    return all_data


# ==========================================================
# 10. 改进版 GNN 训练
# ==========================================================
def train_gnn_improved(target_matrix, num_inputs, max_extra, max_depth,
                       hidden_dim, device, il_epochs=100, ppo_iters=300,
                       episodes_per_iter=24, label="", quick=False):
    from gnn_env import SLPGraphEnv
    from gnn_network import SLPPolicyValueNet
    from train_gnn_slp import train_il, collect_episode, compute_gae, ppo_update, evaluate

    env = SLPGraphEnv(target_matrix, max_extra, max_depth)
    feature_dim = env.feature_dim
    model = SLPPolicyValueNet(feature_dim, hidden_dim, num_gnn_layers=4).to(device)
    print(f"  [{label}] params={sum(p.numel() for p in model.parameters()):,}")

    # === Expert Data ===
    t0 = time.time()
    n_restarts = 30 if quick else 80
    n_rand = 40 if quick else 150
    expert_data = generate_quality_expert_data(
        target_matrix, max_extra, max_depth,
        n_target_restarts=n_restarts, n_random=n_rand,
        use_lookahead=(num_inputs <= 16),
    )
    print(f"  [{label}] Expert: {len(expert_data)} trans ({time.time()-t0:.1f}s)")
    if not expert_data:
        return None

    # === IL with Cosine LR ===
    t0 = time.time()
    il_opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    il_sched = optim.lr_scheduler.CosineAnnealingLR(il_opt, T_max=il_epochs, eta_min=1e-5)

    model.train()
    T_data = len(expert_data)
    indices = np.arange(T_data)
    bs = 128

    for epoch in range(il_epochs):
        np.random.shuffle(indices)
        total_loss, nb = 0, 0
        for start in range(0, T_data, bs):
            end = min(start + bs, T_data)
            idx = indices[start:end]
            feat = torch.FloatTensor(np.array([expert_data[i]['features'] for i in idx])).to(device)
            adj = torch.FloatTensor(np.array([expert_data[i]['adj'] for i in idx])).to(device)
            vmask = torch.FloatTensor(np.array([expert_data[i]['valid_mask'] for i in idx])).to(device)
            v_masks = torch.FloatTensor(np.array([expert_data[i]['v_mask'] for i in idx])).to(device)
            u_tgt = torch.LongTensor([expert_data[i]['u'] for i in idx]).to(device)
            v_tgt = torch.LongTensor([expert_data[i]['v'] for i in idx]).to(device)

            u_logits, v_logits, _ = model(feat, adj, vmask, u_tgt, v_masks)
            loss = F.cross_entropy(u_logits, u_tgt) + F.cross_entropy(v_logits, v_tgt)
            il_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            il_opt.step()
            total_loss += loss.item()
            nb += 1
        il_sched.step()
        if (epoch + 1) % 20 == 0:
            print(f"  [{label}] IL {epoch+1}/{il_epochs}: loss={total_loss/nb:.4f}")

    ev = evaluate(model, target_matrix, max_extra, max_depth, device, 10)
    il_sr = sum(r['solved'] for r in ev) / 10
    il_g = [r['gates'] for r in ev if r['solved']]
    il_g_str = f"{np.mean(il_g):.1f}" if il_g else "N/A"
    print(f"  [{label}] Post-IL: solve={il_sr:.0%}, "
          f"gates={il_g_str} ({time.time()-t0:.1f}s)")

    # === PPO + Self-Imitation ===
    t0 = time.time()
    ppo_opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    ppo_sched = optim.lr_scheduler.CosineAnnealingLR(ppo_opt, T_max=ppo_iters, eta_min=5e-6)

    best_gates = float('inf')
    best_circuit = None
    top_trajs = []  # 自模仿 buffer

    for it in range(ppo_iters):
        model.train()
        all_t, all_r, all_a = [], [], []

        for _ in range(episodes_per_iter):
            env_tmp = SLPGraphEnv(target_matrix, max_extra, max_depth)
            trans, _, solved, ng = collect_episode(env_tmp, model, device)
            if not trans:
                continue
            ret, adv = compute_gae(trans)
            all_t.append(trans)
            all_r.append(ret)
            all_a.append(adv)

            if solved:
                if ng < best_gates:
                    best_gates = ng
                    best_circuit = env_tmp.circuit.copy()
                    print(f"  [{label}] ★ NEW BEST: {best_gates} gates (PPO {it+1})")
                # 自模仿 buffer
                top_trajs.append((ng, [
                    {k: t[k] for k in ('features', 'adj', 'valid_mask', 'v_mask', 'u', 'v')}
                    for t in trans
                ]))
                top_trajs.sort(key=lambda x: x[0])
                top_trajs = top_trajs[:40]

        if all_t:
            ppo_update(model, ppo_opt, all_t, all_r, all_a,
                       device=device, ppo_epochs=4, batch_size=128)
        ppo_sched.step()

        # 自模仿: 每 20 轮用最佳轨迹做 IL
        if (it + 1) % 20 == 0 and top_trajs:
            si_data = []
            for _, traj in top_trajs[:15]:
                si_data.extend(traj)
            if si_data:
                old_lr = ppo_opt.param_groups[0]['lr']
                ppo_opt.param_groups[0]['lr'] = min(old_lr * 2, 5e-4)
                train_il(model, ppo_opt, si_data, device, epochs=3, batch_size=64)
                ppo_opt.param_groups[0]['lr'] = old_lr

        if (it + 1) % 50 == 0:
            ev = evaluate(model, target_matrix, max_extra, max_depth, device, 10)
            sr = sum(r['solved'] for r in ev) / 10
            gs = [r['gates'] for r in ev if r['solved']]
            gs_str = f"{np.mean(gs):.1f}" if gs else "N/A"
            print(f"  [{label}] PPO {it+1}: solve={sr:.0%}, "
                  f"gates={gs_str}, best={best_gates}")

    ppo_time = time.time() - t0

    # === 电路简化 PPO 最优 ===
    best_simplified = best_gates
    if best_circuit:
        simp = simplify_circuit(best_circuit, target_matrix, num_inputs)
        if verify_circuit(simp, target_matrix, num_inputs):
            best_simplified = len(simp)
            if best_simplified < best_gates:
                print(f"  [{label}] PPO best simplified: {best_gates} → {best_simplified}")

    # === 综合评估 ===
    print(f"  [{label}] Comprehensive evaluation...")
    final = comprehensive_evaluate(model, target_matrix, max_extra, max_depth, device,
                                   quick=quick)

    result = {
        'il_solve_rate': il_sr,
        'il_avg_gates': float(np.mean(il_g)) if il_g else None,
        'ppo_best_gates': best_gates if best_gates < float('inf') else None,
        'ppo_best_simplified': best_simplified if best_simplified < float('inf') else None,
        'ppo_time': ppo_time,
        **final,
    }
    print(f"  [{label}] FINAL: best={result.get('best_gates')}, "
          f"mcts={result.get('mcts_gates')}, beam={result.get('beam_gates')}, "
          f"bon={result.get('bon_min')}, solve%={result.get('bon_solve_rate', 0):.0%}")

    torch.save(model.state_dict(), f'model_{label}.pt')
    return result


# ==========================================================
# 11. 消融实验
# ==========================================================
def run_key_ablations(target_matrix, device, quick=False):
    from gnn_env import SLPGraphEnv
    from gnn_network import SLPPolicyValueNet
    from ablation import (
        TransformerPolicyValueNet, MLPPolicyValueNet,
        BinarySLPEnv, train_a2c, train_reinforce,
    )
    from train_gnn_slp import train_il, collect_episode, compute_gae, ppo_update, evaluate

    max_extra, max_depth, hidden = 80, 16, 128
    il_ep = 30 if quick else 60
    ppo_it = 30 if quick else 80

    # Paar + greedy 生成专家数据
    expert = FastGreedyExpert(temperature=2.0)
    expert_det = FastGreedyExpert(temperature=0.0)

    # 先用 Paar 获取可靠轨迹
    expert_data_alg = []
    _, paar_ok, paar_circ = paar_with_circuit(target_matrix)
    if paar_ok and paar_circ:
        traj, solved, ng = replay_circuit_in_env(paar_circ, target_matrix, max_extra, max_depth)
        if solved and traj:
            expert_data_alg.extend(traj * 3)  # Paar 轨迹权重 x3
            print(f"  Paar expert (algebra): {ng} gates, {len(traj)} trans")
    # 再用贪心补充
    for _ in range(40):
        env_tmp = SLPGraphEnv(target_matrix, max_extra, max_depth)
        traj, solved, _ = expert.generate_trajectory(env_tmp)
        if solved and traj:
            expert_data_alg.extend(traj)
    for _ in range(10):
        env_tmp = SLPGraphEnv(target_matrix, max_extra, max_depth)
        traj, solved, _ = expert_det.generate_trajectory(env_tmp)
        if solved and traj:
            expert_data_alg.extend(traj)
    print(f"  Expert data (algebra): {len(expert_data_alg)}")

    expert_data_bin = []
    if paar_ok and paar_circ:
        # BinarySLPEnv 重放 Paar
        env_bin = BinarySLPEnv(target_matrix, max_extra, max_depth)
        obs_bin = env_bin.reset()
        bin_traj = []
        for u, v in paar_circ:
            if u >= env_bin.max_nodes or v >= env_bin.max_nodes:
                break
            if not env_bin.valid[u] or not env_bin.valid[v]:
                break
            v_mask = env_bin.get_v_mask(u)
            if v_mask[v] == 0:
                v_mask2 = env_bin.get_v_mask(v)
                if v_mask2[u] > 0:
                    u, v = v, u
                    v_mask = v_mask2
                else:
                    continue
            bin_traj.append({
                'features': obs_bin['node_features'].copy(),
                'adj': obs_bin['adj'].copy(),
                'valid_mask': obs_bin['valid_mask'].copy(),
                'v_mask': v_mask.copy(),
                'u': u, 'v': v,
            })
            obs_bin, reward, done, info = env_bin.step(u, v)
            if info:
                bin_traj.pop()
                continue
            if done:
                break
        if np.all(env_bin.achieved) and bin_traj:
            expert_data_bin.extend(bin_traj * 3)
    for _ in range(40):
        env_tmp = BinarySLPEnv(target_matrix, max_extra, max_depth)
        traj, solved, _ = expert.generate_trajectory(env_tmp)
        if solved and traj:
            expert_data_bin.extend(traj)
    print(f"  Expert data (binary): {len(expert_data_bin)}")

    KEY_ABLATIONS = {
        'il_only':      {'encoder': 'gnn', 'ppo': 0,      'feat': 'alg', 'layers': 4, 'rl': 'ppo'},
        'full_gnn_ppo': {'encoder': 'gnn', 'ppo': ppo_it, 'feat': 'alg', 'layers': 4, 'rl': 'ppo'},
        'transformer':  {'encoder': 'transformer', 'ppo': ppo_it, 'feat': 'alg', 'layers': 4, 'rl': 'ppo'},
        'mlp':          {'encoder': 'mlp', 'ppo': ppo_it, 'feat': 'alg', 'layers': 4, 'rl': 'ppo'},
        'binary_feat':  {'encoder': 'gnn', 'ppo': ppo_it, 'feat': 'bin', 'layers': 4, 'rl': 'ppo'},
        'reinforce':    {'encoder': 'gnn', 'ppo': ppo_it, 'feat': 'alg', 'layers': 4, 'rl': 'reinforce'},
    }

    results = {}
    for name, cfg in KEY_ABLATIONS.items():
        torch.manual_seed(42)
        np.random.seed(42)
        print(f"\n  --- Ablation: {name} ---")

        use_bin = cfg['feat'] == 'bin'
        env_cls = BinarySLPEnv if use_bin else SLPGraphEnv
        env_tmp = env_cls(target_matrix, max_extra, max_depth)
        feat_dim = env_tmp.feature_dim
        expert_data = expert_data_bin if use_bin else expert_data_alg

        if not expert_data:
            results[name] = {'error': 'no expert data'}
            continue

        enc = cfg['encoder']
        nl = cfg['layers']
        if enc == 'gnn':
            model = SLPPolicyValueNet(feat_dim, hidden, nl).to(device)
        elif enc == 'transformer':
            model = TransformerPolicyValueNet(feat_dim, hidden, nl).to(device)
        else:
            model = MLPPolicyValueNet(feat_dim, hidden, nl).to(device)

        il_opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        train_il(model, il_opt, expert_data, device, epochs=il_ep, batch_size=128)

        ev = evaluate(model, target_matrix, max_extra, max_depth, device, 10)
        il_sr = sum(r['solved'] for r in ev) / 10

        if cfg['ppo'] > 0:
            rl_opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
            env_fn = lambda: env_cls(target_matrix, max_extra, max_depth)
            for it in range(cfg['ppo']):
                if cfg['rl'] == 'ppo':
                    all_t, all_r, all_a = [], [], []
                    for _ in range(12):
                        e = env_fn()
                        tr, _, _, _ = collect_episode(e, model, device)
                        if tr:
                            r, a = compute_gae(tr)
                            all_t.append(tr)
                            all_r.append(r)
                            all_a.append(a)
                    if all_t:
                        ppo_update(model, rl_opt, all_t, all_r, all_a,
                                   device=device, ppo_epochs=3, batch_size=64)
                elif cfg['rl'] == 'reinforce':
                    train_reinforce(model, rl_opt, env_fn, device, n_episodes=12)

        ev = evaluate(model, target_matrix, max_extra, max_depth, device, 15)
        sr = sum(r['solved'] for r in ev) / 15
        gs = [r['gates'] for r in ev if r['solved']]
        results[name] = {
            'il_solve': il_sr, 'final_solve': sr,
            'avg_gates': float(np.mean(gs)) if gs else None,
            'min_gates': min(gs) if gs else None,
        }
        gs_str = f"{np.mean(gs):.1f}" if gs else "N/A"
        print(f"    Result: solve={sr:.0%}, gates={gs_str}")

    return results


# ==========================================================
# 12. 主流程
# ==========================================================
def main():
    args = sys.argv[1:]
    use_gpu = '--gpu' in args
    quick = '--quick' in args
    only_phase = None
    if '--phase' in args:
        idx = args.index('--phase')
        only_phase = int(args[idx + 1])

    device = 'cuda' if (use_gpu or torch.cuda.is_available()) else 'cpu'
    print(f"Device: {device}")
    print(f"Mode: {'QUICK' if quick else 'OVERNIGHT v2 (MCTS + simplify + lookahead)'}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    from benchmark_matrices import load_all_benchmarks
    benchmarks = load_all_benchmarks()
    all_results = {}

    # ============================================================
    # Phase 1: Baselines
    # ============================================================
    if only_phase is None or only_phase == 1:
        print("=" * 65)
        print("PHASE 1: Baselines on All Benchmarks")
        print("=" * 65)
        t_start = time.time()
        baseline_results = {}

        for bm_name, bm_info in benchmarks.items():
            matrix = bm_info['matrix']
            dim = bm_info['dim']
            sota = bm_info['sota_gates']
            tl = 60 if dim <= 32 else 300
            if quick:
                tl = min(tl, 30)

            print(f"\n  {bm_name} ({dim}x{dim}, SOTA={sota})")
            entry = {'dim': dim, 'sota': sota}

            for algo_name, algo_fn in [('paar', fast_paar), ('bp', fast_bp)]:
                t0 = time.time()
                try:
                    if algo_name == 'paar':
                        ng, ok = algo_fn(matrix)
                        elapsed = time.time() - t0
                    else:
                        ng, ok, elapsed = algo_fn(matrix, time_limit=tl)
                    entry[algo_name] = {'gates': ng, 'solved': ok, 'time': round(elapsed, 2)}
                    print(f"    {algo_name:8s}: {ng} gates ({elapsed:.1f}s)")
                except Exception as e:
                    entry[algo_name] = {'error': str(e)}
                    print(f"    {algo_name:8s}: ERROR {e}")

            # Multi-restart greedy (跳过 64x64: 贪心在大矩阵上太慢且无效)
            if dim <= 32:
                t0 = time.time()
                try:
                    best_ng = float('inf')
                    n_restart = 3 if quick else 8
                    for _ in range(n_restart):
                        ng, ok, _ = fast_greedy_solve(matrix, time_limit=tl)
                        if ok and ng < best_ng:
                            best_ng = ng
                    entry['greedy'] = {'gates': best_ng if best_ng < float('inf') else -1,
                                       'solved': best_ng < float('inf'),
                                       'time': round(time.time() - t0, 2)}
                    print(f"    {'greedy':8s}: {entry['greedy']['gates']} gates ({time.time()-t0:.1f}s)")
                except Exception as e:
                    entry['greedy'] = {'error': str(e)}
            else:
                entry['greedy'] = {'gates': -1, 'solved': False, 'time': 0, 'skipped': True}
                print(f"    {'greedy':8s}: SKIPPED (dim={dim} too large for greedy)")

            # Lookahead greedy (小矩阵)
            if dim <= 16:
                t0 = time.time()
                try:
                    from gnn_env import SLPGraphEnv
                    best_ng_la = float('inf')
                    for r in range(5 if not quick else 2):
                        env = SLPGraphEnv(matrix, dim * 5, max(dim, 20))
                        exp = LookaheadGreedyExpert(temperature=0.5 * r)
                        _, solved, ng = exp.generate_trajectory(env)
                        if solved and ng < best_ng_la:
                            best_ng_la = ng
                    entry['lookahead'] = {'gates': best_ng_la if best_ng_la < float('inf') else -1,
                                          'time': round(time.time() - t0, 2)}
                    print(f"    {'lookahead':8s}: {entry['lookahead']['gates']} gates ({time.time()-t0:.1f}s)")
                except Exception as e:
                    entry['lookahead'] = {'error': str(e)}

            baseline_results[bm_name] = entry

        all_results['baselines'] = baseline_results
        print(f"\nPhase 1 done in {time.time()-t_start:.0f}s")

    # ============================================================
    # Phase 2: GNN on Midori 16x16
    # ============================================================
    if only_phase is None or only_phase == 2:
        print("\n" + "=" * 65)
        print("PHASE 2: GNN on Midori 16x16 (MCTS + simplify + lookahead)")
        print("=" * 65)
        t_start = time.time()
        from benchmark_matrices import get_midori_16x16_matrix
        matrix = get_midori_16x16_matrix()

        # Paar 需要 48 gates → max_extra 至少 50; 给 PPO 探索留余量
        res = train_gnn_improved(
            matrix, 16, max_extra=80, max_depth=16, hidden_dim=128, device=device,
            il_epochs=40 if quick else 100,
            ppo_iters=50 if quick else 300,
            episodes_per_iter=16 if quick else 24,
            label="midori16", quick=quick,
        )
        all_results['gnn_midori16'] = res
        print(f"Phase 2 done in {time.time()-t_start:.0f}s")

    # ============================================================
    # Phase 3: GNN on AES 32x32
    # ============================================================
    if only_phase is None or only_phase == 3:
        print("\n" + "=" * 65)
        print("PHASE 3: GNN on AES MixColumns 32x32")
        print("=" * 65)
        t_start = time.time()
        from benchmark_matrices import get_aes_mixcolumns_matrix, get_aes_inv_mixcolumns_matrix

        matrix = get_aes_mixcolumns_matrix()
        # Paar 需要 108 gates → max_extra 至少 120; max_depth 放宽
        res = train_gnn_improved(
            matrix, 32, max_extra=160, max_depth=20, hidden_dim=256, device=device,
            il_epochs=30 if quick else 80,
            ppo_iters=30 if quick else 200,
            episodes_per_iter=12 if quick else 20,
            label="aes_mc32", quick=quick,
        )
        all_results['gnn_aes_mc32'] = res

        if not quick:
            print("\n  --- AES InvMixColumns 32x32 ---")
            matrix2 = get_aes_inv_mixcolumns_matrix()
            # Paar 需要 160 gates → max_extra 至少 170
            res2 = train_gnn_improved(
                matrix2, 32, max_extra=200, max_depth=20, hidden_dim=256, device=device,
                il_epochs=80, ppo_iters=200, episodes_per_iter=20,
                label="aes_imc32",
            )
            all_results['gnn_aes_imc32'] = res2

        print(f"Phase 3 done in {time.time()-t_start:.0f}s")

    # ============================================================
    # Phase 4: Key Ablations
    # ============================================================
    if only_phase is None or only_phase == 4:
        print("\n" + "=" * 65)
        print("PHASE 4: Key Ablation Studies (Midori 16x16)")
        print("=" * 65)
        t_start = time.time()
        from benchmark_matrices import get_midori_16x16_matrix
        matrix = get_midori_16x16_matrix()
        abl_res = run_key_ablations(matrix, device, quick=quick)
        all_results['ablations'] = abl_res
        print(f"\nPhase 4 done in {time.time()-t_start:.0f}s")

    # ============================================================
    # Phase 5: Summary
    # ============================================================
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    if 'baselines' in all_results:
        print(f"\n{'Matrix':30s} | {'SOTA':>5s} | {'Paar':>5s} | {'BP':>5s} | "
              f"{'Grdy':>5s} | {'LA':>5s} | {'GNN':>5s} | {'MCTS':>5s} | {'Simp':>5s}")
        print("-" * 100)
        for bm_name, entry in all_results['baselines'].items():
            sota = str(entry.get('sota') or '-')
            paar = str(entry.get('paar', {}).get('gates', '-'))
            bp = str(entry.get('bp', {}).get('gates', '-'))
            gr = str(entry.get('greedy', {}).get('gates', '-'))
            la = str(entry.get('lookahead', {}).get('gates', '-'))

            gnn_best, mcts_g, simp_g = '-', '-', '-'
            for k, v in all_results.items():
                if k.startswith('gnn_') and isinstance(v, dict):
                    if ((bm_name == 'midori_16x16' and k == 'gnn_midori16') or
                        (bm_name == 'aes_mixcolumns_32x32' and k == 'gnn_aes_mc32') or
                        (bm_name == 'aes_inv_mixcolumns_32x32' and k == 'gnn_aes_imc32')):
                        bg = v.get('best_gates')
                        if bg:
                            gnn_best = str(bg)
                        mg = v.get('mcts_gates')
                        if mg:
                            mcts_g = str(mg)
                        sg = v.get('mcts_simplified')
                        if sg and sg != mg:
                            simp_g = str(sg)

            print(f"{bm_name:30s} | {sota:>5s} | {paar:>5s} | {bp:>5s} | "
                  f"{gr:>5s} | {la:>5s} | {gnn_best:>5s} | {mcts_g:>5s} | {simp_g:>5s}")

    if 'ablations' in all_results:
        print(f"\n{'Ablation':20s} | {'Solve%':>6s} | {'AvgGates':>8s} | {'MinGates':>8s}")
        print("-" * 50)
        for name, res in all_results['ablations'].items():
            if 'error' in res:
                print(f"{name:20s} | ERROR")
            else:
                sr = f"{res['final_solve']:.0%}"
                avg = f"{res['avg_gates']:.1f}" if res.get('avg_gates') else '-'
                mn = str(res.get('min_gates', '-'))
                print(f"{name:20s} | {sr:>6s} | {avg:>8s} | {mn:>8s}")

    print("\n--- GNN Detailed ---")
    for k, v in all_results.items():
        if k.startswith('gnn_') and isinstance(v, dict):
            print(f"\n  {k}:")
            for metric in ['ppo_best_gates', 'ppo_best_simplified', 'best_gates',
                           'mcts_gates', 'mcts_simplified', 'beam_gates',
                           'bon_min', 'bon_avg', 'bon_solve_rate']:
                val = v.get(metric, 'N/A')
                if isinstance(val, float):
                    val = f"{val:.1f}" if val > 1 else f"{val:.0%}"
                print(f"    {metric:25s}: {val}")

    os.makedirs('experiment_results', exist_ok=True)
    with open('experiment_results/overnight_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nEnd: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results → experiment_results/overnight_results.json")


if __name__ == "__main__":
    main()
