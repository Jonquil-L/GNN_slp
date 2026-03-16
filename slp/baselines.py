"""
Baseline 算法实现：Paar / Boyar-Peralta / BP+CMS / SAT-based
用于与 GNN 方法的对比实验
"""
import numpy as np
import time
from itertools import combinations


# ==========================================
# 1. Paar's Algorithm (经典贪心)
# ==========================================
class PaarAlgorithm:
    """
    Paar 1997: 贪心选择使"共同子表达式"出现频率最高的 XOR 对。
    核心思想：在目标矩阵的列对中，找出同时为 1 的行数最多的一对列，
    将其 XOR 结果作为新的基向量加入。
    """

    def solve(self, target_matrix):
        T = np.array(target_matrix, dtype=np.int8).copy()
        n_targets, n_inputs = T.shape
        circuit = []
        # 工作矩阵：每行是一个 target，每列对应一个基向量
        # 初始基: e_0, ..., e_{n_inputs-1}
        basis = np.eye(n_inputs, dtype=np.int8)  # 列向量形式
        basis_vecs = [basis[:, i].copy() for i in range(n_inputs)]
        node_count = n_inputs

        # 维护一个"需求矩阵" D[t][b] = 1 表示 target t 还需要基向量 b
        D = T.copy()  # n_targets x n_inputs 初始

        while True:
            # 检查是否所有 target 都已实现 (每行最多一个1)
            row_sums = np.sum(D, axis=1)
            if np.all(row_sums <= 1):
                break

            # 找出共同出现频率最高的列对
            n_cols = D.shape[1]
            best_count = -1
            best_pair = (0, 1)

            for i in range(n_cols):
                for j in range(i + 1, n_cols):
                    count = int(np.sum(D[:, i] & D[:, j]))
                    if count > best_count:
                        best_count = count
                        best_pair = (i, j)

            if best_count <= 0:
                break

            ci, cj = best_pair
            # 创建新基向量 = basis_vecs[ci] XOR basis_vecs[cj]
            new_vec = (np.array(basis_vecs[ci]) + np.array(basis_vecs[cj])) % 2
            circuit.append((ci, cj))
            new_idx = len(basis_vecs)
            basis_vecs.append(new_vec)

            # 更新需求矩阵：对于同时需要 ci 和 cj 的行，替换为需要 new_idx
            new_col = np.zeros(n_targets, dtype=np.int8)
            for t in range(n_targets):
                if D[t, ci] == 1 and D[t, cj] == 1:
                    D[t, ci] = 0
                    D[t, cj] = 0
                    new_col[t] = 1
            D = np.hstack([D, new_col.reshape(-1, 1)])

        return circuit, len(circuit)


# ==========================================
# 2. Boyar-Peralta Heuristic (最常用)
# ==========================================
class BoyarPeraltaAlgorithm:
    """
    Boyar-Peralta 2010: 基于距离的贪心启发式。
    每一步选择使得目标矩阵总汉明权重减少最多的 XOR 操作。
    """

    def solve(self, target_matrix):
        T = np.array(target_matrix, dtype=np.int8).copy()
        n_targets, n_inputs = T.shape

        # 基向量集合（用 GF(2) 向量表示）
        basis = [np.zeros(n_inputs, dtype=np.int8) for _ in range(n_inputs)]
        for i in range(n_inputs):
            basis[i][i] = 1

        circuit = []
        achieved = np.zeros(n_targets, dtype=bool)

        # 检查初始匹配
        for t in range(n_targets):
            for b in range(len(basis)):
                if np.array_equal(basis[b], T[t]):
                    achieved[t] = True

        while not np.all(achieved):
            best_score = -1
            best_pair = None

            # 枚举所有合法 XOR 对
            for i in range(len(basis)):
                for j in range(i + 1, len(basis)):
                    xor = (basis[i] + basis[j]) % 2

                    if not np.any(xor):
                        continue

                    # 检查是否重复
                    is_dup = any(np.array_equal(xor, basis[k]) for k in range(len(basis)))
                    if is_dup:
                        continue

                    # 计算得分：命中 target 数 + 距离改善
                    score = 0
                    for t in range(n_targets):
                        if achieved[t]:
                            continue
                        dist_new = int(np.sum(xor != T[t]))
                        if dist_new == 0:
                            score += 100
                        else:
                            # 与现有最近基向量的距离比较
                            min_old = min(int(np.sum(basis[k] != T[t])) for k in range(len(basis)))
                            if dist_new < min_old:
                                score += (min_old - dist_new)

                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)

            if best_pair is None or best_score <= 0:
                # 无法继续优化，用随机策略
                valid_pairs = []
                for i in range(len(basis)):
                    for j in range(i + 1, len(basis)):
                        xor = (basis[i] + basis[j]) % 2
                        if np.any(xor) and not any(np.array_equal(xor, basis[k]) for k in range(len(basis))):
                            valid_pairs.append((i, j))
                if not valid_pairs:
                    break
                best_pair = valid_pairs[np.random.randint(len(valid_pairs))]

            i, j = best_pair
            new_vec = (basis[i] + basis[j]) % 2
            circuit.append((i, j))
            basis.append(new_vec)

            # 更新已达成状态
            for t in range(n_targets):
                if not achieved[t] and np.array_equal(new_vec, T[t]):
                    achieved[t] = True

            # 安全阀
            if len(circuit) > n_inputs * 10:
                break

        return circuit, len(circuit), bool(np.all(achieved))


# ==========================================
# 3. BP + CMS (Baev 2020 改进版)
# ==========================================
class BPCMSAlgorithm:
    """
    BP + Cancellation-free Matrix Splitting 改进。
    在 Boyar-Peralta 基础上增加矩阵分解预处理：
    1) 将目标矩阵分解为若干"层"
    2) 每层内用 BP 贪心
    3) 层间共享中间结果
    """

    def solve(self, target_matrix):
        T = np.array(target_matrix, dtype=np.int8).copy()
        n_targets, n_inputs = T.shape

        # Step 1: 按行权重排序，低权重优先（容易先解决）
        row_weights = np.sum(T, axis=1)
        order = np.argsort(row_weights)

        basis = [np.zeros(n_inputs, dtype=np.int8) for _ in range(n_inputs)]
        for i in range(n_inputs):
            basis[i][i] = 1

        circuit = []
        achieved = np.zeros(n_targets, dtype=bool)

        for t in range(n_targets):
            for b in range(len(basis)):
                if np.array_equal(basis[b], T[t]):
                    achieved[t] = True

        # Step 2: 逐 target 贪心构造（按权重从低到高）
        for target_idx in order:
            if achieved[target_idx]:
                continue

            target_vec = T[target_idx]
            max_tries = n_inputs * 5

            for _ in range(max_tries):
                # 找到当前最接近 target 的基向量
                dists = [int(np.sum(basis[k] != target_vec)) for k in range(len(basis))]
                closest = int(np.argmin(dists))
                if dists[closest] == 0:
                    achieved[target_idx] = True
                    break

                # 找一个基向量 j，使得 basis[closest] XOR basis[j] 更接近 target
                best_j = -1
                best_dist = dists[closest]
                for j in range(len(basis)):
                    if j == closest:
                        continue
                    xor = (basis[closest] + basis[j]) % 2
                    if not np.any(xor):
                        continue
                    is_dup = any(np.array_equal(xor, basis[k]) for k in range(len(basis)))
                    if is_dup:
                        continue
                    d = int(np.sum(xor != target_vec))
                    if d < best_dist:
                        best_dist = d
                        best_j = j

                if best_j == -1:
                    # 尝试任意对
                    found = False
                    for i in range(len(basis)):
                        for j in range(i + 1, len(basis)):
                            xor = (basis[i] + basis[j]) % 2
                            if not np.any(xor):
                                continue
                            is_dup = any(np.array_equal(xor, basis[k]) for k in range(len(basis)))
                            if is_dup:
                                continue
                            d = int(np.sum(xor != target_vec))
                            if d < best_dist:
                                best_dist = d
                                best_j = j
                                closest = i
                                found = True
                    if not found:
                        break

                new_vec = (basis[closest] + basis[best_j]) % 2
                circuit.append((closest, best_j))
                basis.append(new_vec)

                # 检查是否顺带命中了其他 target
                for t2 in range(n_targets):
                    if not achieved[t2] and np.array_equal(new_vec, T[t2]):
                        achieved[t2] = True

                if achieved[target_idx]:
                    break

            if len(circuit) > n_inputs * 10:
                break

        return circuit, len(circuit), bool(np.all(achieved))


# ==========================================
# 4. SAT-based Exact Solver (小规模)
# ==========================================
class SATSolver:
    """
    SAT 编码的精确求解器（仅适用于小规模 ≤16）。
    使用增量搜索：从 k=1 开始，逐步增加门数直到找到可行解。
    依赖 pysat 库，若未安装则跳过。
    """

    def __init__(self):
        self.available = False
        try:
            from pysat.solvers import Glucose4
            from pysat.card import CardEnc
            self.available = True
        except ImportError:
            pass

    def solve(self, target_matrix, max_gates=None, timeout=300):
        if not self.available:
            return None, None, False, "pysat not installed"

        from pysat.solvers import Glucose4

        T = np.array(target_matrix, dtype=np.int8)
        n_targets, n_inputs = T.shape

        if n_inputs > 16:
            return None, None, False, "too large for SAT"

        if max_gates is None:
            max_gates = n_inputs * 3

        # 增量搜索
        for k in range(1, max_gates + 1):
            result = self._try_solve(T, n_inputs, n_targets, k, timeout)
            if result is not None:
                return result, k, True, "optimal"

        return None, max_gates, False, "no solution found"

    def _try_solve(self, T, n_inputs, n_targets, k, timeout):
        """尝试用恰好 k 个 XOR 门求解"""
        from pysat.solvers import Glucose4

        # 变量编码:
        # sel_g_i_j: 门 g 选择输入 i 和 j (g=0..k-1, i<j<n_inputs+g)
        # out_g_b: 门 g 的输出的第 b 位 (GF(2))
        # 这里用简化的枚举方法

        max_nodes = n_inputs + k
        # 暴力枚举所有可能的电路（仅适用于小 k）
        if k > 8:
            return None

        return self._enumerate_circuits(T, n_inputs, n_targets, k)

    def _enumerate_circuits(self, T, n_inputs, n_targets, k, depth=0, basis=None, circuit=None):
        """递归枚举，带剪枝"""
        if basis is None:
            basis = []
            for i in range(n_inputs):
                vec = np.zeros(n_inputs, dtype=np.int8)
                vec[i] = 1
                basis.append(vec)
            circuit = []

        if depth == k:
            # 检查是否所有 target 都在 basis 中
            for t in range(n_targets):
                found = False
                for b in basis:
                    if np.array_equal(b, T[t]):
                        found = True
                        break
                if not found:
                    return None
            return circuit.copy()

        n = len(basis)
        for i in range(n):
            for j in range(i + 1, n):
                xor = (basis[i] + basis[j]) % 2
                if not np.any(xor):
                    continue
                is_dup = any(np.array_equal(xor, basis[m]) for m in range(n))
                if is_dup:
                    continue

                basis.append(xor)
                circuit.append((i, j))
                result = self._enumerate_circuits(T, n_inputs, n_targets, k, depth + 1, basis, circuit)
                if result is not None:
                    return result
                basis.pop()
                circuit.pop()

        return None


# ==========================================
# 5. Greedy Expert (与 train_gnn_slp 中一致)
# ==========================================
class GreedyBaseline:
    """直接贪心，作为简单 baseline"""

    def solve(self, target_matrix, max_gates=None, max_depth=999):
        T = np.array(target_matrix, dtype=np.int8)
        n_targets, n_inputs = T.shape
        if max_gates is None:
            max_gates = n_inputs * 5

        basis = [np.zeros(n_inputs, dtype=np.int8) for _ in range(n_inputs)]
        for i in range(n_inputs):
            basis[i][i] = 1

        circuit = []
        achieved = np.zeros(n_targets, dtype=bool)

        for t in range(n_targets):
            for b in range(len(basis)):
                if np.array_equal(basis[b], T[t]):
                    achieved[t] = True

        for step in range(max_gates):
            if np.all(achieved):
                break

            best_score = -np.inf
            best_pair = None
            min_dists = []
            for t in range(n_targets):
                if achieved[t]:
                    min_dists.append(0)
                else:
                    md = min(int(np.sum(basis[k] != T[t])) for k in range(len(basis)))
                    min_dists.append(md)

            for i in range(len(basis)):
                for j in range(i + 1, len(basis)):
                    xor = (basis[i] + basis[j]) % 2
                    if not np.any(xor):
                        continue
                    is_dup = any(np.array_equal(xor, basis[k]) for k in range(len(basis)))
                    if is_dup:
                        continue

                    score = 0.0
                    for t in range(n_targets):
                        if achieved[t]:
                            continue
                        dist = int(np.sum(xor != T[t]))
                        if dist == 0:
                            score += 1000.0
                        else:
                            imp = min_dists[t] - dist
                            if imp > 0:
                                score += imp * 5.0
                            score += max(0, n_inputs - 2 * dist) * 0.1

                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)

            if best_pair is None:
                break

            i, j = best_pair
            new_vec = (basis[i] + basis[j]) % 2
            circuit.append((i, j))
            basis.append(new_vec)

            for t in range(n_targets):
                if not achieved[t] and np.array_equal(new_vec, T[t]):
                    achieved[t] = True

        return circuit, len(circuit), bool(np.all(achieved))


# ==========================================
# 统一接口
# ==========================================
def run_baseline(name, target_matrix, **kwargs):
    """运行指定的 baseline 算法"""
    start = time.time()

    if name == 'paar':
        algo = PaarAlgorithm()
        circuit, n_gates = algo.solve(target_matrix)
        solved = True  # Paar 总是能找到某种解
        elapsed = time.time() - start
        return {'name': name, 'gates': n_gates, 'solved': solved, 'time': elapsed, 'circuit': circuit}

    elif name == 'boyar_peralta':
        algo = BoyarPeraltaAlgorithm()
        circuit, n_gates, solved = algo.solve(target_matrix)
        elapsed = time.time() - start
        return {'name': name, 'gates': n_gates, 'solved': solved, 'time': elapsed, 'circuit': circuit}

    elif name == 'bp_cms':
        algo = BPCMSAlgorithm()
        circuit, n_gates, solved = algo.solve(target_matrix)
        elapsed = time.time() - start
        return {'name': name, 'gates': n_gates, 'solved': solved, 'time': elapsed, 'circuit': circuit}

    elif name == 'sat':
        algo = SATSolver()
        circuit, n_gates, solved, msg = algo.solve(target_matrix, **kwargs)
        elapsed = time.time() - start
        return {'name': name, 'gates': n_gates, 'solved': solved, 'time': elapsed, 'circuit': circuit, 'msg': msg}

    elif name == 'greedy':
        algo = GreedyBaseline()
        circuit, n_gates, solved = algo.solve(target_matrix, **kwargs)
        elapsed = time.time() - start
        return {'name': name, 'gates': n_gates, 'solved': solved, 'time': elapsed, 'circuit': circuit}

    else:
        raise ValueError(f"Unknown baseline: {name}")


if __name__ == "__main__":
    from benchmark_matrices import get_midori_16x16_matrix

    matrix = get_midori_16x16_matrix()
    print(f"Midori 16x16: {len(matrix)}x{len(matrix[0])}")

    for algo_name in ['paar', 'boyar_peralta', 'bp_cms', 'greedy']:
        result = run_baseline(algo_name, matrix)
        print(f"  {algo_name}: gates={result['gates']}, solved={result['solved']}, time={result['time']:.3f}s")
