"""
局部搜索电路优化：在已有电路基础上通过门删除、替换、合并来减少门数。
核心思想：Paar/BP 等贪心算法产生的电路往往包含冗余，局部搜索可以发现并消除。
"""
import numpy as np
import time
from copy import deepcopy


def verify_circuit(circuit, target_matrix, num_inputs):
    """验证电路是否正确计算所有目标向量"""
    T = np.array(target_matrix, dtype=np.int8)
    n_targets = len(T)

    basis = []
    for i in range(num_inputs):
        vec = np.zeros(num_inputs, dtype=np.int8)
        vec[i] = 1
        basis.append(vec)

    for u, v in circuit:
        if u >= len(basis) or v >= len(basis):
            return False
        new_vec = (basis[u] + basis[v]) % 2
        basis.append(new_vec.astype(np.int8))

    for t in range(n_targets):
        found = any(np.array_equal(b, T[t]) for b in basis)
        if not found:
            return False
    return True


def reconstruct_basis(circuit, num_inputs):
    """从电路重建所有中间向量"""
    basis = []
    for i in range(num_inputs):
        vec = np.zeros(num_inputs, dtype=np.int8)
        vec[i] = 1
        basis.append(vec)
    for u, v in circuit:
        new_vec = (basis[u] + basis[v]) % 2
        basis.append(new_vec.astype(np.int8))
    return basis


def find_target_nodes(basis, target_matrix, num_inputs):
    """找到每个目标向量在 basis 中最后出现的位置"""
    T = np.array(target_matrix, dtype=np.int8)
    target_nodes = {}
    for t in range(len(T)):
        for i in range(len(basis) - 1, -1, -1):
            if np.array_equal(basis[i], T[t]):
                target_nodes[t] = i
                break
    return target_nodes


def get_needed_gates(circuit, target_matrix, num_inputs):
    """反向可达性分析：找出所有必要的门"""
    basis = reconstruct_basis(circuit, num_inputs)
    target_nodes = find_target_nodes(basis, target_matrix, num_inputs)

    needed = set(target_nodes.values())
    for gate_idx in range(len(circuit) - 1, -1, -1):
        node_idx = num_inputs + gate_idx
        if node_idx in needed:
            u, v = circuit[gate_idx]
            needed.add(u)
            needed.add(v)
    return needed


def gate_removal_pass(circuit, target_matrix, num_inputs, time_limit=30):
    """
    迭代门删除：尝试移除每个门，若电路仍正确则接受。
    重复直到无法继续删除或超时。
    """
    T = np.array(target_matrix, dtype=np.int8)
    improved = True
    current = list(circuit)
    t0 = time.time()

    while improved:
        if time.time() - t0 > time_limit:
            break
        improved = False
        i = 0
        while i < len(current):
            if time.time() - t0 > time_limit:
                break
            # 尝试删除第 i 个门
            trial = current[:i] + current[i + 1:]
            # 重新索引：删除门 i 后，后续门的引用需要调整
            trial_reindexed = reindex_circuit(trial, num_inputs, skip_gate=i)
            if trial_reindexed is not None and verify_circuit(trial_reindexed, T, num_inputs):
                current = trial_reindexed
                improved = True
                # 不增加 i，因为新的第 i 个门需要重新检查
            else:
                i += 1

    return current


def reindex_circuit(gates_without_skip, num_inputs, skip_gate):
    """
    从原始电路中删除一个门后，重新映射索引。
    skip_gate: 被删除的门在原电路中的索引。
    gates_without_skip: 删除后的门列表（仍用原索引）。
    返回重新索引的电路，若引用无效则返回 None。
    """
    # 建立映射：原节点索引 → 新节点索引
    old_to_new = {}
    for i in range(num_inputs):
        old_to_new[i] = i

    new_idx = num_inputs
    removed_node = num_inputs + skip_gate

    for orig_gate_idx in range(len(gates_without_skip) + 1):
        old_node = num_inputs + (orig_gate_idx if orig_gate_idx < skip_gate else orig_gate_idx + 1)
        if old_node == removed_node:
            continue
        if orig_gate_idx < skip_gate:
            old_to_new[old_node] = old_node  # 前面的门不变
        else:
            old_to_new[num_inputs + orig_gate_idx + 1] = new_idx
            new_idx += 1

    # 重建电路
    # 更简单的方法：直接重建
    return _rebuild_without_gate(gates_without_skip, num_inputs, skip_gate)


def _rebuild_without_gate(original_circuit, num_inputs, skip_gate):
    """从原始电路中删除第 skip_gate 个门，重新构建有效电路"""
    removed_node = num_inputs + skip_gate

    # 建立旧索引到新索引的映射
    old_to_new = {}
    for i in range(num_inputs):
        old_to_new[i] = i

    new_circuit = []
    new_node_idx = num_inputs

    for gate_idx, (u, v) in enumerate(original_circuit):
        old_node = num_inputs + gate_idx
        if gate_idx == skip_gate:
            # 跳过此门，不分配新索引
            continue

        # 检查 u, v 是否引用了被删除的门
        if u == removed_node or v == removed_node:
            return None  # 依赖被删除的门，无法删除
        if u not in old_to_new or v not in old_to_new:
            return None

        new_u = old_to_new[u]
        new_v = old_to_new[v]
        new_circuit.append((new_u, new_v))
        old_to_new[old_node] = new_node_idx
        new_node_idx += 1

    return new_circuit


def simplify_circuit(circuit, target_matrix, num_inputs):
    """
    基于反向可达性的死门消除（比 gate_removal_pass 更快但不那么彻底）。
    """
    T = np.array(target_matrix, dtype=np.int8)
    basis = reconstruct_basis(circuit, num_inputs)
    target_nodes = find_target_nodes(basis, T, num_inputs)

    if len(target_nodes) < len(T):
        return circuit  # 电路不完整

    needed = set(target_nodes.values())
    for gate_idx in range(len(circuit) - 1, -1, -1):
        node_idx = num_inputs + gate_idx
        if node_idx in needed:
            u, v = circuit[gate_idx]
            needed.add(u)
            needed.add(v)

    # 重建：只保留需要的门
    old_to_new = {i: i for i in range(num_inputs)}
    new_circuit = []
    new_idx = num_inputs

    for gate_idx, (u, v) in enumerate(circuit):
        node_idx = num_inputs + gate_idx
        if node_idx in needed:
            new_u = old_to_new.get(u)
            new_v = old_to_new.get(v)
            if new_u is None or new_v is None:
                return circuit
            new_circuit.append((new_u, new_v))
            old_to_new[node_idx] = new_idx
            new_idx += 1

    if verify_circuit(new_circuit, T, num_inputs):
        return new_circuit
    return circuit


def gate_substitution_pass(circuit, target_matrix, num_inputs, max_tries_per_gate=50, time_limit=30):
    """
    门替换：对每个门 (u,v)，尝试用其他合法对 (u',v') 替换。
    如果替换后电路仍正确且总门数不增加，接受替换。
    """
    T = np.array(target_matrix, dtype=np.int8)
    current = list(circuit)
    improved = True
    t0 = time.time()

    while improved:
        if time.time() - t0 > time_limit:
            break
        improved = False
        for gate_idx in range(len(current)):
            if time.time() - t0 > time_limit:
                break
            u_orig, v_orig = current[gate_idx]
            # 在此门位置可用的节点数 = num_inputs + gate_idx
            n_available = num_inputs + gate_idx
            tried = 0

            # 尝试所有可能的替换对
            for u_new in range(n_available):
                if improved:
                    break
                for v_new in range(u_new + 1, n_available):
                    if (u_new, v_new) == (u_orig, v_orig) or (v_new, u_new) == (u_orig, v_orig):
                        continue
                    tried += 1
                    if tried > max_tries_per_gate:
                        break

                    trial = list(current)
                    trial[gate_idx] = (u_new, v_new)
                    if verify_circuit(trial, T, num_inputs):
                        # 尝试简化替换后的电路
                        simplified = simplify_circuit(trial, T, num_inputs)
                        if len(simplified) < len(current):
                            current = simplified
                            improved = True
                            break
                if tried > max_tries_per_gate:
                    break

    return current


def two_for_one_pass(circuit, target_matrix, num_inputs, time_limit=30):
    """
    两门合一：对每对相邻门 (g_i, g_{i+1})，检查是否存在单门替代。
    如果 g_i 产生 a^b, g_{i+1} 产生 (a^b)^c = a^b^c，
    但 a^c 已经在基中，则可以用 (a^c)^b 单门替代。
    """
    T = np.array(target_matrix, dtype=np.int8)
    current = list(circuit)
    improved = True
    t0 = time.time()

    while improved:
        if time.time() - t0 > time_limit:
            break
        improved = False
        i = 0
        while i < len(current) - 1:
            if time.time() - t0 > time_limit:
                break
            # 尝试用一个门替代第 i 和 i+1 个门
            basis_at_i = reconstruct_basis(current[:i], num_inputs)
            n_at_i = len(basis_at_i)

            # 第 i 个门产生的向量
            u1, v1 = current[i]
            if u1 >= n_at_i or v1 >= n_at_i:
                i += 1
                continue
            vec_i = (basis_at_i[u1] + basis_at_i[v1]) % 2

            # 第 i+1 个门产生的向量
            basis_at_i1 = list(basis_at_i) + [vec_i]
            u2, v2 = current[i + 1]
            if u2 >= len(basis_at_i1) or v2 >= len(basis_at_i1):
                i += 1
                continue
            vec_i1 = (basis_at_i1[u2] + basis_at_i1[v2]) % 2

            # 目标：用一个门从 basis_at_i 出发产生 vec_i1
            found_replacement = False
            for a in range(n_at_i):
                for b in range(a + 1, n_at_i):
                    candidate = (basis_at_i[a] + basis_at_i[b]) % 2
                    if np.array_equal(candidate, vec_i1):
                        # 可以用单门 (a, b) 替代两门
                        new_circuit = current[:i] + [(a, b)]
                        # 后续门需要重新索引
                        remaining = _reindex_remaining(
                            current[i + 2:], num_inputs, i, removed_count=1
                        )
                        if remaining is not None:
                            new_circuit += remaining
                            if verify_circuit(new_circuit, T, num_inputs):
                                current = new_circuit
                                found_replacement = True
                                improved = True
                                break
                if found_replacement:
                    break

            if not found_replacement:
                i += 1

    return current


def _reindex_remaining(remaining_gates, num_inputs, start_gate_idx, removed_count):
    """重新索引删除 removed_count 个门后的剩余门"""
    old_to_new = {}
    # 原始门 0..start_gate_idx-1 → 不变
    for i in range(num_inputs + start_gate_idx):
        old_to_new[i] = i

    # 被替代的两个门（索引 start_gate_idx 和 start_gate_idx+1）
    # 新电路在位置 start_gate_idx 有一个新门
    # 所以原始节点 num_inputs + start_gate_idx → 删除
    # 原始节点 num_inputs + start_gate_idx + 1 → 删除
    # 新节点 num_inputs + start_gate_idx → 新的替代门

    # 新替代门的节点
    new_replacement_node = num_inputs + start_gate_idx

    # 旧的两个节点映射到新的一个
    old_node_0 = num_inputs + start_gate_idx
    old_node_1 = num_inputs + start_gate_idx + 1

    # 后续门的旧节点需要偏移 -1
    new_circuit = []
    next_new = new_replacement_node + 1

    for gate_idx, (u, v) in enumerate(remaining_gates):
        old_gate_idx = start_gate_idx + 2 + gate_idx
        old_node = num_inputs + old_gate_idx

        # 映射 u
        if u == old_node_0 or u == old_node_1:
            # 引用了被合并的两个门之一
            # 如果引用 old_node_1（两门合一后的最终结果），映射到新替代门
            if u == old_node_1:
                new_u = new_replacement_node
            else:
                # 引用了中间结果 old_node_0，但它已被删除
                return None
        elif u in old_to_new:
            new_u = old_to_new[u]
        else:
            return None

        # 映射 v
        if v == old_node_0 or v == old_node_1:
            if v == old_node_1:
                new_v = new_replacement_node
            else:
                return None
        elif v in old_to_new:
            new_v = old_to_new[v]
        else:
            return None

        new_circuit.append((new_u, new_v))
        old_to_new[old_node] = next_new
        next_new += 1

    return new_circuit


def full_local_search(circuit, target_matrix, num_inputs, time_limit=60):
    """
    完整局部搜索：依次执行死门消除 → 门删除 → 两门合一 → 门替换。
    重复直到无改善或超时。
    """
    T = np.array(target_matrix, dtype=np.int8)
    best = list(circuit)
    best_gates = len(best)
    t0 = time.time()

    for iteration in range(10):
        if time.time() - t0 > time_limit:
            break

        # 1. 死门消除（快）
        current = simplify_circuit(best, T, num_inputs)
        if len(current) < best_gates:
            best = current
            best_gates = len(best)

        # 2. 逐门删除尝试（中等速度）
        remaining = max(time_limit - (time.time() - t0), 1)
        current = gate_removal_pass(best, T, num_inputs, time_limit=min(remaining * 0.3, 15))
        if len(current) < best_gates:
            best = current
            best_gates = len(best)

        # 3. 两门合一（较慢）
        remaining = max(time_limit - (time.time() - t0), 1)
        if remaining > 2:
            current = two_for_one_pass(best, T, num_inputs, time_limit=min(remaining * 0.4, 15))
            if len(current) < best_gates:
                best = current
                best_gates = len(best)

        # 4. 门替换（最慢，限制尝试次数）
        remaining = max(time_limit - (time.time() - t0), 1)
        if remaining > 2:
            current = gate_substitution_pass(best, T, num_inputs, max_tries_per_gate=30, time_limit=min(remaining * 0.5, 15))
            if len(current) < best_gates:
                best = current
                best_gates = len(best)

        if len(best) == best_gates and iteration > 0:
            break  # 无改善

    return best


def randomized_paar(target_matrix, rng=None):
    """
    随机化 Paar 算法：当多个列对具有相同共现频率时，随机选择一对。
    返回 (circuit, n_gates)。
    """
    if rng is None:
        rng = np.random.RandomState()

    T = np.array(target_matrix, dtype=np.int8).copy()
    n_targets, n_inputs = T.shape
    D = T.copy()

    basis_vecs = []
    for i in range(n_inputs):
        vec = np.zeros(n_inputs, dtype=np.int8)
        vec[i] = 1
        basis_vecs.append(vec)

    circuit = []

    for _ in range(n_inputs * 5):
        row_sums = np.sum(D, axis=1)
        if np.all(row_sums <= 1):
            break

        n_cols = D.shape[1]
        best_count = -1
        tied_pairs = []

        # 向量化：对每列 i，与后续列 j 计算共现数
        for i in range(n_cols):
            col_i = D[:, i]
            if not np.any(col_i):
                continue
            # 批量计算 i 与所有 j>i 的共现数
            counts = np.sum(col_i[:, None] & D[:, i + 1:], axis=0)
            for jj in range(len(counts)):
                c = int(counts[jj])
                if c > best_count:
                    best_count = c
                    tied_pairs = [(i, i + 1 + jj)]
                elif c == best_count and c > 0:
                    tied_pairs.append((i, i + 1 + jj))

        if best_count <= 0:
            break

        # 随机选择一对（核心随机化）
        ci, cj = tied_pairs[rng.randint(len(tied_pairs))]

        new_vec = (np.array(basis_vecs[ci]) + np.array(basis_vecs[cj])) % 2
        circuit.append((ci, cj))
        basis_vecs.append(new_vec)

        new_col = np.zeros(n_targets, dtype=np.int8)
        for t in range(n_targets):
            if D[t, ci] == 1 and D[t, cj] == 1:
                D[t, ci] = 0
                D[t, cj] = 0
                new_col[t] = 1
        D = np.hstack([D, new_col.reshape(-1, 1)])

    return circuit, len(circuit)


def temperature_paar(target_matrix, temperature=0.5, rng=None):
    """
    温度采样 Paar：不总是选最优对，而是按 softmax(score/temp) 概率采样。
    temperature=0 等价于标准 Paar；temperature 越高越随机。
    这能跳出 Paar 的确定性路径，探索不同电路拓扑。
    """
    if rng is None:
        rng = np.random.RandomState()

    T = np.array(target_matrix, dtype=np.int8).copy()
    n_targets, n_inputs = T.shape
    D = T.copy()

    basis_vecs = []
    for i in range(n_inputs):
        vec = np.zeros(n_inputs, dtype=np.int8)
        vec[i] = 1
        basis_vecs.append(vec)

    circuit = []

    for _ in range(n_inputs * 5):
        row_sums = np.sum(D, axis=1)
        if np.all(row_sums <= 1):
            break

        n_cols = D.shape[1]
        pairs = []
        scores = []

        for i in range(n_cols):
            col_i = D[:, i]
            if not np.any(col_i):
                continue
            counts = np.sum(col_i[:, None] & D[:, i + 1:], axis=0)
            for jj in range(len(counts)):
                c = int(counts[jj])
                if c > 0:
                    pairs.append((i, i + 1 + jj))
                    scores.append(c)

        if not pairs:
            break

        scores = np.array(scores, dtype=np.float64)
        if temperature > 0:
            # Softmax 采样
            log_scores = scores / temperature
            log_scores -= log_scores.max()
            probs = np.exp(log_scores)
            probs /= probs.sum()
            chosen = rng.choice(len(pairs), p=probs)
        else:
            chosen = np.argmax(scores)

        ci, cj = pairs[chosen]
        new_vec = (np.array(basis_vecs[ci]) + np.array(basis_vecs[cj])) % 2
        circuit.append((ci, cj))
        basis_vecs.append(new_vec)

        new_col = np.zeros(n_targets, dtype=np.int8)
        for t in range(n_targets):
            if D[t, ci] == 1 and D[t, cj] == 1:
                D[t, ci] = 0
                D[t, cj] = 0
                new_col[t] = 1
        D = np.hstack([D, new_col.reshape(-1, 1)])

    return circuit, len(circuit)


def hybrid_construction(target_matrix, strategy='bp_then_paar', rng=None):
    """
    混合构造策略：结合不同算法的优势。

    策略:
    - 'bp_then_paar': 前半用 BP 距离引导，后半用 Paar 共现频率
    - 'alternating': 交替使用 BP 和 Paar 评分
    - 'row_priority': 按不同行顺序优先构造（类 BP+CMS）
    """
    if rng is None:
        rng = np.random.RandomState()

    T = np.array(target_matrix, dtype=np.int8).copy()
    n_targets, n_inputs = T.shape

    basis = [np.zeros(n_inputs, dtype=np.int8) for _ in range(n_inputs)]
    for i in range(n_inputs):
        basis[i][i] = 1

    circuit = []
    achieved = np.zeros(n_targets, dtype=bool)

    for t in range(n_targets):
        for b in range(len(basis)):
            if np.array_equal(basis[b], T[t]):
                achieved[t] = True

    if strategy == 'row_priority':
        # 随机排列目标行的处理顺序
        order = rng.permutation(n_targets)
        for target_idx in order:
            if achieved[target_idx]:
                continue
            target_vec = T[target_idx]
            for _ in range(n_inputs * 3):
                if achieved[target_idx]:
                    break
                # 找最接近 target 的对
                best_dist = n_inputs + 1
                best_pair = None
                n = len(basis)
                # 优先找能直接命中或大幅缩短距离的对
                for i in range(n):
                    for j in range(i + 1, n):
                        xor = (basis[i] + basis[j]) % 2
                        if not np.any(xor):
                            continue
                        if any(np.array_equal(xor, basis[k]) for k in range(n)):
                            continue
                        d = int(np.sum(xor != target_vec))
                        # 加分：顺带接近其他未完成目标
                        bonus = 0
                        for t2 in range(n_targets):
                            if not achieved[t2] and t2 != target_idx:
                                d2 = int(np.sum(xor != T[t2]))
                                if d2 == 0:
                                    bonus -= 50  # 大奖
                                else:
                                    old_d2 = min(int(np.sum(basis[k] != T[t2])) for k in range(n))
                                    if d2 < old_d2:
                                        bonus -= (old_d2 - d2)
                        score = d + bonus * 0.1
                        if score < best_dist:
                            best_dist = score
                            best_pair = (i, j)
                if best_pair is None:
                    break
                i, j = best_pair
                new_vec = (basis[i] + basis[j]) % 2
                circuit.append((i, j))
                basis.append(new_vec)
                for t2 in range(n_targets):
                    if not achieved[t2] and np.array_equal(new_vec, T[t2]):
                        achieved[t2] = True
            if len(circuit) > n_inputs * 10:
                break

    elif strategy == 'alternating':
        step = 0
        while not np.all(achieved) and len(circuit) < n_inputs * 10:
            use_paar = (step % 3 == 0)  # 每3步用一次Paar风格
            step += 1

            best_score = -1
            best_pair = None
            n = len(basis)

            for i in range(n):
                for j in range(i + 1, n):
                    xor = (basis[i] + basis[j]) % 2
                    if not np.any(xor):
                        continue
                    if any(np.array_equal(xor, basis[k]) for k in range(n)):
                        continue

                    if use_paar:
                        # Paar 风格：计算多少目标能利用这个共同子表达式
                        score = 0
                        for t in range(n_targets):
                            if not achieved[t]:
                                # 检查 xor 是否是 target 的"子集"（作为子表达式）
                                overlap = int(np.sum(xor & T[t]))
                                score += overlap
                    else:
                        # BP 风格：距离改善
                        score = 0
                        for t in range(n_targets):
                            if achieved[t]:
                                continue
                            dist_new = int(np.sum(xor != T[t]))
                            if dist_new == 0:
                                score += 100
                            else:
                                min_old = min(int(np.sum(basis[k] != T[t])) for k in range(n))
                                if dist_new < min_old:
                                    score += (min_old - dist_new)

                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)

            if best_pair is None or best_score <= 0:
                break

            i, j = best_pair
            new_vec = (basis[i] + basis[j]) % 2
            circuit.append((i, j))
            basis.append(new_vec)
            for t in range(n_targets):
                if not achieved[t] and np.array_equal(new_vec, T[t]):
                    achieved[t] = True

    else:  # bp_then_paar
        # 前60%步用BP，后40%用Paar
        max_steps = n_inputs * 5
        bp_steps = int(max_steps * 0.6)

        for step_idx in range(max_steps):
            if np.all(achieved):
                break

            use_bp = step_idx < bp_steps
            best_score = -1
            best_pair = None
            n = len(basis)

            for i in range(n):
                for j in range(i + 1, n):
                    xor = (basis[i] + basis[j]) % 2
                    if not np.any(xor):
                        continue
                    if any(np.array_equal(xor, basis[k]) for k in range(n)):
                        continue

                    score = 0
                    for t in range(n_targets):
                        if achieved[t]:
                            continue
                        dist_new = int(np.sum(xor != T[t]))
                        if dist_new == 0:
                            score += 100
                        elif use_bp:
                            min_old = min(int(np.sum(basis[k] != T[t])) for k in range(n))
                            if dist_new < min_old:
                                score += (min_old - dist_new)
                        else:
                            # Paar: 共现
                            score += int(np.sum(xor & T[t])) * 0.1

                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)

            if best_pair is None or best_score <= 0:
                break
            i, j = best_pair
            new_vec = (basis[i] + basis[j]) % 2
            circuit.append((i, j))
            basis.append(new_vec)
            for t in range(n_targets):
                if not achieved[t] and np.array_equal(new_vec, T[t]):
                    achieved[t] = True

    return circuit, len(circuit), bool(np.all(achieved))


def randomized_bp(target_matrix, rng=None):
    """
    随机化 Boyar-Peralta：当多个对具有相同得分时，随机选择。
    返回 (circuit, n_gates, solved)。
    """
    if rng is None:
        rng = np.random.RandomState()

    T = np.array(target_matrix, dtype=np.int8).copy()
    n_targets, n_inputs = T.shape

    basis = [np.zeros(n_inputs, dtype=np.int8) for _ in range(n_inputs)]
    for i in range(n_inputs):
        basis[i][i] = 1

    circuit = []
    achieved = np.zeros(n_targets, dtype=bool)

    for t in range(n_targets):
        for b in range(len(basis)):
            if np.array_equal(basis[b], T[t]):
                achieved[t] = True

    while not np.all(achieved):
        best_score = -1
        tied_pairs = []

        for i in range(len(basis)):
            for j in range(i + 1, len(basis)):
                xor = (basis[i] + basis[j]) % 2
                if not np.any(xor):
                    continue
                is_dup = any(np.array_equal(xor, basis[k]) for k in range(len(basis)))
                if is_dup:
                    continue

                score = 0
                for t in range(n_targets):
                    if achieved[t]:
                        continue
                    dist_new = int(np.sum(xor != T[t]))
                    if dist_new == 0:
                        score += 100
                    else:
                        min_old = min(int(np.sum(basis[k] != T[t])) for k in range(len(basis)))
                        if dist_new < min_old:
                            score += (min_old - dist_new)

                if score > best_score:
                    best_score = score
                    tied_pairs = [(i, j)]
                elif score == best_score and score > 0:
                    tied_pairs.append((i, j))

        if not tied_pairs or best_score <= 0:
            # 随机选择
            valid_pairs = []
            for i in range(len(basis)):
                for j in range(i + 1, len(basis)):
                    xor = (basis[i] + basis[j]) % 2
                    if np.any(xor) and not any(np.array_equal(xor, basis[k]) for k in range(len(basis))):
                        valid_pairs.append((i, j))
            if not valid_pairs:
                break
            tied_pairs = valid_pairs

        i, j = tied_pairs[rng.randint(len(tied_pairs))]
        new_vec = (basis[i] + basis[j]) % 2
        circuit.append((i, j))
        basis.append(new_vec)

        for t in range(n_targets):
            if not achieved[t] and np.array_equal(new_vec, T[t]):
                achieved[t] = True

        if len(circuit) > n_inputs * 10:
            break

    return circuit, len(circuit), bool(np.all(achieved))


def multi_start_search(target_matrix, n_paar=5000, n_bp=2000, time_limit=600,
                       local_search_top=50, local_search_time=30, verbose=True):
    """
    多起点搜索：大量随机化 Paar/BP + 对最佳结果应用局部搜索。
    返回 (best_circuit, best_gates, stats)。
    """
    T = np.array(target_matrix, dtype=np.int8)
    n_targets, n_inputs = T.shape
    t0 = time.time()

    all_results = []  # (n_gates, circuit, method)
    gate_counts = {'paar': [], 'bp': []}

    # Phase 1: 随机化 Paar
    if verbose:
        print(f"  Multi-start: running {n_paar} randomized Paar trials...")
    for trial in range(n_paar):
        if time.time() - t0 > time_limit * 0.4:
            if verbose:
                print(f"  Paar: stopped at trial {trial} (time limit)")
            break
        rng = np.random.RandomState(trial)
        circuit, n_gates = randomized_paar(T, rng)
        gate_counts['paar'].append(n_gates)
        all_results.append((n_gates, circuit, 'paar'))

    # Phase 1b: 温度采样 Paar（更强随机化，可跳出确定性路径）
    gate_counts['temp_paar'] = []
    if verbose:
        paar_min = min(gate_counts['paar']) if gate_counts['paar'] else 'N/A'
        print(f"  Paar best: {paar_min}. Running temperature Paar trials...")
    for trial in range(n_paar):
        if time.time() - t0 > time_limit * 0.35:
            break
        rng = np.random.RandomState(50000 + trial)
        temp = rng.uniform(0.3, 2.0)
        circuit, n_gates = temperature_paar(T, temperature=temp, rng=rng)
        gate_counts['temp_paar'].append(n_gates)
        all_results.append((n_gates, circuit, 'temp_paar'))
    if verbose and gate_counts['temp_paar']:
        print(f"  Temp Paar: min={min(gate_counts['temp_paar'])}, mean={np.mean(gate_counts['temp_paar']):.1f}")

    # Phase 1c: 混合构造策略
    gate_counts['hybrid'] = []
    if verbose:
        print(f"  Running hybrid construction trials...")
    strategies = ['row_priority', 'alternating', 'bp_then_paar']
    for trial in range(min(n_paar // 2, 500)):
        if time.time() - t0 > time_limit * 0.5:
            break
        rng = np.random.RandomState(80000 + trial)
        strat = strategies[trial % len(strategies)]
        circuit, n_gates, solved = hybrid_construction(T, strategy=strat, rng=rng)
        if solved:
            gate_counts['hybrid'].append(n_gates)
            all_results.append((n_gates, circuit, f'hybrid_{strat}'))
    if verbose and gate_counts['hybrid']:
        print(f"  Hybrid: min={min(gate_counts['hybrid'])}, mean={np.mean(gate_counts['hybrid']):.1f}")

    # Phase 2: 随机化 BP
    if verbose:
        print(f"  Running {n_bp} randomized BP trials...")
    for trial in range(n_bp):
        if time.time() - t0 > time_limit * 0.7:
            if verbose:
                print(f"  BP: stopped at trial {trial} (time limit)")
            break
        rng = np.random.RandomState(10000 + trial)
        circuit, n_gates, solved = randomized_bp(T, rng)
        if solved:
            gate_counts['bp'].append(n_gates)
            all_results.append((n_gates, circuit, 'bp'))

    # 排序，取最佳
    all_results.sort(key=lambda x: x[0])

    if verbose:
        bp_min = min(gate_counts['bp']) if gate_counts['bp'] else 'N/A'
        print(f"  BP best: {bp_min} gates. Overall best before local search: {all_results[0][0]}")

    # Phase 3: 对 top-K 结果应用局部搜索
    if verbose:
        print(f"  Applying local search to top {local_search_top} circuits...")

    best_gates = all_results[0][0]
    best_circuit = all_results[0][1]
    unique_circuits = set()

    for rank, (n_gates, circuit, method) in enumerate(all_results[:local_search_top]):
        if time.time() - t0 > time_limit:
            break

        # 去重（基于电路的门数+前几个门）
        sig = (n_gates, tuple(circuit[:5]) if len(circuit) >= 5 else tuple(circuit))
        if sig in unique_circuits:
            continue
        unique_circuits.add(sig)

        optimized = full_local_search(circuit, T, n_inputs, time_limit=local_search_time)
        if len(optimized) < best_gates:
            if verify_circuit(optimized, T, n_inputs):
                best_gates = len(optimized)
                best_circuit = optimized
                if verbose:
                    print(f"  ★ Local search improved: {n_gates} → {best_gates} gates (from {method} #{rank})")

    elapsed = time.time() - t0
    stats = {
        'paar_trials': len(gate_counts['paar']),
        'bp_trials': len(gate_counts['bp']),
        'paar_min': min(gate_counts['paar']) if gate_counts['paar'] else None,
        'paar_mean': np.mean(gate_counts['paar']) if gate_counts['paar'] else None,
        'bp_min': min(gate_counts['bp']) if gate_counts['bp'] else None,
        'bp_mean': np.mean(gate_counts['bp']) if gate_counts['bp'] else None,
        'final_best': best_gates,
        'elapsed': elapsed,
    }

    if verbose:
        print(f"  Multi-start done: best={best_gates} gates, time={elapsed:.1f}s")
        if gate_counts['paar']:
            hist, edges = np.histogram(gate_counts['paar'], bins=10)
            print(f"  Paar distribution: min={stats['paar_min']}, mean={stats['paar_mean']:.1f}")
        if gate_counts['bp']:
            print(f"  BP distribution: min={stats['bp_min']}, mean={stats['bp_mean']:.1f}")

    return best_circuit, best_gates, stats


def iterated_local_search(target_matrix, num_inputs=None, n_restarts=200,
                          time_limit=300, perturbation_strength=5, verbose=True):
    """
    迭代局部搜索 (ILS)：
    1. 随机化 Paar 生成初始电路
    2. 应用局部搜索
    3. 扰动（随机删除几个门，用贪心补全）
    4. 再次局部搜索
    5. 如果改善则接受，否则以概率接受
    """
    T = np.array(target_matrix, dtype=np.int8)
    n_targets = len(T)
    if num_inputs is None:
        num_inputs = T.shape[1]

    t0 = time.time()
    global_best_gates = float('inf')
    global_best_circuit = None

    for restart in range(n_restarts):
        if time.time() - t0 > time_limit:
            break

        # 生成初始解
        rng = np.random.RandomState(restart * 7 + 42)
        circuit, n_gates = randomized_paar(T, rng)

        # 局部搜索
        circuit = full_local_search(circuit, T, num_inputs, time_limit=15)
        n_gates = len(circuit)

        if n_gates < global_best_gates:
            global_best_gates = n_gates
            global_best_circuit = list(circuit)
            if verbose:
                print(f"  ILS restart {restart}: new best = {global_best_gates} gates")

        # 扰动 + 局部搜索 循环
        current = list(circuit)
        current_gates = n_gates

        for perturb_iter in range(20):
            if time.time() - t0 > time_limit:
                break

            # 扰动：随机删除几个门
            n_remove = min(perturbation_strength, len(current) // 4)
            if n_remove < 1:
                break

            perturbed = list(current)
            remove_indices = sorted(rng.choice(len(perturbed), size=n_remove, replace=False), reverse=True)

            for idx in remove_indices:
                result = _rebuild_without_gate(perturbed, num_inputs, idx)
                if result is not None:
                    perturbed = result

            # 如果扰动后电路不完整，用贪心补全
            if not verify_circuit(perturbed, T, num_inputs):
                perturbed = _greedy_complete(perturbed, T, num_inputs)
                if perturbed is None:
                    continue

            # 局部搜索
            perturbed = full_local_search(perturbed, T, num_inputs, time_limit=10)
            p_gates = len(perturbed)

            # 接受准则
            if p_gates < current_gates:
                current = perturbed
                current_gates = p_gates
                if p_gates < global_best_gates:
                    global_best_gates = p_gates
                    global_best_circuit = list(perturbed)
                    if verbose:
                        print(f"  ILS restart {restart} perturb {perturb_iter}: "
                              f"new best = {global_best_gates} gates")
            elif p_gates <= current_gates + 2:
                # 允许小幅恶化以逃离局部最优
                if rng.random() < 0.3:
                    current = perturbed
                    current_gates = p_gates

    if verbose:
        elapsed = time.time() - t0
        print(f"  ILS done: best={global_best_gates} gates, time={elapsed:.1f}s")

    return global_best_circuit, global_best_gates


def _greedy_complete(partial_circuit, target_matrix, num_inputs):
    """给定部分电路，用贪心策略补全所有目标"""
    T = np.array(target_matrix, dtype=np.int8)
    n_targets = len(T)

    basis = reconstruct_basis(partial_circuit, num_inputs)
    achieved = np.zeros(n_targets, dtype=bool)
    for t in range(n_targets):
        for b in basis:
            if np.array_equal(b, T[t]):
                achieved[t] = True
                break

    if np.all(achieved):
        return partial_circuit

    circuit = list(partial_circuit)
    for _ in range(num_inputs * 3):
        if np.all(achieved):
            break

        best_score = -1
        best_pair = None

        min_dists = []
        for t in range(n_targets):
            if achieved[t]:
                min_dists.append(0)
            else:
                md = min(int(np.sum(b != T[t])) for b in basis)
                min_dists.append(md)

        n = len(basis)
        for i in range(n):
            for j in range(i + 1, n):
                xor = (basis[i] + basis[j]) % 2
                if not np.any(xor):
                    continue
                if any(np.array_equal(xor, basis[k]) for k in range(n)):
                    continue

                score = 0
                for t in range(n_targets):
                    if achieved[t]:
                        continue
                    dist = int(np.sum(xor != T[t]))
                    if dist == 0:
                        score += 1000
                    elif dist < min_dists[t]:
                        score += (min_dists[t] - dist) * 5

                if score > best_score:
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            return None

        i, j = best_pair
        new_vec = (basis[i] + basis[j]) % 2
        circuit.append((i, j))
        basis.append(new_vec)

        for t in range(n_targets):
            if not achieved[t] and np.array_equal(new_vec, T[t]):
                achieved[t] = True

    if np.all(achieved):
        return circuit
    return None


def deep_gate_substitution(circuit, target_matrix, num_inputs, time_limit=120):
    """
    深度门替换：对每个门尝试所有合法替换对（不限 max_tries），
    如果替换后经简化能减少门数则接受。比 gate_substitution_pass 更彻底。
    """
    T = np.array(target_matrix, dtype=np.int8)
    current = list(circuit)
    best_gates = len(current)
    t0 = time.time()
    improved = True

    while improved:
        improved = False
        if time.time() - t0 > time_limit:
            break
        for gate_idx in range(len(current)):
            if time.time() - t0 > time_limit:
                break
            u_orig, v_orig = current[gate_idx]
            n_available = num_inputs + gate_idx

            for u_new in range(n_available):
                if improved:
                    break
                for v_new in range(u_new + 1, n_available):
                    if (u_new, v_new) == (u_orig, v_orig) or (v_new, u_new) == (u_orig, v_orig):
                        continue
                    trial = list(current)
                    trial[gate_idx] = (u_new, v_new)
                    if verify_circuit(trial, T, num_inputs):
                        simplified = simplify_circuit(trial, T, num_inputs)
                        if len(simplified) < best_gates:
                            current = simplified
                            best_gates = len(simplified)
                            improved = True
                            break

    return current


def multi_gate_swap(circuit, target_matrix, num_inputs, window=3, time_limit=300):
    """
    多门窗口替换：用 1 或 2 个门替代连续 window 个门。
    这能发现 two_for_one_pass 无法找到的非相邻多门合并。
    """
    T = np.array(target_matrix, dtype=np.int8)
    current = list(circuit)
    best_gates = len(current)
    t0 = time.time()
    improved = True

    while improved:
        improved = False
        if time.time() - t0 > time_limit:
            break

        for w in range(window, 1, -1):  # 先试大窗口
            if improved:
                break
            for start in range(len(current) - w + 1):
                if time.time() - t0 > time_limit:
                    break

                # 尝试删除 [start:start+w] 中的一个或多个门
                for skip_count in range(1, w):
                    if improved:
                        break
                    # 尝试所有 skip 组合太多，只尝试单门删除
                    for skip_pos in range(start, start + w):
                        trial = _rebuild_without_gate(current, num_inputs, skip_pos)
                        if trial is not None and verify_circuit(trial, T, num_inputs):
                            simplified = simplify_circuit(trial, T, num_inputs)
                            if len(simplified) < best_gates:
                                current = simplified
                                best_gates = len(simplified)
                                improved = True
                                break

    return current


def exhaustive_local_search(circuit, target_matrix, num_inputs, time_limit=300,
                            verbose=False):
    """
    穷举局部搜索：结合所有优化手段，更大搜索半径。
    专为 AES 等难实例设计。
    """
    T = np.array(target_matrix, dtype=np.int8)
    best = list(circuit)
    best_gates = len(best)
    t0 = time.time()

    for iteration in range(20):
        if time.time() - t0 > time_limit:
            break

        old_gates = best_gates

        # 1. 死门消除
        current = simplify_circuit(best, T, num_inputs)
        if len(current) < best_gates:
            best, best_gates = current, len(current)

        # 2. 门删除
        remaining = max(time_limit - (time.time() - t0), 1)
        current = gate_removal_pass(best, T, num_inputs, time_limit=min(remaining * 0.2, 20))
        if len(current) < best_gates:
            best, best_gates = current, len(current)

        # 3. 两门合一
        remaining = max(time_limit - (time.time() - t0), 1)
        if remaining > 5:
            current = two_for_one_pass(best, T, num_inputs, time_limit=min(remaining * 0.3, 20))
            if len(current) < best_gates:
                best, best_gates = current, len(current)

        # 4. 深度门替换（无 max_tries 限制）
        remaining = time_limit - (time.time() - t0)
        if remaining > 10:
            current = deep_gate_substitution(
                best, T, num_inputs, time_limit=min(remaining * 0.4, 60)
            )
            if len(current) < best_gates:
                best, best_gates = current, len(current)

        # 5. 多门窗口替换
        remaining = time_limit - (time.time() - t0)
        if remaining > 10:
            current = multi_gate_swap(
                best, T, num_inputs, window=3, time_limit=min(remaining * 0.5, 60)
            )
            if len(current) < best_gates:
                best, best_gates = current, len(current)

        if verbose and best_gates < old_gates:
            print(f"    Exhaustive LS iter {iteration}: {old_gates} → {best_gates}")

        if best_gates == old_gates:
            break

    return best


if __name__ == "__main__":
    from benchmark_matrices import load_all_benchmarks

    benchmarks = load_all_benchmarks()

    for name, info in benchmarks.items():
        T = np.array(info['matrix'], dtype=np.int8)
        n_inputs = info['dim']
        sota = info['sota_gates']

        print(f"\n{'='*60}")
        print(f"Benchmark: {name} ({n_inputs}x{n_inputs}), SOTA={sota}")
        print(f"{'='*60}")

        # 先跑标准 Paar 作为基线
        from baselines import PaarAlgorithm
        paar = PaarAlgorithm()
        paar_circuit, paar_gates = paar.solve(T)
        print(f"  Standard Paar: {paar_gates} gates")

        # 多起点搜索
        best_circuit, best_gates, stats = multi_start_search(
            T, n_paar=2000, n_bp=500, time_limit=120,
            local_search_top=20, local_search_time=15
        )

        print(f"\n  RESULT: {best_gates} gates (Paar={paar_gates}, SOTA={sota})")
        if sota and best_gates <= sota:
            print(f"  ★★★ MATCHES OR BEATS SOTA! ★★★")
        elif sota:
            print(f"  Gap to SOTA: {best_gates - sota} gates")
