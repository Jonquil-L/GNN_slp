import numpy as np


class SLPGraphEnv:
    """
    SLP 环境：向量化操作 + beam search 支持。
    比原版快 3-5x（get_obs/step/get_v_mask 全部向量化）。
    """

    def __init__(self, target_matrix, max_extra_nodes=50, max_depth=10, ref_gates=None,
                 best_known=None):
        self.target = np.array(target_matrix, dtype=np.int8)
        self.num_targets, self.num_inputs = self.target.shape
        self.max_extra = max_extra_nodes
        self.max_depth = max_depth
        self.max_nodes = self.num_inputs + max_extra_nodes
        self.feature_dim = self.num_inputs + self.num_targets + 3
        # ref_gates: 用 Paar 基线校准效率奖励，而不是 num_inputs*2
        self.ref_gates = ref_gates
        # best_known: 已知最优门数，用于 AlphaZero 奖励
        self.best_known = best_known

    def reset(self):
        self.nodes = np.zeros((self.max_nodes, self.num_inputs), dtype=np.int8)
        for i in range(self.num_inputs):
            self.nodes[i, i] = 1
        self.depth = np.zeros(self.max_nodes, dtype=np.int32)
        self.valid = np.zeros(self.max_nodes, dtype=bool)
        self.valid[:self.num_inputs] = True
        self.next_idx = self.num_inputs
        self.achieved = np.zeros(self.num_targets, dtype=bool)
        self.circuit = []
        self.parents = {}

        # 向量化初始匹配
        input_nodes = self.nodes[:self.num_inputs]
        for t in range(self.num_targets):
            if np.any(np.all(input_nodes == self.target[t], axis=1)):
                self.achieved[t] = True

        self.min_dist = self._compute_min_distances()
        return self.get_obs()

    def _compute_min_distances(self):
        dists = np.full(self.num_targets, self.num_inputs + 1, dtype=np.float32)
        valid_nodes = self.nodes[self.valid]
        if len(valid_nodes) == 0:
            return dists
        for t in range(self.num_targets):
            if self.achieved[t]:
                dists[t] = 0.0
            else:
                d = np.sum(valid_nodes != self.target[t], axis=1)
                dists[t] = float(np.min(d))
        return dists

    def step(self, u, v):
        if u == v or u >= self.max_nodes or v >= self.max_nodes:
            return self.get_obs(), -5.0, True, "invalid"
        if not self.valid[u] or not self.valid[v]:
            return self.get_obs(), -5.0, True, "invalid"

        new_depth = max(int(self.depth[u]), int(self.depth[v])) + 1
        if new_depth > self.max_depth:
            return self.get_obs(), -5.0, True, "depth_exceeded"

        new_vec = (self.nodes[u].astype(np.int16) + self.nodes[v].astype(np.int16)) % 2
        new_vec = new_vec.astype(np.int8)

        if not np.any(new_vec):
            return self.get_obs(), -5.0, True, "zero"

        # 向量化重复检查
        valid_nodes = self.nodes[self.valid]
        if np.any(np.all(valid_nodes == new_vec, axis=1)):
            return self.get_obs(), -5.0, True, "duplicate"

        if self.next_idx >= self.max_nodes:
            return self.get_obs(), -5.0, True, "no_slots"

        idx = self.next_idx
        self.nodes[idx] = new_vec
        self.depth[idx] = new_depth
        self.valid[idx] = True
        self.next_idx += 1
        self.circuit.append((u, v))
        self.parents[idx] = (u, v)

        # 奖励
        reward = -1.0
        remaining = int(np.sum(~self.achieved))
        newly_achieved = 0
        any_hamming_improvement = False

        for t in range(self.num_targets):
            if not self.achieved[t]:
                dist = int(np.sum(new_vec != self.target[t]))
                if dist == 0:
                    self.achieved[t] = True
                    newly_achieved += 1
                    progress = 1.0 - (remaining - newly_achieved) / max(self.num_targets, 1)
                    reward += 10.0 + 8.0 * progress
                elif dist < self.min_dist[t]:
                    reward += 1.5 * (self.min_dist[t] - dist)
                    self.min_dist[t] = dist
                    any_hamming_improvement = True

        # 冗余惩罚：新向量未改进任何目标且与已有向量过于接近
        if newly_achieved == 0 and not any_hamming_improvement:
            min_hamming_to_existing = np.min(np.sum(valid_nodes != new_vec, axis=1))
            if min_hamming_to_existing <= 1:
                reward -= 2.0

        done = False
        if np.all(self.achieved):
            n_gates = len(self.circuit)
            ref = self.ref_gates if self.ref_gates else self.num_inputs * 3
            efficiency = max(0.0, 1.0 - n_gates / max(ref * 1.8, 1))
            reward += 50.0 + 80.0 * efficiency
            # AlphaZero 奖励：直接奖励击败已知最优
            if self.best_known is not None and n_gates < self.best_known:
                reward += (self.best_known - n_gates) * 20.0
            done = True
        elif self.next_idx >= self.max_nodes:
            done = True

        return self.get_obs(), reward, done, ""

    def get_obs(self):
        features = np.zeros((self.max_nodes, self.feature_dim), dtype=np.float32)
        valid_idx = np.where(self.valid)[0]

        if len(valid_idx) > 0:
            valid_nodes = self.nodes[valid_idx]
            # GF(2) 向量
            features[valid_idx, :self.num_inputs] = valid_nodes.astype(np.float32)
            # 归一化深度
            features[valid_idx, self.num_inputs] = self.depth[valid_idx].astype(np.float32) / max(self.max_depth, 1)
            # valid 标记
            features[valid_idx, self.num_inputs + 1] = 1.0

            # 批量距离计算 (向量化核心)
            unachieved = np.where(~self.achieved)[0]
            if len(unachieved) > 0:
                target_vecs = self.target[unachieved]  # (n_ua, n_inputs)
                # broadcasting: (n_valid, 1, n_inputs) vs (1, n_ua, n_inputs)
                dists = np.sum(valid_nodes[:, None, :] != target_vecs[None, :, :], axis=2)
                for k, t in enumerate(unachieved):
                    features[valid_idx, self.num_inputs + 2 + t] = dists[:, k].astype(np.float32) / max(self.num_inputs, 1)

            # is_target (向量化)
            for t in range(self.num_targets):
                matches = np.all(valid_nodes == self.target[t], axis=1)
                if np.any(matches):
                    features[valid_idx[matches], -1] = 1.0

        valid_f = self.valid.astype(np.float32)
        adj = np.outer(valid_f, valid_f)
        return {"node_features": features, "adj": adj, "valid_mask": valid_f.copy()}

    def get_v_mask(self, u):
        mask = self.valid.astype(np.float32).copy()
        mask[u] = 0.0
        # 向量化深度检查
        over_depth = (np.maximum(int(self.depth[u]), self.depth) + 1) > self.max_depth
        mask[over_depth] = 0.0
        return mask

    def get_hamming_v_mask(self, u, fallback=True):
        """
        Target-aware v masking: 仅保留能使某个未达成目标的 Hamming 距离严格下降的 v。
        若 mask 全零则 fallback 到 get_v_mask(u)。
        复杂度 O(N * T * num_inputs)，对典型规模可忽略。
        """
        base_mask = self.get_v_mask(u)

        unachieved = np.where(~self.achieved)[0]
        if len(unachieved) == 0:
            return base_mask

        valid_v_indices = np.where(base_mask > 0)[0]
        if len(valid_v_indices) == 0:
            return base_mask

        # 向量化 XOR: nodes[u] XOR nodes[each valid v]
        u_vec = self.nodes[u]  # (num_inputs,)
        v_vecs = self.nodes[valid_v_indices]  # (n_valid, num_inputs)
        new_vecs = (u_vec.astype(np.int16) + v_vecs.astype(np.int16)) % 2  # (n_valid, num_inputs)

        # 过滤零向量
        nonzero = np.any(new_vecs, axis=1)

        # 检查重复
        valid_nodes_set = set()
        for idx in np.where(self.valid)[0]:
            valid_nodes_set.add(self.nodes[idx].tobytes())
        not_dup = np.array([nv.tobytes() not in valid_nodes_set for nv in new_vecs.astype(np.int8)])

        # 计算到所有未达成目标的 Hamming 距离
        target_vecs = self.target[unachieved]  # (n_unachieved, num_inputs)
        # broadcasting: (n_valid, 1, num_inputs) vs (1, n_unachieved, num_inputs)
        dists = np.sum(new_vecs[:, None, :].astype(np.int8) != target_vecs[None, :, :], axis=2)  # (n_valid, n_unachieved)

        # 当前各未达成目标的最小距离
        current_min = self.min_dist[unachieved]  # (n_unachieved,)

        # v 有用条件：创建的向量比当前最近向量更接近某个目标
        improves = dists < current_min[None, :]  # (n_valid, n_unachieved)
        exact_match = np.any(dists == 0, axis=1)
        useful = np.any(improves, axis=1) | exact_match

        # 综合过滤
        useful = useful & nonzero & not_dup

        # 构建 hamming mask
        hamming_mask = np.zeros(self.max_nodes, dtype=np.float32)
        hamming_mask[valid_v_indices[useful]] = 1.0

        # Fallback: 如果过滤太激进（全零），回退到基础 mask
        if fallback and np.sum(hamming_mask) == 0:
            return base_mask

        return hamming_mask

    def get_target_aware_u_mask(self):
        """
        过滤 u 候选：仅保留与某个未达成目标有 bit 重叠的节点。
        输入基向量始终保留。若过滤后 < 2 个节点则 fallback。
        """
        base_mask = self.valid.astype(np.float32).copy()

        unachieved = np.where(~self.achieved)[0]
        if len(unachieved) == 0:
            return base_mask

        valid_idx = np.where(self.valid)[0]
        if len(valid_idx) < 2:
            return base_mask

        valid_nodes = self.nodes[valid_idx]  # (n_valid, num_inputs)
        target_vecs = self.target[unachieved]  # (n_ua, num_inputs)

        # 对每个 valid node，检查与任意未达成目标的 bit 重叠
        # overlap[i, j] = sum(valid_nodes[i] & target_vecs[j])
        overlap = np.einsum('vi,ti->vt', valid_nodes.astype(np.int16), target_vecs.astype(np.int16))
        u_useful = np.any(overlap >= 1, axis=1)  # (n_valid,)

        # 输入基向量始终保留
        for i, idx in enumerate(valid_idx):
            if idx < self.num_inputs:
                u_useful[i] = True

        u_mask = np.zeros(self.max_nodes, dtype=np.float32)
        u_mask[valid_idx[u_useful]] = 1.0

        # Fallback: 过滤后不足 2 个则回退
        if np.sum(u_mask) < 2:
            return base_mask

        return u_mask

    def step_fast(self, u, v):
        """快速 step: 跳过 get_obs() 计算，MCTS/搜索专用，快 5-10x"""
        if u == v or u >= self.max_nodes or v >= self.max_nodes:
            return -5.0, True, "invalid"
        if not self.valid[u] or not self.valid[v]:
            return -5.0, True, "invalid"

        new_depth = max(int(self.depth[u]), int(self.depth[v])) + 1
        if new_depth > self.max_depth:
            return -5.0, True, "depth_exceeded"

        new_vec = (self.nodes[u].astype(np.int16) + self.nodes[v].astype(np.int16)) % 2
        new_vec = new_vec.astype(np.int8)

        if not np.any(new_vec):
            return -5.0, True, "zero"

        valid_nodes = self.nodes[self.valid]
        if np.any(np.all(valid_nodes == new_vec, axis=1)):
            return -5.0, True, "duplicate"

        if self.next_idx >= self.max_nodes:
            return -5.0, True, "no_slots"

        idx = self.next_idx
        self.nodes[idx] = new_vec
        self.depth[idx] = new_depth
        self.valid[idx] = True
        self.next_idx += 1
        self.circuit.append((u, v))
        self.parents[idx] = (u, v)

        reward = -1.0
        remaining = int(np.sum(~self.achieved))
        newly_achieved = 0
        any_hamming_improvement = False

        for t in range(self.num_targets):
            if not self.achieved[t]:
                dist = int(np.sum(new_vec != self.target[t]))
                if dist == 0:
                    self.achieved[t] = True
                    newly_achieved += 1
                    progress = 1.0 - (remaining - newly_achieved) / max(self.num_targets, 1)
                    reward += 10.0 + 8.0 * progress
                elif dist < self.min_dist[t]:
                    reward += 1.5 * (self.min_dist[t] - dist)
                    self.min_dist[t] = dist
                    any_hamming_improvement = True

        # 冗余惩罚
        if newly_achieved == 0 and not any_hamming_improvement:
            min_hamming_to_existing = np.min(np.sum(valid_nodes != new_vec, axis=1))
            if min_hamming_to_existing <= 1:
                reward -= 2.0

        done = False
        if np.all(self.achieved):
            n_gates = len(self.circuit)
            ref = self.ref_gates if self.ref_gates else self.num_inputs * 3
            efficiency = max(0.0, 1.0 - n_gates / max(ref * 1.8, 1))
            reward += 50.0 + 80.0 * efficiency
            if self.best_known is not None and n_gates < self.best_known:
                reward += (self.best_known - n_gates) * 20.0
            done = True
        elif self.next_idx >= self.max_nodes:
            done = True

        return reward, done, ""

    def set_best_known(self, n):
        """设置已知最优门数，用于 AlphaZero 奖励"""
        self.best_known = n

    def copy(self):
        """高效复制，用于 beam search / MCTS"""
        new = SLPGraphEnv.__new__(SLPGraphEnv)
        new.target = self.target  # 共享引用，不可变
        new.num_targets = self.num_targets
        new.num_inputs = self.num_inputs
        new.max_extra = self.max_extra
        new.max_depth = self.max_depth
        new.max_nodes = self.max_nodes
        new.feature_dim = self.feature_dim
        new.ref_gates = self.ref_gates
        new.best_known = self.best_known
        new.nodes = self.nodes.copy()
        new.depth = self.depth.copy()
        new.valid = self.valid.copy()
        new.next_idx = self.next_idx
        new.achieved = self.achieved.copy()
        new.circuit = list(self.circuit)
        new.parents = dict(self.parents)
        new.min_dist = self.min_dist.copy()
        return new
