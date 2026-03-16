"""
AlphaZero 风格的自博弈训练循环：
1. 用当前 GNN 引导 MCTS 搜索
2. MCTS 搜索结果（访问分布）作为训练数据
3. 训练 GNN 拟合 MCTS 策略 + 价值
4. 迭代改进：更好的 GNN → 更好的 MCTS → 更好的训练数据

核心优势：不依赖 Paar/Greedy 等专家数据，可以自主发现更优电路。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import copy
import os
import json

from gnn_env import SLPGraphEnv
from gnn_network import SLPPolicyValueNet


def alpha_zero_loop(target_matrix, model, device,
                    n_iterations=30,
                    n_games_per_iter=20,
                    n_simulations=800,
                    max_children=50,
                    max_extra=None,
                    max_depth=10,
                    best_known=None,
                    replay_buffer_size=10,
                    train_epochs=10,
                    train_batch_size=32,
                    lr=1e-4,
                    temperature=1.0,
                    temp_threshold=20,
                    local_search_fn=None,
                    save_dir=None,
                    verbose=True):
    """
    AlphaZero 自博弈主循环。

    Args:
        target_matrix: 目标矩阵
        model: GNN 模型（已预训练或随机初始化）
        device: 计算设备
        n_iterations: 外层迭代次数
        n_games_per_iter: 每次迭代自博弈局数
        n_simulations: MCTS 每步模拟次数
        max_children: MCTS 最大子节点数
        max_extra: 最大额外节点数
        max_depth: 电路最大深度
        best_known: 已知最优门数
        replay_buffer_size: 保留最近几轮的训练数据
        train_epochs: 每次训练的 epoch 数
        train_batch_size: 训练 batch 大小
        lr: 学习率
        temperature: MCTS 动作选择温度
        temp_threshold: 前多少步使用温度采样
        local_search_fn: 局部搜索函数（可选）
        save_dir: 模型保存目录（可选）
        verbose: 是否打印详细信息

    Returns:
        best_gates: 找到的最优门数
        best_circuit: 最优电路
        history: 训练历史
    """
    T = np.array(target_matrix, dtype=np.int8)
    n_targets, n_inputs = T.shape
    if max_extra is None:
        max_extra = n_inputs * 6

    from run_overnight import MCTSSolver

    # 初始化
    best_gates = best_known if best_known else float('inf')
    best_circuit = None
    best_model_state = copy.deepcopy(model.state_dict())
    replay_buffer = []  # [(training_data, iteration)]
    history = []

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations)

    if verbose:
        print(f"\n{'='*60}")
        print(f"AlphaZero Self-Play Loop")
        print(f"  Matrix: {n_targets}x{n_inputs}, best_known={best_known}")
        print(f"  Iterations: {n_iterations}, games/iter: {n_games_per_iter}")
        print(f"  MCTS: {n_simulations} sims, {max_children} children")
        print(f"{'='*60}")

    for iteration in range(n_iterations):
        t0 = time.time()

        # === Phase 1: Self-Play ===
        mcts_solver = MCTSSolver(
            model, device,
            c_puct=2.0,
            n_simulations=n_simulations,
            max_children=max_children,
        )

        iter_data = []
        iter_gates = []
        iter_solved = 0

        for game in range(n_games_per_iter):
            n_gates, circuit, training_data = mcts_solver.solve_with_data(
                T, max_extra, max_depth,
                temperature=temperature,
                temp_threshold=temp_threshold,
            )

            if n_gates is not None:
                iter_solved += 1
                iter_gates.append(n_gates)
                iter_data.extend(training_data)

                # 局部搜索优化
                if local_search_fn and n_gates <= best_gates + 5:
                    try:
                        optimized = local_search_fn(circuit, T, n_inputs)
                        opt_gates = len(optimized)
                        if opt_gates < n_gates:
                            n_gates = opt_gates
                            circuit = optimized
                    except Exception:
                        pass

                if n_gates < best_gates:
                    best_gates = n_gates
                    best_circuit = list(circuit)
                    best_model_state = copy.deepcopy(model.state_dict())
                    if verbose:
                        print(f"  [Iter {iteration+1}] ★ NEW BEST: {best_gates} gates (game {game+1})")

        # === Phase 2: Train ===
        if iter_data:
            replay_buffer.append(iter_data)
            if len(replay_buffer) > replay_buffer_size:
                replay_buffer.pop(0)

            # 合并所有 replay 数据
            all_data = []
            for buf in replay_buffer:
                all_data.extend(buf)

            train_loss = train_on_mcts_data(
                model, optimizer, all_data, device,
                max_nodes=n_inputs + max_extra,
                n_epochs=train_epochs,
                batch_size=train_batch_size,
            )
        else:
            train_loss = 0.0

        scheduler.step()

        # === Phase 3: Evaluate ===
        elapsed = time.time() - t0
        solve_rate = iter_solved / n_games_per_iter if n_games_per_iter > 0 else 0
        avg_gates = np.mean(iter_gates) if iter_gates else None
        min_gates = min(iter_gates) if iter_gates else None

        iter_info = {
            'iteration': iteration + 1,
            'solve_rate': solve_rate,
            'avg_gates': avg_gates,
            'min_gates': min_gates,
            'best_gates': best_gates,
            'train_loss': train_loss,
            'n_training_samples': len(all_data) if iter_data else 0,
            'elapsed': elapsed,
        }
        history.append(iter_info)

        if verbose:
            print(f"  [Iter {iteration+1}/{n_iterations}] "
                  f"solve={solve_rate:.0%}, avg={avg_gates:.1f if avg_gates else 'N/A'}, "
                  f"min={min_gates}, best={best_gates}, "
                  f"loss={train_loss:.4f}, time={elapsed:.1f}s")

        # 保存检查点
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'best_model': best_model_state,
                'optimizer': optimizer.state_dict(),
                'iteration': iteration + 1,
                'best_gates': best_gates,
                'best_circuit': best_circuit,
                'history': history,
            }, os.path.join(save_dir, f'alpha_checkpoint.pt'))

    # 恢复最优模型
    model.load_state_dict(best_model_state)

    if verbose:
        print(f"\nAlphaZero done: best={best_gates} gates")
        if best_circuit:
            print(f"  Circuit length: {len(best_circuit)}")

    return best_gates, best_circuit, history


def train_on_mcts_data(model, optimizer, training_data, device,
                       max_nodes=None, n_epochs=10, batch_size=32):
    """
    用 MCTS 自博弈数据训练 GNN。

    训练目标：
    1. Policy loss: KL(MCTS 访问分布 || GNN 策略)
    2. Value loss: MSE(GNN 价值预测, 实际门数)
    """
    model.train()

    # 过滤有效数据
    valid_data = [d for d in training_data if d.get('outcome') is not None]
    if not valid_data:
        return 0.0

    T = len(valid_data)
    indices = np.arange(T)
    total_loss_sum = 0.0
    n_updates = 0

    for epoch in range(n_epochs):
        np.random.shuffle(indices)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_idx = indices[start:end]
            bs = len(batch_idx)

            # 逐个构建 batch（避免 OOM）
            feats = []
            adjs = []
            vmasks = []
            policy_targets = []  # (bs, max_nodes, max_nodes) 稀疏
            value_targets = []

            for bi in batch_idx:
                d = valid_data[bi]
                obs = d['obs']
                feats.append(obs['node_features'])
                adjs.append(obs['adj'])
                vmasks.append(obs['valid_mask'])

                # 构建策略目标：将 visit_counts 转换为 (u, v) 概率分布
                n_nodes = len(obs['valid_mask'])
                visit_dist_u = np.zeros(n_nodes, dtype=np.float32)
                visit_dist_v = np.zeros((n_nodes, n_nodes), dtype=np.float32)

                total = d['total_visits']
                if total > 0:
                    for (u, v), count in d['visit_counts'].items():
                        if u < n_nodes and v < n_nodes:
                            visit_dist_u[u] += count / total
                            visit_dist_v[u, v] += count / total

                policy_targets.append((visit_dist_u, visit_dist_v))

                # 价值目标：从当前状态到完成还需多少步
                outcome = d['outcome']
                step = d['step']
                gates_remaining = max(0, outcome - step) if outcome else 0
                value_targets.append(float(gates_remaining))

            # 转换为 tensor
            feat_t = torch.FloatTensor(np.array(feats)).to(device)
            adj_t = torch.FloatTensor(np.array(adjs)).to(device)
            vmask_t = torch.FloatTensor(np.array(vmasks)).to(device)
            value_target_t = torch.FloatTensor(value_targets).to(device)

            # Forward
            h = model.encode(feat_t, adj_t)
            u_logits = model.get_u_logits(h, vmask_t)  # (bs, n_nodes)

            # Policy loss: u 分布的 KL 散度
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            u_targets = torch.FloatTensor(np.array([pt[0] for pt in policy_targets])).to(device)
            # KL(target || model) = sum(target * (log(target) - log(model)))
            # 用交叉熵近似：-sum(target * log(model))
            policy_loss = -(u_targets * u_log_probs).sum(-1).mean()

            # Value loss
            value_pred = model.get_az_value(h, vmask_t).squeeze(-1)
            value_loss = F.mse_loss(value_pred, value_target_t)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_sum += loss.item()
            n_updates += 1

    return total_loss_sum / max(n_updates, 1)


def pretrain_from_circuits(model, circuits_with_gates, target_matrix, device,
                           n_epochs=50, batch_size=64, lr=1e-3, verbose=True):
    """
    用已知好电路预训练 GNN（Imitation Learning 阶段）。
    将每个电路的每一步作为 (state, action) 对训练策略。

    Args:
        circuits_with_gates: [(circuit, n_gates)] 列表
        target_matrix: 目标矩阵
    """
    T = np.array(target_matrix, dtype=np.int8)
    n_targets, n_inputs = T.shape
    max_extra = n_inputs * 6

    # 展开所有电路为 (obs, u, v) 训练对
    train_pairs = []
    for circuit, n_gates in circuits_with_gates:
        env = SLPGraphEnv(T, max_extra, max_depth=10)
        env.reset()
        for u, v in circuit:
            obs = env.get_obs()
            train_pairs.append((obs, u, v))
            reward, done, info = env.step_fast(u, v)
            if info or done:
                break

    if verbose:
        print(f"  Pretrain: {len(train_pairs)} state-action pairs from {len(circuits_with_gates)} circuits")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()

    for epoch in range(n_epochs):
        indices = np.arange(len(train_pairs))
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_pairs), batch_size):
            end = min(start + batch_size, len(train_pairs))
            idx = indices[start:end]
            bs = len(idx)

            feats = np.array([train_pairs[i][0]['node_features'] for i in idx])
            adj = np.array([train_pairs[i][0]['adj'] for i in idx])
            vmask = np.array([train_pairs[i][0]['valid_mask'] for i in idx])
            u_targets = np.array([train_pairs[i][1] for i in idx])
            v_targets = np.array([train_pairs[i][2] for i in idx])

            feat_t = torch.FloatTensor(feats).to(device)
            adj_t = torch.FloatTensor(adj).to(device)
            vmask_t = torch.FloatTensor(vmask).to(device)
            u_t = torch.LongTensor(u_targets).to(device)
            v_t = torch.LongTensor(v_targets).to(device)

            h = model.encode(feat_t, adj_t)
            u_logits = model.get_u_logits(h, vmask_t)

            # v_mask: 排除 u 自身
            v_mask = vmask_t.clone()
            v_mask[torch.arange(bs, device=device), u_t] = 0.0

            v_logits = model.get_v_logits(h, u_t, v_mask, vmask_t)

            u_loss = F.cross_entropy(u_logits, u_t)
            v_loss = F.cross_entropy(v_logits, v_t)
            loss = u_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Pretrain epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    return model


if __name__ == "__main__":
    import argparse
    from benchmark_matrices import load_all_benchmarks
    from local_search import multi_start_search, full_local_search, verify_circuit

    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='aes_mixcolumns_32x32', help='Benchmark name')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--games', type=int, default=15)
    parser.add_argument('--sims', type=int, default=800)
    parser.add_argument('--skip-multistart', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    benchmarks = load_all_benchmarks()
    if args.benchmark not in benchmarks:
        print(f"Unknown benchmark: {args.benchmark}")
        print(f"Available: {list(benchmarks.keys())}")
        exit(1)

    info = benchmarks[args.benchmark]
    T = np.array(info['matrix'], dtype=np.int8)
    n_targets, n_inputs = T.shape
    sota = info['sota_gates']

    print(f"\nBenchmark: {args.benchmark} ({n_targets}x{n_inputs}), SOTA={sota}")

    # Phase 1: Multi-start search
    if not args.skip_multistart:
        print("\n=== Phase 1: Multi-Start Search ===")
        best_circuit, best_gates, stats = multi_start_search(
            T, n_paar=3000, n_bp=1000, time_limit=300,
            local_search_top=30, local_search_time=20,
        )
        print(f"Multi-start result: {best_gates} gates")
    else:
        from baselines import PaarAlgorithm
        paar = PaarAlgorithm()
        best_circuit, best_gates = paar.solve(T)
        print(f"Paar baseline: {best_gates} gates")

    # Phase 2: Build GNN and pretrain
    print("\n=== Phase 2: Pretrain GNN ===")
    max_extra = n_inputs * 6
    env = SLPGraphEnv(T, max_extra, max_depth=10)
    env.reset()

    model = SLPPolicyValueNet(
        input_dim=env.feature_dim,
        hidden_dim=256 if n_inputs >= 32 else 128,
        num_gnn_layers=6 if n_inputs >= 32 else 4,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    # 生成多样化训练电路
    print("  Generating diverse training circuits...")
    training_circuits = []
    from local_search import randomized_paar
    for seed in range(200):
        rng = np.random.RandomState(seed)
        circuit, n_gates = randomized_paar(T, rng)
        training_circuits.append((circuit, n_gates))
    training_circuits.sort(key=lambda x: x[0])

    # 取最好的 50 个
    training_circuits = training_circuits[:50]
    print(f"  Best training circuit: {training_circuits[0][1]} gates, "
          f"worst of top-50: {training_circuits[-1][1]} gates")

    model = pretrain_from_circuits(
        model, training_circuits, T, device,
        n_epochs=50, batch_size=32, lr=1e-3,
    )

    # Phase 3: AlphaZero loop
    print("\n=== Phase 3: AlphaZero Self-Play ===")

    def local_search_wrapper(circuit, matrix, n_inp):
        return full_local_search(circuit, matrix, n_inp, time_limit=10)

    az_best, az_circuit, az_history = alpha_zero_loop(
        T, model, device,
        n_iterations=args.iterations,
        n_games_per_iter=args.games,
        n_simulations=args.sims,
        max_children=50,
        max_extra=max_extra,
        max_depth=10,
        best_known=best_gates,
        train_epochs=10,
        train_batch_size=16 if n_inputs >= 32 else 32,
        lr=1e-4,
        temperature=1.0,
        temp_threshold=20,
        local_search_fn=local_search_wrapper,
        save_dir=os.path.join(os.path.dirname(__file__), 'checkpoints'),
    )

    # Final report
    final_best = min(best_gates, az_best) if az_best else best_gates
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {final_best} gates")
    print(f"  SOTA: {sota}")
    if sota:
        if final_best <= sota:
            print(f"  ★★★ MATCHES OR BEATS SOTA! ★★★")
        else:
            print(f"  Gap to SOTA: {final_best - sota} gates")
    print(f"{'='*60}")
