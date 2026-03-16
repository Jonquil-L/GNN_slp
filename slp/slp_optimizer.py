"""
SLP (Shortest Linear Program) 优化器：GNN + MCTS 混合框架。

四层优化策略:
  Layer 0: 代数分解 (AES-specific) — 利用 GF(2^8) 循环矩阵结构
  Layer 1: 随机化多起点 Paar/BP + 局部搜索
  Layer 2: GNN 预训练 + AlphaZero 自博弈
  Layer 3: 深度 MCTS 搜索 + 穷举局部搜索

用法:
    python slp_optimizer.py --gpu                          # 全部密码学 benchmark
    python slp_optimizer.py --gpu --bench aes_mixcolumns_32x32
    python slp_optimizer.py --gpu --quick                  # 快速模式
    python slp_optimizer.py --experiment                   # 结题报告完整实验
    python slp_optimizer.py --experiment --quick            # 结题报告快速版
"""
import numpy as np
import torch
import time
import os
import json
import argparse

from benchmark_matrices import load_all_benchmarks
from baselines import PaarAlgorithm, BoyarPeraltaAlgorithm
from local_search import (
    multi_start_search, iterated_local_search, full_local_search,
    verify_circuit, randomized_paar, exhaustive_local_search,
)
from gnn_env import SLPGraphEnv
from gnn_network import SLPPolicyValueNet
from alpha_slp import alpha_zero_loop, pretrain_from_circuits


def is_aes_mixcolumns(target_matrix, n_inputs):
    """检测是否为 AES MixColumns (32x32)"""
    return n_inputs == 32 and len(target_matrix) == 32


def layer0_algebraic(T, n_inputs, verbose=True):
    """
    Layer 0: 代数分解攻击 (AES-specific)。
    利用 GF(2^8) 循环矩阵结构生成多种代数电路变体，
    每个变体用穷举局部搜索优化。

    Returns: list of (circuit, n_gates, name)
    """
    if not is_aes_mixcolumns(T, n_inputs):
        return []

    if verbose:
        print(f"\n  --- Layer 0: Algebraic Decomposition (AES-specific) ---")

    try:
        from algebraic_decompose import generate_algebraic_circuits
        all_circuits = generate_algebraic_circuits(T, verbose=verbose)

        # 对所有代数变体运行穷举局部搜索
        results = []
        for circuit, n_gates, name in all_circuits:
            if verbose:
                print(f"    Running exhaustive LS on {name} ({n_gates} gates)...")
            opt = exhaustive_local_search(circuit, T, n_inputs, time_limit=60, verbose=verbose)
            opt_gates = len(opt)
            if verify_circuit(opt, T, n_inputs):
                results.append((opt, opt_gates, f"{name}_exhaustive"))
                if verbose and opt_gates < n_gates:
                    print(f"    ★ {name}: {n_gates} → {opt_gates} gates")

        results.sort(key=lambda x: x[1])
        if verbose and results:
            print(f"  Layer 0 best: {results[0][1]} gates ({results[0][2]})")
        return results

    except Exception as e:
        if verbose:
            print(f"  Layer 0 failed: {e}")
        return []


def optimize_pipeline(target_matrix, n_inputs, sota_gates=None, device='cpu',
                      quick=False, verbose=True):
    """
    完整的 SLP 优化流水线。

    Returns:
        results: dict with best_gates, best_circuit, phase details
    """
    T = np.array(target_matrix, dtype=np.int8)
    n_targets = len(T)
    results = {'n_inputs': n_inputs, 'n_targets': n_targets, 'sota': sota_gates}
    t_total = time.time()

    # === Baseline ===
    paar = PaarAlgorithm()
    paar_circuit, paar_gates = paar.solve(T)
    results['paar_gates'] = paar_gates
    if verbose:
        print(f"  Paar baseline: {paar_gates} gates")

    best_gates = paar_gates
    best_circuit = paar_circuit

    # === Layer 0: Algebraic Decomposition (AES-specific) ===
    algebraic_results = layer0_algebraic(T, n_inputs, verbose=verbose)
    if algebraic_results:
        alg_circuit, alg_gates, alg_name = algebraic_results[0]
        results['algebraic_gates'] = alg_gates
        results['algebraic_name'] = alg_name
        if alg_gates < best_gates and verify_circuit(alg_circuit, T, n_inputs):
            best_gates = alg_gates
            best_circuit = alg_circuit

    # === Layer 1: Multi-Start + Local Search ===
    if verbose:
        print(f"\n  --- Layer 1: Multi-Start Search ---")

    n_paar = 1000 if quick else 5000
    n_bp = 300 if quick else 2000
    time_limit_ms = 60 if quick else 300
    ls_top = 10 if quick else 50
    ls_time = 10 if quick else 30

    ms_circuit, ms_gates, ms_stats = multi_start_search(
        T, n_paar=n_paar, n_bp=n_bp,
        time_limit=time_limit_ms,
        local_search_top=ls_top,
        local_search_time=ls_time,
        verbose=verbose,
    )

    if ms_gates < best_gates and verify_circuit(ms_circuit, T, n_inputs):
        best_gates = ms_gates
        best_circuit = ms_circuit

    results['multistart_gates'] = ms_gates
    results['multistart_stats'] = ms_stats

    # ILS (Iterated Local Search)
    if verbose:
        print(f"\n  --- Layer 1b: Iterated Local Search ---")

    ils_n = 50 if quick else 200
    ils_time = 30 if quick else 180

    ils_circuit, ils_gates = iterated_local_search(
        T, num_inputs=n_inputs,
        n_restarts=ils_n,
        time_limit=ils_time,
        verbose=verbose,
    )

    if ils_gates < best_gates and verify_circuit(ils_circuit, T, n_inputs):
        best_gates = ils_gates
        best_circuit = ils_circuit

    results['ils_gates'] = ils_gates

    # Exhaustive local search on best so far
    if verbose:
        print(f"\n  --- Layer 1c: Exhaustive Local Search on best ({best_gates} gates) ---")
    exh_time = 30 if quick else 120
    exh_circuit = exhaustive_local_search(
        best_circuit, T, n_inputs, time_limit=exh_time, verbose=verbose
    )
    exh_gates = len(exh_circuit)
    if exh_gates < best_gates and verify_circuit(exh_circuit, T, n_inputs):
        if verbose:
            print(f"  ★ Exhaustive LS improved: {best_gates} → {exh_gates}")
        best_gates = exh_gates
        best_circuit = exh_circuit
    results['exhaustive_ls_gates'] = exh_gates

    if verbose:
        print(f"\n  Layer 1 result: {best_gates} gates (Paar={paar_gates})")
        if sota_gates:
            print(f"  Gap to SOTA: {best_gates - sota_gates}")

    # === Layer 2: AlphaZero Self-Play ===
    if verbose:
        print(f"\n  --- Layer 2: AlphaZero Self-Play ---")

    max_extra = n_inputs * 6
    env = SLPGraphEnv(T, max_extra, max_depth=10)
    env.reset()

    hidden_dim = 256 if n_inputs >= 32 else 128
    n_layers = 6 if n_inputs >= 32 else 4

    model = SLPPolicyValueNet(
        input_dim=env.feature_dim,
        hidden_dim=hidden_dim,
        num_gnn_layers=n_layers,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    # 生成预训练数据：混合 Paar + 代数变体
    if verbose:
        print(f"  Generating diverse training circuits for pretraining...")
    training_circuits = []
    for seed in range(500 if not quick else 100):
        rng = np.random.RandomState(seed)
        circuit, n_gates = randomized_paar(T, rng)
        training_circuits.append((circuit, n_gates))

    # 加入代数变体（提供不同拓扑的训练数据）
    if algebraic_results:
        for alg_c, alg_g, alg_n in algebraic_results[:10]:
            training_circuits.append((alg_c, alg_g))

    training_circuits.sort(key=lambda x: x[1])
    training_circuits = training_circuits[:100 if not quick else 30]

    if verbose:
        print(f"  Best pretraining circuit: {training_circuits[0][1]} gates")
        print(f"  Training set: {len(training_circuits)} circuits, "
              f"range [{training_circuits[0][1]}, {training_circuits[-1][1]}]")

    # 预训练
    model = pretrain_from_circuits(
        model, training_circuits, T, device,
        n_epochs=30 if quick else 60,
        batch_size=16 if n_inputs >= 32 else 32,
        lr=1e-3,
        verbose=verbose,
    )

    # AlphaZero with scaled-up parameters
    def local_search_wrapper(circuit, matrix, n_inp):
        return full_local_search(circuit, matrix, n_inp, time_limit=10)

    az_iters = 10 if quick else 40       # 30 → 40
    az_games = 8 if quick else 25        # 20 → 25
    az_sims = 400 if quick else 1200     # 800 → 1200

    az_best, az_circuit, az_history = alpha_zero_loop(
        T, model, device,
        n_iterations=az_iters,
        n_games_per_iter=az_games,
        n_simulations=az_sims,
        max_children=60,                  # 50 → 60
        max_extra=max_extra,
        max_depth=10,
        best_known=best_gates,
        train_epochs=10,                  # 8 → 10
        train_batch_size=16 if n_inputs >= 32 else 32,
        lr=1e-4,
        temperature=1.0,
        temp_threshold=20,                # 15 → 20
        local_search_fn=local_search_wrapper,
        save_dir=None,
        verbose=verbose,
    )

    if az_best and az_best < best_gates:
        if az_circuit and verify_circuit(az_circuit, T, n_inputs):
            best_gates = az_best
            best_circuit = az_circuit

    results['az_best'] = az_best
    results['az_history'] = az_history

    # === Layer 3: Deep MCTS + Final Local Search ===
    if verbose:
        print(f"\n  --- Layer 3: Deep MCTS ---")

    from run_overnight import MCTSSolver, beam_search_solve

    # Deep MCTS with scaled-up simulations
    deep_sims = 800 if quick else 2400    # 1600 → 2400
    deep_restarts = 3 if quick else 15    # 10 → 15
    mcts_solver = MCTSSolver(
        model, device,
        c_puct=2.0,
        n_simulations=deep_sims,
        max_children=100,                  # 80 → 100
    )
    mcts_gates, mcts_circuit = mcts_solver.solve(
        T, max_extra, 10, n_restarts=deep_restarts
    )

    if mcts_gates and mcts_circuit:
        # 穷举局部搜索优化 MCTS 结果
        optimized = exhaustive_local_search(mcts_circuit, T, n_inputs, time_limit=60)
        if verify_circuit(optimized, T, n_inputs):
            opt_gates = len(optimized)
            if verbose:
                print(f"  MCTS: {mcts_gates} → {opt_gates} gates (after exhaustive LS)")
            if opt_gates < best_gates:
                best_gates = opt_gates
                best_circuit = optimized

    results['mcts_gates'] = mcts_gates

    # Beam search with wider beam
    beam_w = 20 if quick else 50          # 20 → 50
    beam_gates = beam_search_solve(model, T, max_extra, 10, device, beam_width=beam_w)
    if beam_gates:
        results['beam_gates'] = beam_gates
        if verbose:
            print(f"  Beam search (w={beam_w}): {beam_gates} gates")

    # === Final: Exhaustive local search on global best ===
    if verbose:
        print(f"\n  --- Final exhaustive optimization ({best_gates} gates) ---")
    final_time = 60 if quick else 300
    final_circuit = exhaustive_local_search(
        best_circuit, T, n_inputs, time_limit=final_time, verbose=verbose
    )
    final_gates = len(final_circuit)
    if final_gates < best_gates and verify_circuit(final_circuit, T, n_inputs):
        if verbose:
            print(f"  ★ Final optimization: {best_gates} → {final_gates}")
        best_gates = final_gates
        best_circuit = final_circuit

    # Final summary
    total_time = time.time() - t_total
    results['best_gates'] = best_gates
    results['best_circuit'] = best_circuit
    results['total_time'] = total_time

    return results


def run_experiment_suite(device='cpu', quick=False):
    """
    结题报告实验套件。设计 5 组实验，全面展示框架能力。

    实验 1: 随机矩阵 — Paar vs 本框架（展示通用优化能力）
    实验 2: 密码学 benchmark — 与已知结果对比
    实验 3: 消融实验 — 各层贡献分析
    实验 4: 可扩展性 — 不同矩阵规模的表现
    实验 5: 收敛曲线 — AlphaZero 训练过程可视化
    """
    from benchmark_matrices import get_random_matrix, load_all_benchmarks

    all_exp = {}
    t_start = time.time()

    # ================================================================
    # 实验 1: 随机矩阵上的优化效果
    # 目的: 展示框架在无已知代数结构矩阵上的通用优化能力
    # 预期: 相比 Paar baseline 稳定改进 5-15%
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  实验 1: 随机矩阵优化 (Paar baseline vs 本框架)")
    print(f"{'='*70}")

    random_results = []
    dims = [8, 16, 32]
    seeds_per_dim = 2 if quick else 3

    for dim in dims:
        for seed in range(seeds_per_dim):
            name = f"random_{dim}x{dim}_s{seed}"
            M = np.array(get_random_matrix(dim, dim, seed=dim * 100 + seed), dtype=np.int8)

            # Paar baseline
            paar = PaarAlgorithm()
            paar_circuit, paar_gates = paar.solve(M)

            # 多起点搜索 + 局部搜索
            n_paar_trials = 200 if quick else 500
            ms_time = 20 if quick else 60
            ms_circuit, ms_gates, _ = multi_start_search(
                M, n_paar=n_paar_trials, n_bp=0,
                time_limit=ms_time,
                local_search_top=5, local_search_time=5,
                verbose=False,
            )

            # ILS
            ils_time = 10 if quick else 30
            ils_circuit, ils_gates = iterated_local_search(
                M, num_inputs=dim, n_restarts=30 if quick else 100,
                time_limit=ils_time, verbose=False,
            )

            best = min(ms_gates, ils_gates)
            best_circuit = ms_circuit if ms_gates <= ils_gates else ils_circuit

            # 穷举局部搜索
            exh = exhaustive_local_search(best_circuit, M, dim, time_limit=15, verbose=False)
            exh_gates = len(exh)
            if exh_gates < best and verify_circuit(exh, M, dim):
                best = exh_gates

            improve = paar_gates - best
            pct = improve / paar_gates * 100 if paar_gates > 0 else 0

            random_results.append({
                'name': name, 'dim': dim,
                'paar': paar_gates, 'ours': best,
                'improve': improve, 'pct': pct,
            })
            print(f"  {name}: Paar={paar_gates}, Ours={best}, "
                  f"改进={improve} gates ({pct:.1f}%)")

    # 按维度汇总
    print(f"\n  --- 按矩阵规模汇总 ---")
    for dim in dims:
        subset = [r for r in random_results if r['dim'] == dim]
        avg_paar = np.mean([r['paar'] for r in subset])
        avg_ours = np.mean([r['ours'] for r in subset])
        avg_pct = np.mean([r['pct'] for r in subset])
        print(f"  {dim}x{dim}: Paar avg={avg_paar:.1f}, Ours avg={avg_ours:.1f}, "
              f"平均改进={avg_pct:.1f}%")

    all_exp['exp1_random'] = random_results

    # ================================================================
    # 实验 2: 密码学 benchmark
    # 目的: 在有已知 SOTA 的经典矩阵上定位本方法的水平
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  实验 2: 密码学 benchmark (Layer 0 + Layer 1)")
    print(f"{'='*70}")

    benchmarks = load_all_benchmarks()
    crypto_results = []

    for bname, info in benchmarks.items():
        T = np.array(info['matrix'], dtype=np.int8)
        n_inputs = info['dim']
        sota = info['sota_gates']

        print(f"\n  [{bname}] dim={n_inputs}, SOTA={sota}")

        # Paar baseline
        paar = PaarAlgorithm()
        paar_circuit, paar_gates = paar.solve(T)

        # Layer 0: 代数分解
        algebraic_results = layer0_algebraic(T, n_inputs, verbose=False)
        alg_best = algebraic_results[0][1] if algebraic_results else None

        # Layer 1: 多起点
        ms_circuit, ms_gates, _ = multi_start_search(
            T, n_paar=300 if quick else 800,
            n_bp=0,
            time_limit=30 if quick else 90,
            local_search_top=10, local_search_time=10,
            verbose=False,
        )
        ils_circuit, ils_gates = iterated_local_search(
            T, num_inputs=n_inputs,
            n_restarts=30 if quick else 100,
            time_limit=20 if quick else 60,
            verbose=False,
        )

        best = min(ms_gates, ils_gates)
        best_circuit = ms_circuit if ms_gates <= ils_gates else ils_circuit
        if alg_best and algebraic_results[0][1] < best:
            best = algebraic_results[0][1]
            best_circuit = algebraic_results[0][0]

        exh = exhaustive_local_search(best_circuit, T, n_inputs, time_limit=30, verbose=False)
        exh_gates = len(exh)
        if exh_gates < best and verify_circuit(exh, T, n_inputs):
            best = exh_gates

        res = {
            'name': bname, 'dim': n_inputs,
            'paar': paar_gates, 'ours': best,
            'sota': sota,
            'gap_to_paar': paar_gates - best,
            'gap_to_sota': best - sota if sota else None,
        }
        crypto_results.append(res)
        sota_str = f", SOTA差距={best - sota}" if sota else ""
        print(f"    Paar={paar_gates}, Ours={best}, 改进Paar={paar_gates - best}{sota_str}")

    all_exp['exp2_crypto'] = crypto_results

    # ================================================================
    # 实验 3: 消融实验 — 各组件贡献
    # 目的: 证明每一层都有贡献，不是堆砌
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  实验 3: 消融实验 (AES MixColumns)")
    print(f"{'='*70}")

    aes_info = benchmarks['aes_mixcolumns_32x32']
    T_aes = np.array(aes_info['matrix'], dtype=np.int8)
    n_aes = 32

    # (a) 仅 Paar
    paar = PaarAlgorithm()
    _, paar_g = paar.solve(T_aes)
    print(f"  (a) Paar only:              {paar_g} gates")

    # (b) Paar + 局部搜索
    ms_c, ms_g, _ = multi_start_search(
        T_aes, n_paar=300 if quick else 800, n_bp=0,
        time_limit=20 if quick else 60,
        local_search_top=5, local_search_time=5, verbose=False,
    )
    print(f"  (b) Paar + LocalSearch:     {ms_g} gates")

    # (c) Paar + ILS
    ils_c, ils_g = iterated_local_search(
        T_aes, num_inputs=n_aes, n_restarts=30 if quick else 100,
        time_limit=20 if quick else 60, verbose=False,
    )
    print(f"  (c) Paar + ILS:             {ils_g} gates")

    # (d) Paar + 穷举局部搜索
    best_c = ms_c if ms_g <= ils_g else ils_c
    best_g = min(ms_g, ils_g)
    exh_c = exhaustive_local_search(best_c, T_aes, n_aes, time_limit=30, verbose=False)
    exh_g = len(exh_c)
    if not verify_circuit(exh_c, T_aes, n_aes):
        exh_g = best_g
    print(f"  (d) + Exhaustive LS:        {exh_g} gates")

    # (e) 代数分解
    alg_res = layer0_algebraic(T_aes, n_aes, verbose=False)
    alg_g = alg_res[0][1] if alg_res else "N/A"
    print(f"  (e) Algebraic decomp:       {alg_g} gates")

    ablation = {
        'paar_only': paar_g,
        'paar_ls': ms_g,
        'paar_ils': ils_g,
        'exhaustive_ls': exh_g,
        'algebraic': alg_g,
    }
    all_exp['exp3_ablation'] = ablation

    # ================================================================
    # 实验 4: 可扩展性分析
    # 目的: 展示框架在不同规模下的表现和运行时间
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  实验 4: 可扩展性分析")
    print(f"{'='*70}")

    scale_results = []
    for dim in [8, 16, 32]:
        M = np.array(get_random_matrix(dim, dim, seed=42), dtype=np.int8)

        # Paar
        paar = PaarAlgorithm()
        t0 = time.time()
        _, paar_g = paar.solve(M)
        paar_time = time.time() - t0

        # 本框架 (Layer 1)
        t0 = time.time()
        ms_c, ms_g, _ = multi_start_search(
            M, n_paar=100 if quick else 300, n_bp=0,
            time_limit=10 if quick else 30,
            local_search_top=5, local_search_time=5, verbose=False,
        )
        ils_c, ils_g = iterated_local_search(
            M, num_inputs=dim, n_restarts=20 if quick else 50,
            time_limit=10 if quick else 20, verbose=False,
        )
        best_g = min(ms_g, ils_g)
        ours_time = time.time() - t0

        scale_results.append({
            'dim': dim, 'paar': paar_g, 'ours': best_g,
            'paar_time': round(paar_time, 2),
            'ours_time': round(ours_time, 2),
            'improve_pct': round((paar_g - best_g) / paar_g * 100, 1) if paar_g > 0 else 0,
        })
        print(f"  {dim}x{dim}: Paar={paar_g} ({paar_time:.1f}s), "
              f"Ours={best_g} ({ours_time:.1f}s), "
              f"改进={paar_g - best_g} ({scale_results[-1]['improve_pct']}%)")

    all_exp['exp4_scalability'] = scale_results

    # ================================================================
    # 实验 5: AlphaZero 收敛曲线 (仅在有 GPU 时)
    # 目的: 展示自博弈训练过程中 gate 数的下降
    # ================================================================
    if device != 'cpu':
        print(f"\n{'='*70}")
        print(f"  实验 5: AlphaZero 收敛曲线 (16x16 random)")
        print(f"{'='*70}")

        dim_az = 16
        M_az = np.array(get_random_matrix(dim_az, dim_az, seed=42), dtype=np.int8)

        max_extra = dim_az * 6
        env = SLPGraphEnv(M_az, max_extra, max_depth=10)
        env.reset()

        model = SLPPolicyValueNet(
            input_dim=env.feature_dim, hidden_dim=128,
            num_gnn_layers=4, num_heads=4, dropout=0.1,
        ).to(device)

        # 预训练
        training_circuits = []
        for seed in range(30 if quick else 80):
            rng = np.random.RandomState(seed)
            c, g = randomized_paar(M_az, rng)
            training_circuits.append((c, g))
        training_circuits.sort(key=lambda x: x[1])
        training_circuits = training_circuits[:20]

        model = pretrain_from_circuits(
            model, training_circuits, M_az, device,
            n_epochs=10, batch_size=32, lr=1e-3, verbose=False,
        )

        paar = PaarAlgorithm()
        _, paar_g = paar.solve(M_az)

        def ls_wrap(circuit, matrix, n_inp):
            return full_local_search(circuit, matrix, n_inp, time_limit=5)

        az_iters = 3 if quick else 8
        az_best, az_circuit, az_history = alpha_zero_loop(
            M_az, model, device,
            n_iterations=az_iters,
            n_games_per_iter=3 if quick else 8,
            n_simulations=100 if quick else 300,
            max_children=30, max_extra=max_extra, max_depth=10,
            best_known=paar_g,
            train_epochs=3, train_batch_size=32, lr=1e-4,
            temperature=1.0, temp_threshold=10,
            local_search_fn=ls_wrap,
            save_dir=None, verbose=True,
        )

        convergence = {
            'paar_baseline': paar_g,
            'az_final': az_best,
            'history': az_history,
            'improve': paar_g - az_best if az_best else 0,
        }
        all_exp['exp5_convergence'] = convergence
        print(f"  Paar={paar_g}, AlphaZero best={az_best}, "
              f"改进={paar_g - az_best if az_best else 0}")
    else:
        print(f"\n  实验 5: 跳过 (需要 GPU, 用 --gpu 启用)")
        all_exp['exp5_convergence'] = {'skipped': True, 'reason': 'no GPU'}

    # ================================================================
    # 总结
    # ================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  实验总结")
    print(f"{'='*70}")
    print(f"  总耗时: {total_time:.0f}s ({total_time/60:.1f} min)")

    if 'exp1_random' in all_exp:
        avg_pct = np.mean([r['pct'] for r in all_exp['exp1_random']])
        print(f"  随机矩阵平均改进: {avg_pct:.1f}% over Paar")

    if 'exp4_scalability' in all_exp:
        for r in all_exp['exp4_scalability']:
            print(f"  {r['dim']}x{r['dim']}: 改进 {r['improve_pct']}%")

    # 保存
    save_path = os.path.join(os.path.dirname(__file__), 'experiment_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_exp, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  结果已保存: {save_path}")

    return all_exp


def main():
    parser = argparse.ArgumentParser(description='SLP Optimizer: GNN + MCTS 混合框架')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--bench', type=str, default=None,
                        help='Specific benchmark (e.g., aes_mixcolumns_32x32)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with reduced search')
    parser.add_argument('--layer1-only', action='store_true',
                        help='Only run Layer 0+1 (algebraic + multi-start + local search)')
    parser.add_argument('--experiment', action='store_true',
                        help='Run full experiment suite for report')
    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.experiment:
        run_experiment_suite(device=device, quick=args.quick)
        return

    benchmarks = load_all_benchmarks()

    if args.bench:
        if args.bench not in benchmarks:
            print(f"Unknown benchmark: {args.bench}")
            print(f"Available: {list(benchmarks.keys())}")
            return
        targets = {args.bench: benchmarks[args.bench]}
    else:
        targets = benchmarks

    all_results = {}

    for name, info in targets.items():
        T = np.array(info['matrix'], dtype=np.int8)
        n_inputs = info['dim']
        sota = info['sota_gates']

        print(f"\n{'='*60}")
        print(f"Benchmark: {name} ({n_inputs}x{n_inputs}), SOTA={sota}")
        print(f"{'='*60}")

        if args.layer1_only:
            # Layer 0 + Layer 1
            algebraic_results = layer0_algebraic(T, n_inputs, verbose=True)

            ms_circuit, ms_gates, ms_stats = multi_start_search(
                T, n_paar=5000 if not args.quick else 1000,
                n_bp=2000 if not args.quick else 500,
                time_limit=300 if not args.quick else 60,
                local_search_top=50, local_search_time=30,
            )
            ils_circuit, ils_gates = iterated_local_search(
                T, num_inputs=n_inputs,
                n_restarts=200 if not args.quick else 50,
                time_limit=180 if not args.quick else 30,
            )
            best = min(ms_gates, ils_gates)
            best_circuit = ms_circuit if ms_gates <= ils_gates else ils_circuit

            if algebraic_results:
                alg_gates = algebraic_results[0][1]
                if alg_gates < best:
                    best = alg_gates
                    best_circuit = algebraic_results[0][0]

            # Exhaustive LS on best
            exh = exhaustive_local_search(best_circuit, T, n_inputs, time_limit=120)
            exh_gates = len(exh)
            if exh_gates < best and verify_circuit(exh, T, n_inputs):
                best = exh_gates

            results = {
                'best_gates': best,
                'multistart_gates': ms_gates,
                'ils_gates': ils_gates,
                'algebraic_gates': algebraic_results[0][1] if algebraic_results else None,
                'sota': sota,
            }
        else:
            results = optimize_pipeline(
                T, n_inputs, sota_gates=sota, device=device,
                quick=args.quick, verbose=True,
            )

        all_results[name] = results

        print(f"\n  FINAL: {results['best_gates']} gates")
        if sota:
            gap = results['best_gates'] - sota
            if gap <= 0:
                print(f"  ★★★ MATCHES OR BEATS SOTA ({sota})! ★★★")
            else:
                print(f"  Gap to SOTA ({sota}): {gap} gates")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'Benchmark':<20} {'Paar':>6} {'Ours':>6} {'SOTA':>6} {'Gap':>6}")
    print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for name, res in all_results.items():
        paar = res.get('paar_gates', '?')
        ours = res['best_gates']
        sota = res.get('sota', None)
        gap = f"{ours - sota:+d}" if sota else "?"
        sota_str = str(sota) if sota else "?"
        print(f"{name:<20} {paar:>6} {ours:>6} {sota_str:>6} {gap:>6}")

    # 保存结果
    save_path = os.path.join(os.path.dirname(__file__), 'slp_optimizer_results.json')
    serializable = {}
    for name, res in all_results.items():
        s = {k: v for k, v in res.items()
             if k not in ('best_circuit', 'az_history', 'multistart_stats')}
        if 'az_history' in res and res['az_history']:
            s['az_final_iter'] = res['az_history'][-1] if res['az_history'] else None
        serializable[name] = s

    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
