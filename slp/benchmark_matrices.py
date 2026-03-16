"""
Benchmark 矩阵集合：覆盖密码学核心场景
"""
import numpy as np


# ==========================================
# 工具函数：GF(2^8) 矩阵乘法展开
# ==========================================
def gf2_multiply_matrix(coeff):
    """
    将 GF(2^8) 中乘以 coeff 展开为 8x8 二进制矩阵。
    不可约多项式: x^8 + x^4 + x^3 + x + 1 (0x11B)
    """
    M = np.zeros((8, 8), dtype=np.int8)
    val = coeff
    for col in range(8):
        bits = val
        for row in range(8):
            M[row, col] = bits & 1
            bits >>= 1
        # xtime: val = val << 1, 若溢出则 XOR 0x1B
        val <<= 1
        if val & 0x100:
            val ^= 0x11B
        val &= 0xFF
    # 重新生成：逐列表示 coeff * x^col 在 GF(2^8) 中的结果
    M = np.zeros((8, 8), dtype=np.int8)
    for col in range(8):
        result = _gf2_mul(coeff, 1 << col)
        for row in range(8):
            M[row, col] = (result >> row) & 1
    return M


def _gf2_mul(a, b):
    """GF(2^8) 乘法，模 x^8+x^4+x^3+x+1"""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return p


def _build_block_circulant(coeffs, block_size=4):
    """
    构建分块循环矩阵。
    coeffs: [c0, c1, c2, c3] 表示一行的 GF(2^8) 系数
    返回 (block_size*8) x (block_size*8) 的二进制矩阵
    """
    n = block_size
    dim = n * 8
    M = np.zeros((dim, dim), dtype=np.int8)
    for row_block in range(n):
        for col_block in range(n):
            coeff = coeffs[(col_block - row_block) % n]
            sub = gf2_multiply_matrix(coeff)
            M[row_block*8:(row_block+1)*8, col_block*8:(col_block+1)*8] = sub
    return M


# ==========================================
# 1. AES MixColumns (32x32)
# ==========================================
def get_aes_mixcolumns_matrix():
    """
    AES MixColumns 的 32x32 二进制矩阵。
    系数矩阵 (GF(2^8)):
    [2 3 1 1]
    [1 2 3 1]
    [1 1 2 3]
    [3 1 1 2]
    """
    coeffs = [2, 3, 1, 1]
    return _build_block_circulant(coeffs, 4).tolist()


# ==========================================
# 2. AES InvMixColumns (32x32)
# ==========================================
def get_aes_inv_mixcolumns_matrix():
    """
    AES InvMixColumns 的 32x32 二进制矩阵。
    系数矩阵 (GF(2^8)):
    [0x0E 0x0B 0x0D 0x09]
    [0x09 0x0E 0x0B 0x0D]
    [0x0D 0x09 0x0E 0x0B]
    [0x0B 0x0D 0x09 0x0E]
    """
    coeffs = [0x0E, 0x0B, 0x0D, 0x09]
    return _build_block_circulant(coeffs, 4).tolist()


# ==========================================
# 3. Camellia P-function (64x64)
# ==========================================
def get_camellia_p_matrix():
    """
    Camellia 的 P-function 作为 64x64 二进制矩阵。
    Camellia 使用 8 个 8-bit S-box 输出通过 P-function 混合。
    P-function 定义:
      z1 = x1 ^ x3 ^ x4 ^ x6 ^ x7 ^ x8
      z2 = x1 ^ x2 ^ x4 ^ x5 ^ x7 ^ x8
      z3 = x1 ^ x2 ^ x3 ^ x5 ^ x6 ^ x8
      z4 = x2 ^ x3 ^ x4 ^ x5 ^ x6 ^ x7
      z5 = x1 ^ x2 ^ x6 ^ x7 ^ x8
      (... 类似模式，此处使用完整 Camellia 规范)

    这里构建的是 Camellia 字节级别的线性混合层。
    """
    # Camellia P-function 的字节级 XOR 混合模式
    # 每一行表示输出字节 zi 由哪些输入字节 xj 参与 XOR
    P_byte = np.array([
        [1, 0, 1, 1, 0, 1, 1, 1],  # z1
        [1, 1, 0, 1, 1, 0, 1, 1],  # z2
        [1, 1, 1, 0, 1, 1, 0, 1],  # z3
        [0, 1, 1, 1, 1, 1, 1, 0],  # z4
        [1, 1, 0, 0, 0, 1, 1, 1],  # z5
        [0, 1, 1, 0, 1, 0, 1, 1],  # z6
        [0, 0, 1, 1, 1, 1, 0, 1],  # z7
        [1, 0, 0, 1, 1, 1, 1, 0],  # z8
    ], dtype=np.int8)

    # 展开为 64x64: 每个字节是 8 bit，P_byte[i][j]=1 表示 I_8 块
    I8 = np.eye(8, dtype=np.int8)
    O8 = np.zeros((8, 8), dtype=np.int8)

    M = np.zeros((64, 64), dtype=np.int8)
    for i in range(8):
        for j in range(8):
            block = I8 if P_byte[i, j] == 1 else O8
            M[i*8:(i+1)*8, j*8:(j+1)*8] = block

    return M.tolist()


# ==========================================
# 4. PRESENT pLayer (64x64)
# ==========================================
def get_present_player_matrix():
    """
    PRESENT 密码的 pLayer (比特置换层) 作为 64x64 置换矩阵。
    置换规则: P(i) = 16*i mod 63, for i=0..62; P(63)=63
    """
    M = np.zeros((64, 64), dtype=np.int8)
    for i in range(64):
        if i == 63:
            j = 63
        else:
            j = (16 * i) % 63
        M[j, i] = 1
    return M.tolist()


# ==========================================
# 5. Midori-like (16x16)
# ==========================================
def get_midori_16x16_matrix():
    """类 Midori 的 16x16 轻量级二元矩阵"""
    I4 = np.eye(4, dtype=np.int8)
    O4 = np.zeros((4, 4), dtype=np.int8)
    P = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ], dtype=np.int8)

    row1 = np.hstack([O4, P, I4, I4])
    row2 = np.hstack([I4, O4, P, I4])
    row3 = np.hstack([I4, I4, O4, P])
    row4 = np.hstack([P, I4, I4, O4])

    return np.vstack([row1, row2, row3, row4]).tolist()


# ==========================================
# 6. 随机矩阵生成器
# ==========================================
def get_random_matrix(n_rows, n_cols, seed=None):
    """生成随机二进制矩阵（保证无全零行）"""
    rng = np.random.RandomState(seed)
    while True:
        M = rng.randint(0, 2, size=(n_rows, n_cols)).astype(np.int8)
        if np.all(np.any(M, axis=1)):
            return M.tolist()


def get_random_matrix_suite(seeds_per_size=5):
    """生成一组随机矩阵用于泛化性测试"""
    suite = []
    for dim in [8, 16, 32, 64]:
        for seed in range(seeds_per_size):
            m = get_random_matrix(dim, dim, seed=dim * 100 + seed)
            suite.append({
                'name': f'random_{dim}x{dim}_s{seed}',
                'matrix': m,
                'dim': dim,
            })
    return suite


# ==========================================
# 汇总所有 benchmark
# ==========================================
BENCHMARKS = {
    'midori_16x16': {
        'fn': get_midori_16x16_matrix,
        'dim': 16,
        'sota_gates': None,
        'description': '轻量级密码 Midori-like 扩散层',
    },
    'aes_mixcolumns_32x32': {
        'fn': get_aes_mixcolumns_matrix,
        'dim': 32,
        'sota_gates': 103,
        'description': 'AES MixColumns (核心 benchmark)',
    },
    'aes_inv_mixcolumns_32x32': {
        'fn': get_aes_inv_mixcolumns_matrix,
        'dim': 32,
        'sota_gates': 105,
        'description': 'AES InvMixColumns (逆运算)',
    },
    'camellia_p_64x64': {
        'fn': get_camellia_p_matrix,
        'dim': 64,
        'sota_gates': None,
        'description': 'Camellia P-function (大规模测试)',
    },
    'present_player_64x64': {
        'fn': get_present_player_matrix,
        'dim': 64,
        'sota_gates': None,
        'description': 'PRESENT pLayer (轻量级密码)',
    },
}


def load_all_benchmarks():
    """加载全部 benchmark 矩阵"""
    result = {}
    for name, info in BENCHMARKS.items():
        matrix = info['fn']()
        result[name] = {
            'matrix': matrix,
            'dim': info['dim'],
            'sota_gates': info['sota_gates'],
            'description': info['description'],
        }
    return result


if __name__ == "__main__":
    benchmarks = load_all_benchmarks()
    for name, info in benchmarks.items():
        m = np.array(info['matrix'])
        print(f"{name}: {m.shape}, ones={np.sum(m)}, sota={info['sota_gates']}")

    # 随机矩阵
    suite = get_random_matrix_suite()
    print(f"\nRandom suite: {len(suite)} matrices")
    for s in suite[:3]:
        print(f"  {s['name']}: {s['dim']}x{s['dim']}")
