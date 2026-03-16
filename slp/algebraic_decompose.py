"""
AES MixColumns 代数分解优化。

核心思想 (Maximov 2019 风格):
1. AES MixColumns 是 GF(2^8) 上的循环矩阵 [2,3,1,1]
2. 3 = 2 XOR 1，所以 3*x = 2*x XOR x
3. 2*x (xtime) 在 bit-level 是线性移位 + 条件 XOR 0x1B
4. 循环矩阵意味着 4 行共享大量子表达式
5. 先在 byte-level 找共享子表达式，再在 bit-level 优化 xtime

目标: 利用代数结构产生比 Paar(108) 更优的初始电路，
      作为局部搜索和 AlphaZero 的更好起点。
"""
import numpy as np
from local_search import verify_circuit, full_local_search, reconstruct_basis


def get_xtime_circuit(byte_offset, num_inputs=32):
    """
    xtime (乘以 2 in GF(2^8)) 的 bit-level 电路。
    输入: 8 bits at positions [byte_offset*8 .. byte_offset*8+7]
    即 b7 b6 b5 b4 b3 b2 b1 b0

    xtime(b) = b << 1 XOR (0x1B if b7 else 0)
    bit-level:
      r0 = b7           (from reduction)
      r1 = b0 XOR b7    (shift + reduction)
      r2 = b1
      r3 = b2 XOR b7    (reduction)
      r4 = b3 XOR b7    (reduction)
      r5 = b4
      r6 = b5
      r7 = b6

    Returns: dict mapping output bit positions to either:
      - input index (no gate needed)
      - (input_a, input_b) XOR gate needed
    """
    base = byte_offset * 8
    b = [base + i for i in range(8)]  # b[0]=LSB, b[7]=MSB

    # xtime result bits
    result = {}
    result[0] = b[7]              # r0 = b7
    result[1] = (b[0], b[7])      # r1 = b0 XOR b7
    result[2] = b[1]              # r2 = b1
    result[3] = (b[2], b[7])      # r3 = b2 XOR b7
    result[4] = (b[3], b[7])      # r4 = b3 XOR b7
    result[5] = b[4]              # r5 = b4
    result[6] = b[5]              # r6 = b5
    result[7] = b[6]              # r7 = b6
    return result


def algebraic_aes_mixcolumns(verbose=True):
    """
    用代数分解构造 AES MixColumns 电路。

    AES MixColumns 对 4 字节 (a, b, c, d) 计算:
      s0 = 2a + 3b + c + d = 2a + 2b + b + c + d = 2(a+b) + (b+c+d)
      s1 = a + 2b + 3c + d = 2(b+c) + (a+c+d)
      s2 = a + b + 2c + 3d = 2(c+d) + (a+b+d)
      s3 = 3a + a + b + 2d = 2(d+a) + (a+b+c)  [循环]

    关键子表达式共享:
      - p = a+b, q = b+c, r = c+d, s = d+a  (4 个 byte-XOR，每个 8 gates = 32 gates)
        但注意 p+q = a+c, q+r = b+d, p+r = a+d+b+c...
        实际上 s = p + q + r (因为 a+d = (a+b)+(b+c)+(c+d))
        所以只需 p, q, r, 然后 s = p XOR q XOR r (省 8 gates!)

      - t = b+c+d = q+d = r+b, 但也 = a + (a+b+c+d)
        a+b+c+d = p + r = q + s
        所以: b+c+d = a + (p+r), a+c+d = b + (p+r), etc.
        令 sum4 = a+b+c+d = p + r (8 gates)

      - 2(a+b) = xtime(p): 需要约 4 XOR gates (因为 xtime 只在 4 位上需要 XOR)
        类似 2(b+c) = xtime(q), 2(c+d) = xtime(r), 2(d+a) = xtime(s)

    总门数估算:
      byte-XOR p,q,r: 3 * 8 = 24 gates
      s = p XOR q XOR r: 2 * 8 = 16 gates (分两步)
      sum4 = p XOR r: 8 gates
      b+c+d = sum4 XOR a: 8 gates
      a+c+d = sum4 XOR b: 8 gates
      a+b+d = sum4 XOR c: 8 gates
      a+b+c = sum4 XOR d: 8 gates
        子总: 24 + 16 + 8 + 32 = 80 gates
      xtime(p), xtime(q), xtime(r), xtime(s): 每个 4 gates = 16 gates
      s0 = xtime(p) + (b+c+d): 8 gates
      s1 = xtime(q) + (a+c+d): 8 gates
      s2 = xtime(r) + (a+b+d): 8 gates
      s3 = xtime(s) + (a+b+c): 8 gates
        子总: 16 + 32 = 48 gates

    粗估: ~128 gates (比 Paar 的 108 差!)

    优化 1: xtime 共享 b7
      xtime 需要 XOR b7 到 3 个位 (bit1, bit3, bit4)
      但 b7 是输入，不需要额外门。实际 xtime 每个需要 3 XOR (不是 4)
      → 4 * 3 = 12 gates

    优化 2: sum4 共享
      b+c+d = a XOR sum4, 但我们也可以写 b+c+d = q XOR d (只要 d 可用)
      这省了 sum4 的中间步骤...

    让我们换个更紧凑的分解。

    更优分解 (Maximov 风格):
      定义:
        t = a XOR b XOR c XOR d  (全和)
      则:
        s0 = t XOR a XOR 2(a XOR b) = t XOR a XOR xtime(a XOR b)
        因为: 2a+3b+c+d = 2a+2b+b+c+d = 2(a+b) + (a + t) [因为 b+c+d = a+t]
              = xtime(a+b) + a + t

      类似:
        s0 = xtime(a+b) + a + t
        s1 = xtime(b+c) + b + t
        s2 = xtime(c+d) + c + t
        s3 = xtime(d+a) + d + t

      门数:
        a+b: 8, b+c: 8, c+d: 8, d+a = (a+b)+(b+c)+(c+d): 16
          但 d+a 可以从已有的算: d+a = t + (b+c) = ... 不对
          d+a = (a+b) XOR (b+c) XOR (c+d) 不对!
          a+d = a+b + b+c + c+d 当且仅当 GF(2): a+d = (a+b)+(b+c)+(c+d) = a+2b+2c+d = a+d ✓

        更好: a+b=p, b+c=q, c+d=r
          d+a = p XOR q XOR r (需要 2*8 = 16 gates? 不, p XOR q 是 8 gates, 再 XOR r 是 8 gates)
          但实际我们可以复用: p XOR q = a XOR c, 然后 (a XOR c) XOR r = a XOR d
          这需要 8 + 8 = 16 gates

        或者: t = a+b+c+d. a+d = t + b + c = t + q
          t 需要: p + r = 8 gates (因为 p = a+b, r = c+d)
          a+d = t + q = 8 gates
          总共用 8 + 8 = 16 gates 算 t 和 a+d

        好，继续:
          p = a+b: 8 gates
          q = b+c: 8 gates
          r = c+d: 8 gates
          t = p + r: 8 gates  (= a+b+c+d)
          s = t + q: 8 gates  (= a+d, 因为 a+b+c+d + b+c = a+d)
            等等不对! t + q = (a+b+c+d) + (b+c) = a+d ✓

          byte-level sums: p, q, r, t, s 用了 5*8 = 40 gates

          xtime(p): 3 gates (XOR b7 to bits 1,3,4)
          xtime(q): 3 gates
          xtime(r): 3 gates
          xtime(s): 3 gates
          xtime 共 12 gates

          a+t: 8 gates
          b+t: 8 gates
          c+t: 8 gates
          d+t: 8 gates
          或者利用 a+t = b+c+d, b+t = a+c+d, etc.

          s0 = xtime(p) + a + t = xtime(p) + (bcd)
          s1 = xtime(q) + b + t = xtime(q) + (acd)
          s2 = xtime(r) + c + t = xtime(r) + (abd)
          s3 = xtime(s) + d + t = xtime(s) + (abc)

          bcd = a+t: 8 gates
          acd = b+t: 8 gates
          abd = c+t: 8 gates
          abc = d+t: 8 gates
          → 32 gates

          最终 XOR:
          s0 = xtime(p) + bcd: 8 gates
          s1 = xtime(q) + acd: 8 gates
          s2 = xtime(r) + abd: 8 gates
          s3 = xtime(s) + abc: 8 gates
          → 32 gates

        总: 40 + 12 + 32 + 32 = 116 gates

        这比 108 还差! 需要更多优化...

    让我实际按照 Maximov 的方法编码，然后用局部搜索优化。
    关键: 产生一个 algebraically-structured 的初始电路，
          即使它不是最优的，局部搜索可能比从 Paar 开始压缩得更多，
          因为结构不同。

    Returns: (circuit, n_gates) or None
    """
    # 32 inputs: bytes a[0:8], b[8:16], c[16:24], d[24:32]
    n_inputs = 32

    # 我们用 basis 列表跟踪所有向量，circuit 记录 (u,v) 对
    basis = []
    for i in range(n_inputs):
        vec = np.zeros(n_inputs, dtype=np.int8)
        vec[i] = 1
        basis.append(vec)
    circuit = []

    def xor_gate(u, v):
        """添加 XOR 门，返回新节点索引"""
        new_vec = (basis[u].astype(np.int16) + basis[v].astype(np.int16)) % 2
        new_vec = new_vec.astype(np.int8)
        idx = len(basis)
        basis.append(new_vec)
        circuit.append((u, v))
        return idx

    def byte_xor(byte_a_start, byte_b_start):
        """对两个 byte 做逐位 XOR，返回 8 个新节点索引"""
        results = []
        for bit in range(8):
            idx = xor_gate(byte_a_start + bit, byte_b_start + bit)
            results.append(idx)
        return results

    def byte_xor_nodes(nodes_a, nodes_b):
        """对两组 8 个节点做逐位 XOR"""
        results = []
        for bit in range(8):
            idx = xor_gate(nodes_a[bit], nodes_b[bit])
            results.append(idx)
        return results

    def xtime_nodes(byte_nodes):
        """
        xtime: 乘以 2 in GF(2^8), modulo x^8+x^4+x^3+x+1
        输入 byte_nodes = [b0, b1, ..., b7] (LSB first)
        输出:
          r0 = b7
          r1 = b0 ^ b7
          r2 = b1
          r3 = b2 ^ b7
          r4 = b3 ^ b7
          r5 = b4
          r6 = b5
          r7 = b6
        只需 3 个 XOR 门 (bits 1, 3, 4 需要 XOR b7)
        """
        b = byte_nodes  # [b0..b7]
        r = [0] * 8
        r[0] = b[7]            # 无门
        r[1] = xor_gate(b[0], b[7])  # 1 gate
        r[2] = b[1]            # 无门
        r[3] = xor_gate(b[2], b[7])  # 1 gate
        r[4] = xor_gate(b[3], b[7])  # 1 gate
        r[5] = b[4]            # 无门
        r[6] = b[5]            # 无门
        r[7] = b[6]            # 无门
        return r

    # 输入字节
    a = list(range(0, 8))    # byte 0
    b = list(range(8, 16))   # byte 1
    c = list(range(16, 24))  # byte 2
    d = list(range(24, 32))  # byte 3

    if verbose:
        print("  Algebraic decomposition:")

    # Step 1: Pairwise byte XORs
    # p = a+b (8 gates)
    p = byte_xor_nodes(a, b)
    if verbose:
        print(f"    p = a+b: {len(circuit)} total gates")

    # q = b+c (8 gates)
    q = byte_xor_nodes(b, c)
    if verbose:
        print(f"    q = b+c: {len(circuit)} total gates")

    # r = c+d (8 gates)
    r = byte_xor_nodes(c, d)
    if verbose:
        print(f"    r = c+d: {len(circuit)} total gates")

    # t = a+b+c+d = p + r (8 gates)
    t = byte_xor_nodes(p, r)
    if verbose:
        print(f"    t = a+b+c+d: {len(circuit)} total gates")

    # s = d+a = t + q (because (a+b+c+d)+(b+c) = a+d) (8 gates)
    s = byte_xor_nodes(t, q)
    if verbose:
        print(f"    s = d+a: {len(circuit)} total gates")

    # Step 2: xtime of pairwise sums (3 gates each)
    xt_p = xtime_nodes(p)  # 2(a+b)
    xt_q = xtime_nodes(q)  # 2(b+c)
    xt_r = xtime_nodes(r)  # 2(c+d)
    xt_s = xtime_nodes(s)  # 2(d+a)
    if verbose:
        print(f"    xtime(p,q,r,s): {len(circuit)} total gates")

    # Step 3: t XOR each input byte (= sum of other 3 bytes)
    # b+c+d = t + a (8 gates)
    bcd = byte_xor_nodes(t, a)
    # a+c+d = t + b (8 gates)
    acd = byte_xor_nodes(t, b)
    # a+b+d = t + c (8 gates)
    abd = byte_xor_nodes(t, c)
    # a+b+c = t + d (8 gates)
    abc = byte_xor_nodes(t, d)
    if verbose:
        print(f"    3-byte sums: {len(circuit)} total gates")

    # Step 4: Final outputs
    # s0 = xtime(p) + bcd = 2(a+b) + (b+c+d) (8 gates)
    s0 = byte_xor_nodes(xt_p, bcd)
    # s1 = xtime(q) + acd = 2(b+c) + (a+c+d) (8 gates)
    s1 = byte_xor_nodes(xt_q, acd)
    # s2 = xtime(r) + abd = 2(c+d) + (a+b+d) (8 gates)
    s2 = byte_xor_nodes(xt_r, abd)
    # s3 = xtime(s) + abc = 2(d+a) + (a+b+c) (8 gates)
    s3 = byte_xor_nodes(xt_s, abc)

    n_gates = len(circuit)
    if verbose:
        print(f"    Final outputs: {n_gates} total gates")

    return circuit, n_gates


def algebraic_aes_v2(verbose=True):
    """
    更紧凑的代数分解 v2。

    核心优化: 共享更多中间值。

    s0[i] = 2(a+b)[i] + b[i] + c[i] + d[i]
          = 2(a+b)[i] + (a+b)[i] + a[i] + c[i] + d[i]
          = 3(a+b)[i] + a[i] + (c+d)[i]
          = (2(a+b) + (a+b))[i] + a[i] + r[i]

    等等，3*x = xtime(x) XOR x，所以:
      s0 = 3*p + a + r  其中 p=a+b, r=c+d

    3*p = xtime(p) XOR p
    所以 s0 = xtime(p) + p + a + r = xtime(p) + b + r  (因为 p+a = b)
    验证: xtime(a+b) + b + c + d ✓ (因为 2(a+b)+3b+c+d = 2a+2b+2b+b+c+d = 2a+b+c+d... 不对)

    重新推导:
      s0 = 2a + 3b + c + d
      3b = 2b + b = xtime(b) + b
      s0 = 2a + xtime(b) + b + c + d
      s0 = xtime(a) + xtime(b) + (xtime(a) + 2a = 0 不对...)

    回到基础:
      s0 = 2a + 3b + c + d = 2a + (2+1)b + c + d = 2(a+b) + b + c + d
      s1 = a + 2b + 3c + d = 2(b+c) + a + c + d
      s2 = a + b + 2c + 3d = 2(c+d) + a + b + d
      s3 = 3a + a + b + 2d = 2(d+a) + a + b + c  [等等,  3a + b + c + 2d]
        s3 = 3a + b + c + 2d = (2+1)a + b + c + 2d = 2(a+d) + a + b + c

    OK 就是之前的结论。让我尝试另一个分解:

    定义 u = a+c, v = b+d (cross sums)
    t = u + v = a+b+c+d

    s0 = 2a + 3b + c + d
       = 2a + 2b + b + c + d
       = 2(a+b) + b + c + d
       这和之前一样...

    尝试 Maximov 的实际技巧: 先计算所有 xtime，再利用 xtime 的共性。

    xtime(a), xtime(b), xtime(c), xtime(d): 各 3 gates = 12 gates

    s0 = xtime(a) + xtime(b) + b + c + d
       = xtime(a) + xtime(b) + (b+c+d)
    s1 = xtime(b) + xtime(c) + (a+c+d)
    s2 = xtime(c) + xtime(d) + (a+b+d)
    s3 = xtime(d) + xtime(a) + (a+b+c)

    验证 s0: 2a + 2b + b + c + d = 2a + 3b + c + d ✓

    子表达式:
      xa = xtime(a): 3 gates
      xb = xtime(b): 3 gates
      xc = xtime(c): 3 gates
      xd = xtime(d): 3 gates

      xa_xb = xa + xb: 8 gates
      xb_xc = xb + xc: 8 gates
      xc_xd = xc + xd: 8 gates
      xd_xa = xd + xa: 8 gates
      但 xd_xa = xa_xb + xb_xc + xc_xd (类似之前)

      t = a+b+c+d
      bcd = t+a, acd = t+b, abd = t+c, abc = t+d

      方案 A:
        t 用 (a+b) + (c+d) = 8 + 8 + 8 = ... 需要先算 a+b 和 c+d
        不一定需要，t = a+c + b+d = ...

      这个分解的优势: xtime(a) 是 3 gates per byte，
      而之前 xtime(a+b) 也是 3 gates per byte，
      所以 xtime 部分门数相同。

      区别在于: 之前需要 p,q,r,s,t 共 5 个 byte-XOR (40 gates)
      这个需要 xa+xb, xb+xc, xc+xd (24 gates for 3, 第4个可由前3个推出 16 gates)
        加 t 及其衍生 (a+b: 8, c+d: 8, t=8, bcd=8, acd=8, abd=8, abc=8 = 56 gates)
      总 xtime pairs + sums: 40 + 56 = 96... 更差

    结论: 前面 v1 的分解已经是比较合理的了。
    问题是 ~116 gates 比 Paar 的 108 还多。
    但局部搜索可能有不同效果，因为电路拓扑完全不同。

    让我尝试更激进的共享。

    v2 关键优化:
    1. 不单独算 bcd, acd, abd, abc，而是利用已有的 p, q, r
       bcd = b + r = b + (c+d) → 用 r 直接加 b  (已有 r)
       acd = a + r + ... 不对  acd = a+c+d, 而 r = c+d, 所以 acd = a + r ✓ (8 gates)
       abd = a+b+d, p = a+b, 所以 abd = p + d ✓ (8 gates)
       abc = a+b+c, 而 abc = p + c ✓ (8 gates)
       bcd = b+c+d = q + d ✓ (8 gates) [q = b+c]

    这和之前用 t+x 一样都是 32 gates...但省了 t 的 8 gates!

    不: 之前需要 s = d+a 也需要 8 gates (或者 16 通过 p+q+r)。
    如果不需要 t:
      p = a+b: 8
      q = b+c: 8
      r = c+d: 8
      s = d+a: 需要算。可以从 p+q+r? 不: p+q+r = a+b+b+c+c+d = a+d ✓ 需要 16 gates (p+q=8, then +r=8)
      或者直接 a+d: 8 gates (从原始输入)

      如果直接从输入算 s = byte_xor(a, d): 8 gates
      总 pairwise: 4*8 = 32 gates

      bcd = q + d: 8
      acd = a + r: 8  (不对: a + r = a + c + d = acd ✓)
      abd = p + d: 8
      abc = p + c: 8
      3-byte sums: 32 gates

      xtime: 4*3 = 12 gates
      final xor: 4*8 = 32 gates

      总: 32 + 32 + 12 + 32 = 108 gates!

    等等... 这正好是 108! 和 Paar 一样!
    但结构完全不同 — 这是一个 algebraically structured 的 108-gate 电路。
    局部搜索在这个结构上可能找到不同的优化路径!

    更进一步: 能否省更多?
    - xtime 共享: 4 个 xtime 都需要 XOR 最高位。如果有重复...
      xtime(p) 和 xtime(q) 的 b7 不同，没法共享
    - 但 bcd = q + d 里，如果 d 的某些位恰好可以和 xtime 共享...

    让我试试用重叠来省门。
    """
    n_inputs = 32
    basis = []
    for i in range(n_inputs):
        vec = np.zeros(n_inputs, dtype=np.int8)
        vec[i] = 1
        basis.append(vec)
    circuit = []

    def xor_gate(u, v):
        new_vec = (basis[u].astype(np.int16) + basis[v].astype(np.int16)) % 2
        new_vec = new_vec.astype(np.int8)
        # Check if this vector already exists
        for k in range(len(basis)):
            if np.array_equal(basis[k], new_vec):
                return k  # Reuse existing node, no new gate!
        idx = len(basis)
        basis.append(new_vec)
        circuit.append((u, v))
        return idx

    def byte_xor_nodes(nodes_a, nodes_b):
        results = []
        for bit in range(8):
            idx = xor_gate(nodes_a[bit], nodes_b[bit])
            results.append(idx)
        return results

    def xtime_nodes(byte_nodes):
        b = byte_nodes
        r = [0] * 8
        r[0] = b[7]
        r[1] = xor_gate(b[0], b[7])
        r[2] = b[1]
        r[3] = xor_gate(b[2], b[7])
        r[4] = xor_gate(b[3], b[7])
        r[5] = b[4]
        r[6] = b[5]
        r[7] = b[6]
        return r

    a = list(range(0, 8))
    b = list(range(8, 16))
    c = list(range(16, 24))
    d = list(range(24, 32))

    if verbose:
        print("  Algebraic v2 (with dedup):")

    # Pairwise sums (directly from inputs)
    p = byte_xor_nodes(a, b)   # a+b
    q = byte_xor_nodes(b, c)   # b+c
    r = byte_xor_nodes(c, d)   # c+d
    s = byte_xor_nodes(d, a)   # d+a (直接从输入)
    if verbose:
        print(f"    Pairwise sums: {len(circuit)} gates")

    # xtime of pairwise sums
    xt_p = xtime_nodes(p)
    xt_q = xtime_nodes(q)
    xt_r = xtime_nodes(r)
    xt_s = xtime_nodes(s)
    if verbose:
        print(f"    + xtime: {len(circuit)} gates")

    # 3-byte sums using pairwise sums
    bcd = byte_xor_nodes(q, d)   # (b+c) + d = b+c+d
    acd = byte_xor_nodes(a, r)   # a + (c+d) = a+c+d
    abd = byte_xor_nodes(p, d)   # (a+b) + d = a+b+d
    abc = byte_xor_nodes(p, c)   # (a+b) + c = a+b+c
    if verbose:
        print(f"    + 3-byte sums: {len(circuit)} gates")

    # Final outputs
    s0 = byte_xor_nodes(xt_p, bcd)  # 2(a+b) + b+c+d
    s1 = byte_xor_nodes(xt_q, acd)  # 2(b+c) + a+c+d
    s2 = byte_xor_nodes(xt_r, abd)  # 2(c+d) + a+b+d
    s3 = byte_xor_nodes(xt_s, abc)  # 2(d+a) + a+b+c
    if verbose:
        print(f"    + Final: {len(circuit)} gates total")

    return circuit, len(circuit)


def algebraic_aes_v3(verbose=True):
    """
    v3: 更激进的子表达式共享。

    关键洞察: xtime(x) 只改变 5 个位中的 3 个需要 XOR b7。
    如果我们在 "最终 XOR" 阶段融合 xtime 和加法，可以省门。

    例如 s0[1] = xtime(p)[1] + bcd[1] = (p[0] XOR p[7]) + bcd[1]
    如果 p[0] + bcd[1] 或 p[7] + bcd[1] 已经存在...

    更系统的方法: 生成所有可能的分解变体，
    每个变体用不同的子表达式共享策略，
    然后对所有变体运行局部搜索，取最优。
    """
    results = []

    # v2 基础版
    c1, g1 = algebraic_aes_v2(verbose=False)
    results.append((c1, g1, 'v2_base'))

    # v1 基础版
    c2, g2 = algebraic_aes_mixcolumns(verbose=False)
    results.append((c2, g2, 'v1_base'))

    # 变体: 用不同顺序计算 pairwise sums
    for variant in range(4):
        c, g = _algebraic_variant(variant, verbose=False)
        if c is not None:
            results.append((c, g, f'variant_{variant}'))

    if verbose:
        for c, g, name in results:
            print(f"    {name}: {g} gates")

    results.sort(key=lambda x: x[1])
    return results


def _algebraic_variant(variant_id, verbose=False):
    """生成代数分解的变体"""
    n_inputs = 32
    basis = []
    for i in range(n_inputs):
        vec = np.zeros(n_inputs, dtype=np.int8)
        vec[i] = 1
        basis.append(vec)
    circuit = []

    def xor_gate(u, v):
        new_vec = (basis[u].astype(np.int16) + basis[v].astype(np.int16)) % 2
        new_vec = new_vec.astype(np.int8)
        for k in range(len(basis)):
            if np.array_equal(basis[k], new_vec):
                return k
        idx = len(basis)
        basis.append(new_vec)
        circuit.append((u, v))
        return idx

    def byte_xor_nodes(nodes_a, nodes_b):
        return [xor_gate(nodes_a[bit], nodes_b[bit]) for bit in range(8)]

    def xtime_nodes(byte_nodes):
        b = byte_nodes
        r = [0] * 8
        r[0] = b[7]
        r[1] = xor_gate(b[0], b[7])
        r[2] = b[1]
        r[3] = xor_gate(b[2], b[7])
        r[4] = xor_gate(b[3], b[7])
        r[5] = b[4]
        r[6] = b[5]
        r[7] = b[6]
        return r

    a = list(range(0, 8))
    b = list(range(8, 16))
    c = list(range(16, 24))
    d = list(range(24, 32))

    if variant_id == 0:
        # 变体 0: 先算 cross sums u=a+c, v=b+d
        u = byte_xor_nodes(a, c)  # a+c
        v = byte_xor_nodes(b, d)  # b+d
        t = byte_xor_nodes(u, v)  # a+b+c+d

        # xtime(a+b): 需要 a+b
        p = byte_xor_nodes(a, b)
        xt_p = xtime_nodes(p)

        # s0 = xt_p + t + a = xt_p + (b+c+d)
        # b+c+d = t + a
        bcd = byte_xor_nodes(t, a)
        s0 = byte_xor_nodes(xt_p, bcd)

        # xtime(b+c)
        q = byte_xor_nodes(b, c)
        xt_q = xtime_nodes(q)
        acd = byte_xor_nodes(t, b)
        s1 = byte_xor_nodes(xt_q, acd)

        # xtime(c+d)
        r = byte_xor_nodes(c, d)
        xt_r = xtime_nodes(r)
        abd = byte_xor_nodes(t, c)
        s2 = byte_xor_nodes(xt_r, abd)

        # xtime(d+a)
        s_da = byte_xor_nodes(d, a)
        xt_s = xtime_nodes(s_da)
        abc = byte_xor_nodes(t, d)
        s3 = byte_xor_nodes(xt_s, abc)

    elif variant_id == 1:
        # 变体 1: 先算 xtime 再合并
        xt_a = xtime_nodes(a)
        xt_b = xtime_nodes(b)
        xt_c = xtime_nodes(c)
        xt_d = xtime_nodes(d)

        # s0 = xt_a + xt_b + b + c + d = (xt_a + xt_b) + (b + c + d)
        xt_ab = byte_xor_nodes(xt_a, xt_b)
        xt_bc = byte_xor_nodes(xt_b, xt_c)
        xt_cd = byte_xor_nodes(xt_c, xt_d)
        xt_da = byte_xor_nodes(xt_d, xt_a)

        # b+c+d, a+c+d, etc. via chain
        bc = byte_xor_nodes(b, c)
        bcd = byte_xor_nodes(bc, d)
        cd = byte_xor_nodes(c, d)
        acd = byte_xor_nodes(a, cd)
        ab = byte_xor_nodes(a, b)
        abd = byte_xor_nodes(ab, d)
        abc = byte_xor_nodes(ab, c)

        s0 = byte_xor_nodes(xt_ab, bcd)
        s1 = byte_xor_nodes(xt_bc, acd)
        s2 = byte_xor_nodes(xt_cd, abd)
        s3 = byte_xor_nodes(xt_da, abc)

    elif variant_id == 2:
        # 变体 2: 融合 xtime 和加法
        # s0 = 2(a+b) + b + c + d
        # 逐位构造，尝试最大化复用

        p = byte_xor_nodes(a, b)
        q = byte_xor_nodes(b, c)
        r = byte_xor_nodes(c, d)
        s_da = byte_xor_nodes(d, a)

        # 3-byte sums using chain: b+c+d = b + c + d
        # 但 b+c = q 已有
        bcd = byte_xor_nodes(q, d)
        # a+c+d: a + r
        acd = byte_xor_nodes(a, r)
        # a+b+d: p + d
        abd = byte_xor_nodes(p, d)
        # a+b+c: p + c
        abc = byte_xor_nodes(p, c)

        # xtime 在最后
        xt_p = xtime_nodes(p)
        xt_q = xtime_nodes(q)
        xt_r = xtime_nodes(r)
        xt_s = xtime_nodes(s_da)

        s0 = byte_xor_nodes(xt_p, bcd)
        s1 = byte_xor_nodes(xt_q, acd)
        s2 = byte_xor_nodes(xt_r, abd)
        s3 = byte_xor_nodes(xt_s, abc)

    elif variant_id == 3:
        # 变体 3: Maximov 风格 — 用 "rotational trick"
        # 算 s0 的电路，然后通过输入重映射获得 s1, s2, s3
        # 这样共享 xtime 逻辑

        # s0 = 2(a+b) + b + c + d
        # Compute for s0 first
        p = byte_xor_nodes(a, b)
        xt_p = xtime_nodes(p)
        # b+c+d = (b+c) + d
        bc = byte_xor_nodes(b, c)
        bcd = byte_xor_nodes(bc, d)
        s0 = byte_xor_nodes(xt_p, bcd)

        # s1 = 2(b+c) + a + c + d, reuse bc
        xt_bc = xtime_nodes(bc)
        cd = byte_xor_nodes(c, d)
        acd = byte_xor_nodes(a, cd)
        s1 = byte_xor_nodes(xt_bc, acd)

        # s2 = 2(c+d) + a + b + d, reuse cd
        xt_cd = xtime_nodes(cd)
        abd = byte_xor_nodes(p, d)  # reuse p = a+b
        s2 = byte_xor_nodes(xt_cd, abd)

        # s3 = 2(d+a) + a + b + c
        da = byte_xor_nodes(d, a)
        xt_da = xtime_nodes(da)
        abc = byte_xor_nodes(p, c)  # reuse p = a+b
        s3 = byte_xor_nodes(xt_da, abc)

    else:
        return None, None

    return circuit, len(circuit)


def generate_algebraic_circuits(target_matrix, n_variants=10, verbose=True):
    """
    生成多个代数分解变体的电路，验证并返回有效的。
    """
    T = np.array(target_matrix, dtype=np.int8)
    n_inputs = T.shape[1]
    valid_circuits = []

    # 收集所有变体
    all_variants = algebraic_aes_v3(verbose=verbose)
    if all_variants is None:
        all_variants = []

    for circuit, n_gates, name in all_variants:
        if verify_circuit(circuit, T, n_inputs):
            valid_circuits.append((circuit, n_gates, name))
            if verbose:
                print(f"    {name}: {n_gates} gates (VALID)")
        else:
            if verbose:
                print(f"    {name}: {n_gates} gates (INVALID)")

    # 对每个有效电路运行局部搜索
    optimized = []
    for circuit, n_gates, name in valid_circuits:
        opt = full_local_search(circuit, T, n_inputs, time_limit=30)
        opt_gates = len(opt)
        if verify_circuit(opt, T, n_inputs):
            optimized.append((opt, opt_gates, f"{name}_optimized"))
            if verbose:
                print(f"    {name}: {n_gates} → {opt_gates} gates (after local search)")

    all_circuits = valid_circuits + optimized
    all_circuits.sort(key=lambda x: x[1])
    return all_circuits


if __name__ == "__main__":
    from benchmark_matrices import get_aes_mixcolumns_matrix

    T = np.array(get_aes_mixcolumns_matrix(), dtype=np.int8)
    print(f"AES MixColumns: {T.shape}")

    print("\n=== v1: Basic algebraic decomposition ===")
    c1, g1 = algebraic_aes_mixcolumns(verbose=True)
    valid1 = verify_circuit(c1, T, 32)
    print(f"  Gates: {g1}, Valid: {valid1}")

    print("\n=== v2: With dedup ===")
    c2, g2 = algebraic_aes_v2(verbose=True)
    valid2 = verify_circuit(c2, T, 32)
    print(f"  Gates: {g2}, Valid: {valid2}")

    print("\n=== All variants ===")
    all_circuits = generate_algebraic_circuits(T, verbose=True)
    if all_circuits:
        print(f"\n  Best: {all_circuits[0][1]} gates ({all_circuits[0][2]})")
