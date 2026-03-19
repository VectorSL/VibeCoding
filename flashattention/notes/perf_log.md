# FlashAttention 性能优化记录

## 格式
每次优化记录：
- **日期**: YYYY-MM-DD
- **优化项**: 简述
- **改动内容**: 具体改了什么
- **性能变化**: 耗时对比
- **结论**: 是否有效

---

## Round 0: 初始实现
- **日期**: 2026-03-18
- **优化项**: 基础框架搭建
- **改动内容**:
  - 创建 ccsrc/flash_attention_fwd.cu (forward kernel)
  - 创建 ccsrc/flash_attention_bwd.cu (backward kernel)
  - 创建 python/attention.py (Python绑定)
  - 创建 setup.py (编译配置)
  - 创建 run.py (运行入口)
- **性能变化**: 待测试
- **结论**: 初始实现完成

---

## Round 1: 优化尝试 (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **PyTorch参考**: 0.05ms

| 优化项 | 改动 | 性能 | 结论 |
|--------|------|------|------|
| Baseline | 简单kernel | 87-89ms | 基准 |
| THREADS=256 | 增加threads | 165ms | ❌ 更慢 |
| THREADS=128 + constant memory | 使用常量内存 | 166ms | ❌ 更慢 |
| BLOCK_N=96 | 增大block | FAILED | ❌ shared memory过大 |
| BLOCK_N=128 | 增大block | FAILED | ❌ shared memory过大 |
| #pragma unroll 8 | 循环展开 | 164ms | ❌ 更慢 |
| Double buffering | 预取优化 | FAILED | ❌ shared memory过大 |
| Warp级并行 | 多Q行处理 | 119ms | ❌ 更慢 |
| Vectorized loads | float2向量化 | 88ms | ➖ 差不多 |

**最终结论**: 最佳性能 ~87ms，PyTorch ~0.05ms (差距1700x)

**原因分析**:
- PyTorch使用CUDA Tensor Cores进行矩阵乘法加速
- PyTorch使用深度内核融合减少内存带宽
- 手写CUDA难以达到同样优化级别

---

## Round 2: 线程并行 dot product (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: 让所有128线程并行参与dot product计算，用block reduction汇总
- **改动内容**: 每个线程负责部分D维度的乘加，通过warp+shared memory reduction得到完整dot product
- **性能变化**: 87ms → 15.5ms
- **结论**: ✅ 有效，提升5.6x

---

## Round 3: 多Q行tiling + thread-per-D (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: THREADS=64 (一个线程对应一个D维度)，BLOCK_Q=4 (每block处理4个Q行)
- **改动内容**:
  - 将THREADS从128改为64，匹配D=64，每线程负责一个D维度
  - 每个block处理BLOCK_Q=4个Q行，提高block利用率
  - 减少grid大小(N/4)，增加每block工作量
- **性能变化**: 15.5ms → 5.3ms
- **结论**: ✅ 有效，提升2.9x (累计从87ms→5.3ms，16.4x)

---

## Round 4: 单warp kernel (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: THREADS=32 (单warp)，消除__syncthreads
- **改动内容**: 每线程处理2个D维度，warp shuffle替代block reduction
- **性能变化**: 5.3ms → 7.3ms
- **结论**: ❌ 更慢，32线程加载shared memory太慢，occupancy低

---

## Round 5: BLOCK_Q=8 + __expf (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: 基于Round 3增大BLOCK_Q到8，使用__expf快速数学
- **改动内容**: BLOCK_Q从4增到8，expf改为__expf
- **性能变化**: 5.3ms → 4.9ms
- **结论**: ✅ 小幅提升 (累计 87ms → 4.9ms, 17.8x)

---

## Round 6: Warp-per-Q-row 架构 (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: 每个warp独立处理一个Q行，block内4个warp并行处理4个Q行
- **改动内容**:
  - 4 warps (128 threads)，每warp处理一个Q行
  - dot product只需warp shuffle，热循环中无__syncthreads
  - 只在KV tile加载时sync一次
  - 每lane处理2个D维度 (D=64, 32 lanes)
- **性能变化**: 4.9ms → 2.66ms
- **结论**: ✅ 有效，提升1.84x (累计 87ms → 2.66ms, 32.7x)

---

## Round 7: 8 warps per block (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: NUM_WARPS从4增到8，256线程，提高occupancy
- **改动内容**: NUM_WARPS=8, THREADS=256
- **性能变化**: 2.66ms → 2.20ms
- **结论**: ✅ 有效，提升21% (累计 87ms → 2.20ms, 39.5x)

---

## Round 8-11: 无效尝试 (2026-03-19)

| Round | 优化项 | 性能 | 结论 |
|-------|--------|------|------|
| 8 | half shared memory + BLOCK_N=128 | 2.23ms | ➖ 持平 |
| 9 | shared memory bank conflict padding | 2.29ms | ❌ 无效 |
| 10 | Lane-per-KV (每lane算完整dot product) | 5.85ms | ❌ 更慢 |
| 11 | 寄存器缓存4个KV positions | 3.40ms | ❌ 更慢 |

---

## Round 12: 修复编译架构 + 去掉寄存器限制 (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: 修复setup.py架构检测bug (sm_80→sm_89)，去掉-maxrregcount=128
- **改动内容**:
  - 修复get_device_capability只取major的bug，现在正确生成sm_89
  - 去掉-maxrregcount=128，让编译器自由分配寄存器
- **性能变化**: 2.20ms → 2.14ms
- **结论**: ✅ 小幅提升 (累计 87ms → 2.14ms, 40.7x)

---

## Round 13: WMMA Tensor Core 加速 QK^T (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: 使用 CUDA WMMA (16x16x16) 计算 QK^T 矩阵乘法
- **改动内容**:
  - 引入 mma.h，使用 wmma::mma_sync 做 half 精度矩阵乘
  - 16 Q rows × 16 KV positions 一次 WMMA 计算
  - D=64 拆成 4 个 WMMA_K=16 的 tile
  - 4 warps: warp 0 做 WMMA，所有 warps 并行做 softmax + V 累加
- **性能变化**: 2.14ms → 1.32ms
- **结论**: ✅ 有效，提升38% (累计 87ms → 1.32ms, 65.9x)

---

## Round 14-15: WMMA 扩展尝试 (2026-03-19)

| Round | 优化项 | 性能 | 结论 |
|-------|--------|------|------|
| 14 | 4 warps全做WMMA + WMMA P*V | FAIL/1.80ms | ❌ 正确性失败/更慢 |
| 15 | 2 warps WMMA + BLOCK_KV=32 | 1.50ms | ❌ 比Round13慢 |

**分析**: Round 13 (1 warp WMMA, BLOCK_KV=16) 是最优配置。增大BLOCK_KV会增加shared memory和scores矩阵大小，V累加遍历更多KV positions，反而更慢。

**当前最佳**: Round 13, 1.32ms (87ms → 1.32ms, 65.9x)

---

## Round 16: WMMA for both QK^T and PV (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: P×V 矩阵乘法也用 WMMA Tensor Core
- **改动内容**:
  - Step 1: WMMA 计算 S = Q @ K^T (16x16)
  - Step 2: 标量 online softmax，生成 P_half (16x16)
  - Step 3: WMMA 计算 O_tile = P @ V (16xD，D 拆成 16-wide tiles，4 warps 并行)
  - Step 4: 标量 rescale running O accumulator
  - O_acc 和 softmax 状态存在 shared memory
- **性能变化**: 1.32ms → 0.76ms
- **结论**: ✅ 大幅提升42% (累计 87ms → 0.76ms, 114.5x)

---

## Round 17: BLOCK_KV=32 双 WMMA tile (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: BLOCK_KV 从 16 增到 32，减少 tile 迭代次数
- **改动内容**:
  - QK^T: 2 个 WMMA (warp 0 和 warp 1 各算 16 个 KV)
  - PV: 每个 D-tile 做 2 次 WMMA (K 维度拆成 2 个 16)
  - tile 迭代从 32 次减到 16 次
- **性能变化**: 0.76ms → 0.65ms
- **结论**: ✅ 有效，提升14% (累计 87ms → 0.65ms, 133.8x)

---

## Round 18: BLOCK_KV=64, 4 WMMA tiles (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: BLOCK_KV 从 32 增到 64，tile 迭代从 16 减到 8
- **改动内容**:
  - QK^T: 4 warps 各算 16 KV positions (全部参与)
  - PV: 每个 D-tile 累加 4 个 K-tile (unrolled)
- **性能变化**: 0.65ms → 0.60ms
- **结论**: ✅ 有效 (累计 87ms → 0.60ms, 145x)

---

## Round 19: 并行化 softmax + O_acc rescaling (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: softmax 从单线程串行改为全线程并行
- **改动内容**:
  - scale 应用: 128 线程并行处理 16×64 scores
  - row max: 4 线程/行并行 reduce
  - exp + P_half: 128 线程并行
  - row sum: 4 线程/行并行 reduce
  - O_acc rescale: 128 线程并行
- **性能变化**: 0.60ms → 0.41ms
- **结论**: ✅ 大幅提升32% (累计 87ms → 0.41ms, 212x)

---

## Round 20: 消除 O_tile，WMMA 直接累加到 O_acc (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: PV WMMA 直接累加到 O_acc，省掉 O_tile 和 add 步骤
- **改动内容**:
  - WMMA accumulator 用 load_matrix_sync 从 O_acc 初始化
  - WMMA mma_sync 直接累加 PV 结果
  - store_matrix_sync 写回 O_acc
  - 省掉 O_tile shared memory (4KB) 和一次 syncthreads + add
- **性能变化**: 0.41ms → 0.31ms
- **结论**: ✅ 大幅提升25% (累计 87ms → 0.31ms, 280.6x)

---

## Round 21-22: 无效尝试 (2026-03-19)

| Round | 优化项 | 性能 | 结论 |
|-------|--------|------|------|
| 21 | 8 warps + BLOCK_KV=128 | 0.39ms | ❌ shared memory 超限，occupancy 下降 |
| 22a | S_float/P_half 共享内存 | FAIL | ❌ 数据损坏，正确性失败 |
| 22b | 合并 scale+max, sum+rescale syncs | 0.33ms | ➖ 持平 |

**当前最佳**: Round 20, 0.31ms (87ms → 0.31ms, 280.6x)

---

## Round 23-24: sync 合并 + vectorized load (2026-03-19)

| Round | 优化项 | 性能 | 结论 |
|-------|--------|------|------|
| 23 | K/V 共用 shared memory | 0.345ms | ❌ 多一次 V 加载 sync 抵消 |
| 24a | 合并 exp+sum 步骤 | 0.342ms | ➖ 持平 |
| 24b | float4 vectorized K/V load | 0.259ms | ✅ 提升16% |

**Round 24b 详情**:
- 全 tile 时用 float4 (128-bit) 向量化加载 K 和 V
- 每次 float4 加载 8 个 half 值
- 减少 global memory 事务数量
- **性能变化**: 0.31ms → 0.26ms (累计 87ms → 0.26ms, 334.6x)

---

## Round 25: 8 threads/row softmax (2026-03-19)
- **测试配置**: B=1, H=8, N=512, M=512, D=64
- **优化项**: softmax reduce 从 4 threads/row 改为 8 threads/row
- **改动内容**:
  - scale+max: 8 threads/row × 16 rows = 128 threads 全部参与
  - exp+sum: 同样 8 threads/row，每线程处理 8 个 KV
  - 消除了之前 64 个线程空闲的问题
- **性能变化**: 0.26ms → 0.22ms
- **结论**: ✅ 提升16% (累计 87ms → 0.22ms, 395.5x)

---

## Round 26-28: 三大优化方向尝试 (2026-03-19)

| Round | 优化项 | 性能 | 结论 |
|-------|--------|------|------|
| 26 | Warp shuffle softmax reduce | 0.219ms | ✅ 省1KB smem，性能持平 |
| 27 | Double buffer K (cp.async prefetch) | 0.306ms | ❌ smem增加降低occupancy |
| 28 | K/V共用smem (20KB) | 0.232ms | ❌ 多一次V加载+sync抵消occupancy提升 |

**当前最佳**: Round 26, 0.219ms (87ms → 0.22ms, 395.5x)
**vs PyTorch**: 0.06ms, 差距 ~3.7x

---

## Round 29: BLOCK_Q=32 尝试 (2026-03-19)

| Config | BLOCK_Q=16 | BLOCK_Q=32 | PyTorch |
|--------|-----------|-----------|---------|
| 1,8,512,512,64 | 0.219ms | 0.228ms | 0.058ms |
| 1,8,1024,1024,64 | 0.830ms | 0.790ms | 0.164ms |
| 2,16,1024,1024,64 | 2.309ms | 2.734ms | 0.383ms |

- BLOCK_Q=32 需要 8 warps (256 threads), ~40KB shared memory
- 在小配置上稍慢，大配置上持平或更慢
- **结论**: ❌ BLOCK_Q=16 仍然最优

**当前最佳**: Round 26, BLOCK_Q=16, 0.219ms (87ms → 0.22ms, 395.5x)

---

## 使用方法

1. **编译**:
   ```bash
   cd flashattention
   python setup.py build_ext --inplace
   ```

2. **运行测试**:
   ```bash
   python run.py
   ```

3. **单独运行**:
   ```bash
   python run.py --correctness  # 正确性测试
   python run.py --performance  # 性能测试
   ```
