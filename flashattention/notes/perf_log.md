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
