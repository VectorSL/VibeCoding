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
