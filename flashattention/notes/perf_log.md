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
