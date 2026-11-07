# multi_tensor_adam 算子开发文档

## 一、接口介绍

### 算子名字

本次开发算子为 multi_tensor_adam，公共接口为 flag_train.deepspeed.multi_tensor_adam，测试算子 ID 为 multi_tensor_adam。在 FlagTrain 中提供通用 Triton 实现，并通过运行时分发选择不同平台的专用实现。

当前多平台验收范围为 FP32、FP16、BF16 的前向优化器更新计算。涉及 Adam（L2 正则化）与 AdamW（解耦权重衰减）两种模式。反向传播（由于优化器本身不属于前向计算图）不在本次验证范围内。

### 输入输出参数

from flag_train.deepspeed import multi_tensor_adam

multi_tensor_adam(
    chunk_size, noop_flag, tensor_lists, 
    lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay
)

输入 tensor_lists 包含 4 个列表：[g_list, p_list, m_list, v_list]，分别对应梯度、参数、一阶矩、二阶矩。所有 Tensor 必须位于同一加速设备且为连续内存。入口检查维度、类型、形状及参数合法性，边界通过掩码（Mask）处理。

### 计算语义

原版 CUDA 计算逻辑被完整移植：
若 mode == 0（L2 正则化 Adam）：
  g = g + weight_decay * p
若 mode == 1（AdamW 解耦权重衰减）：
  （不影响 g，在更新时直接作用于 p）

更新公式（以 FP32 精度计算后 Cast 回原类型）：
  m = beta1 * m + (1 - beta1) * g
  v = beta2 * v + (1 - beta2) * g * g
  m_hat = m / (1 - beta1^step)
  v_hat = v / (1 - beta2^step)
  update = m_hat / (sqrt(v_hat) + epsilon) + (weight_decay * p if mode == 1 else 0)
  p = p - lr * update

## 二、代码结构

项目：[FlagTrain](https://github.com/flagos-ai/FlagTrain)；提交记录：[PR #29](https://github.com/flagos-ai/FlagTrain/pull/29)

## 三、实现方案与优化方法

### 突破原版 48 Tensor 的硬限制
原版 DeepSpeed CUDA 实现将每个 Tensor 的指针和大小封装在 Kernel 启动参数（TensorListMetadata）中，受限于 Kernel 参数 4KB 上限，单次 Launch 最多只能处理 48 个 Tensor，超出则需多次 Launch。
Triton 版将元数据打包成 Device Tensors 传入显存，Kernel 内部通过 tl.load 查表进行间接寻址（block_to_tensor 和 block_to_offset）。彻底解除了单次 Launch 的并发上限，实现任意数量的 Tensor 融合。

### Host 侧缓存：消除 Python 调度开销
若每次 step() 都去遍历 Python 列表提取 data_ptr 并构建映射表，Python 的 Host 开销将远超 Kernel 执行时间。在 Host 侧基于 (张量地址, 大小) 建立缓存，由于训练时传入的 Tensor 对象地址固定，后续 step 直接命中缓存，将 Host 调度开销降至几乎为零。

### 编译期特化：消除运行时分支
利用 Triton 的 tl.constexpr，将 MODE（0 或 1）和 DTYPE（fp32/fp16/bf16）作为编译期常量传入。编译器会为不同模式和数据类型生成无分支的优化代码。原版 CUDA 不支持 bf16，Triton 版补齐了这一支持。

### 异构硬件分块适配
在不同硬件上测试了 BLOCK_SIZE 对性能的影响。例如在昇腾 NPU 上，由于 Unified Buffer 限制，BLOCK_SIZE 过大会导致编译期报显存溢出；而在部分 GPU 上，增大 BLOCK_SIZE 能显著提升带宽利用率。因此针对不同平台适配了最佳的 Block 配置。

## 四、测试方案

### 测试 shape
核心测试 tensor 数量为 16 个（模拟真实优化器参数组），每个 Tensor 的 numel 维度涵盖：1024, 4096, 16384, 65536, 262144, 1048576。
除常规尺寸外，还测试了非对齐边界尺寸（如 1, 17, 1023, 1025）以保证边界掩码的安全。

### Baseline 口径
- **NVIDIA / Hygon 平台**：Baseline 为 DeepSpeed 原版 CUDA 融合算子（FusedAdam）。这是最公平的对比，反映内核质量。
- **其他平台**（昇腾、沐曦、天数、PPU）：因 DeepSpeed CUDA 不可用，Baseline 回退为纯 Torch 组合的循环实现（逐 Tensor 进行 Adam 更新）。加速比反映融合收益。

### 功能验收
使用仓库统一的 train_assert_close 路径比较结果。FP32 容差 atol=1e-4，FP16/BF16 容差按 8 * eps * max(abs) 计算。

### 性能验收
speedup = baseline_latency / flagtrain_latency
验收要求：各平台纳入验收的有效样例平均 speedup 表现良好，且核心场景不低于 0.8x。

## 五、各平台测试结果

- **NVIDIA (H20)**：DeepSpeed CUDA 作为 baseline，平均加速比 1.39x，最高 2.89x。
- **Hygon (海光)**：DeepSpeed CUDA 作为 baseline，平均加速比 2.24x，最高 3.35x。
- **Metax (沐曦)**：torch reference 作为 baseline，平均加速比 56.5x。
- **PPU (平头哥)**：torch reference 作为 baseline，平均加速比 210.5x。
- **Iluvatar (天数)**：torch reference 作为 baseline，平均加速比 129.4x。
- **Ascend (昇腾)**：正确性测试通过。受官方 do_bench_npu 计时器偏差影响（Profiler 实测 440us，接口报 1166ms），性能数据不计入结论。

## 六、开发中遇到的问题与处理

### 昇腾平台的 UB 内存限制
昇腾 NPU 的 Unified Buffer 较小，在设置 BLOCK_SIZE=16384 时触发 UB overflow 编译错误。通过将该平台的 BLOCK_SIZE 下调至 4096，解决了该问题并取得了最佳性能。

### 昇腾平台的计时器偏差
昇腾官方提供的 do_bench_npu 接口底层基于 Profiler，对微秒级 Kernel 的测量误差极大。通过编写自定义的 perf_counter + npu.synchronize 计时逻辑，还原了真实的设备侧性能。

## 七、测试命令与复现

先激活自己的虚拟环境并安装对应厂商兼容的 Torch／FlagTree。

### 功能测试
python3 -m pytest tests/deepspeed/test_multi_tensor_adam.py --ref cpu -q -ra

### 性能测试
python3 -m pytest benchmark/deepspeed/test_multi_tensor_adam.py -q -s --mode=kernel

## 八、总结与后续工作
本次开发在 NVIDIA、Hygon、Ascend、Metax、PPU、Iluvatar 上完成了对应环境的验证。
主要优化是突破单次 Launch 数量限制、Host 侧缓存消除开销、以及编译期特化。
所有已测平台均通过了完整的功能正确性校验。
