# 基于自适应压缩感知与状态空间模型的超声图像增强方法

超声图像增强流程：**RF/B 模式成像 → HEVC 结构先验 → 自适应 Block K-SVD → USR-Mamba → 定量评价**。[最终论文](thesis.pdf)保留在根目录。

| 阶段 | 目录 | 输入与输出 |
| --- | --- | --- |
| 成像与预处理 | `imaging/` | 射频通道或 B 模式图像 → 统一灰度 TIFF/PGM |
| 结构先验 | `hevc_prior/` | PGM → HEVC 码流、TU/PU 数值数组 |
| 自适应稀疏去噪 | `block_ksvd/` | 图像与结构先验 → 去噪图像、连续与离散尺度图 |
| 状态空间增强 | `usr_mamba/` | 成对图像训练；低线数图像推理 → 增强图像 |
| 指标计算 | `evaluation/` | 配准图像、尺度图及标注 → 单图与分组指标 |

`run.py` 提供统一入口，`configs/paper.json` 集中定义实验参数；数据位置、输出位置、MATLAB、编码器和计算设备均由命令行传入。`common/` 只处理配置、输入约定及 MATLAB 调用。

完整步骤见[复现指南](docs/reproduction.md)，输入格式见[数据契约](docs/data_contract.md)，实测范围见[验证记录](docs/validation.md)。

归档包含流程源码、参数、必要许可证与论文，不包含采集数据、样本清单、ROI 实测标注、权重、实验结果、安装环境、编译产物或开发备份。运行目录默认排除在 Git 之外。

```sh
python -B run.py --help
python -B tools/verify_archive.py
```
