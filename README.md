# 基于自适应压缩感知与状态空间模型的超声图像增强方法

本项目实现从超声 RF 数据或 B 模式图像出发，提取 HEVC 分块结构先验，进行自适应 Block K-SVD 去噪，再通过 USR-Mamba 增强低线数图像并计算评价指标的完整流程。[最终论文](thesis.pdf)保留在根目录。

论文主路径为 **32 线图像 → HEVC 结构先验 → 三尺度 Block K-SVD 去噪 → USR-Mamba 推理 → 与原生 128 线参考图像比较**。网络训练使用预处理后的原始 32 线／128 线配对图像；前级去噪接入推理路径。32／128 表示波束线数，网络输入输出保持相同像素尺寸。

## 完整流程图

实线表示主要处理和数据依赖，虚线表示可选对照或辅助评价输入。图中的阶段名与 `run.py` 子命令对应；训练、推理和评价通过清单、权重及输出目录衔接，需要分别执行。阶段编号与下方分步说明一致，以边框颜色区分阶段，文字和底色跟随显示主题。

```mermaid
flowchart TD
    %% 阶段标题放在普通节点中，避免跨子图连线穿过分组标题与边界。
    RF["RF 通道数据<br/>与采集参数"] --> BF["<b>1. RF 成像（可选）</b><br/>beamform<br/>延迟求和、包络与扫描转换"]
    BF --> BM["32 线／128 线 B 模式图像"]
    Existing["已有 B 模式图像"] --> BM
    BM --> Prep["<b>1. 图像预处理</b><br/>prepare<br/>灰度校验、裁剪与缩放"]
    Prep --> Low["原始 32 线 TIFF"]
    Prep --> High["原始 128 线 TIFF"]
    Prep --> PGM["同像素 PGM"]

    PGM --> HEVC["<b>2. HEVC 结构先验</b><br/>prior<br/>对待处理低线数图像编码"]
    HEVC --> Stream["HEVC 码流"]
    HEVC --> Blocks["TU／PU 分块数组"]
    Blocks --> Maps["<b>3. 自适应尺度图</b><br/>TU 插值与 PU 灰度调节<br/>平滑与尺度选择"]
    Low --> Maps
    Maps --> KSVD["<b>3. Block K-SVD 去噪</b><br/>denoise<br/>字典学习、稀疏编码与加权重建"]
    Low --> KSVD
    KSVD --> Denoised["去噪 TIFF<br/>连续／离散尺度图与耗时"]

    Low -->|train / val| Pairs["pairs.csv<br/>按来源划分的原始配对图像"]
    High -->|train / val| Pairs
    Pairs --> Train["<b>4. USR-Mamba 训练</b><br/>train<br/>训练集增强、对抗训练与验证"]
    Train --> Weights["USR-Mamba 权重<br/>参数与损失记录"]
    Denoised -->|cascade 测试图像| Infer["<b>5. 低线数图像增强</b><br/>infer<br/>加载权重并增强 32 线图像"]
    Weights --> Infer
    Infer --> Enhanced["增强 TIFF<br/>保持输入尺寸与相对目录"]

    Low -.-> Fixed["<b>3. 固定块对照</b><br/>fixed<br/>各固定尺度独立重建"]
    Fixed -.-> Eval["<b>6. 定量评价</b><br/>evaluate<br/>依据 evaluation.csv 计算指标"]
    Low -.->|原始图像对照| Eval
    Denoised -.->|去噪对照／连续尺度图| Eval
    Blocks -.->|TU 数组| Eval
    Annotations["ROI 与二值掩码（可选）"] -.-> Eval
    High -->|测试集参考| Eval
    Enhanced --> Eval
    Eval --> Reports["metrics.csv：单图指标<br/>group_means.csv：分组均值"]

    classDef imaging stroke:#6688cc,stroke-width:2px;
    classDef structure stroke:#c49a3a,stroke-width:2px;
    classDef denoising stroke:#57a373,stroke-width:2px;
    classDef training stroke:#a079c2,stroke-width:2px;
    classDef inference stroke:#4c9eaa,stroke-width:2px;
    classDef metrics stroke:#cf7a70,stroke-width:2px;
    class RF,BF,BM,Existing,Prep,Low,High,PGM imaging;
    class HEVC,Stream,Blocks structure;
    class Maps,KSVD,Denoised,Fixed denoising;
    class Pairs,Train,Weights training;
    class Infer,Enhanced inference;
    class Annotations,Eval,Reports metrics;
```

Block K-SVD 提供两组配置：`cascade` 使用 4、8、12 三种块尺寸，作为 USR-Mamba 前级；`denoising` 使用 4、8、12、16、20、24 六种块尺寸，用于独立去噪实验。完整图中的 `denoise` 按所选配置分别运行，`fixed` 为各固定块尺寸生成独立对照结果。

## 阶段与代码入口

| 阶段 | 目录 | 输入与输出 |
| --- | --- | --- |
| 1. 成像与预处理 | `imaging/` | `beamform`（可选）、`prepare`：RF 或 B 模式图像 → 对齐的灰度 TIFF／PGM |
| 2. 结构先验 | `hevc_prior/` | `prior`：PGM → HEVC 码流、TU／PU 数值数组 |
| 3. 自适应稀疏去噪 | `block_ksvd/` | `denoise`／`fixed`：图像及先验 → 去噪图像、尺度图与耗时 |
| 4. 网络训练 | `usr_mamba/` | `train`：原始低／高线数配对图像及清单 → 权重与训练／验证记录 |
| 5. 状态空间增强 | `usr_mamba/` | `infer`：去噪低线数图像及权重 → 增强图像 |
| 6. 指标计算 | `evaluation/` | `evaluate`：配准图像、尺度图及标注 → 单图与分组指标 |

`run.py` 提供统一入口，`configs/paper.json` 集中定义实验参数；数据位置、输出位置、MATLAB、编码器和计算设备均由命令行传入。`common/` 只处理配置、输入约定及 MATLAB 调用。

## 分步处理流程

### 1. RF 成像与图像预处理

输入可为包含通道信号及物理采集参数的 MAT 文件，也可为已有的单帧 8 位灰度 TIFF。RF 输入先通过 `beamform` 生成 B 模式 TIFF，已有图像直接进入 `prepare`。低／高线数配对必须配准并采用相同预处理，训练、验证和测试来源由数据提供者事先划分。

```mermaid
flowchart TD
    RF["RF 通道数据、频率<br/>采样率与发射延迟"] --> Window["逐扫描线平移 Hanning 窗<br/>并加权通道"]
    Window --> DAS["按传播延迟取样<br/>并进行通道求和"]
    DAS --> Envelope["归一化与 Hilbert 包络检测"]
    Envelope --> Log["对数压缩<br/>双线性扫描转换"]
    Log --> TIFF["B 模式灰度 TIFF"]
    Existing["已有 B 模式 TIFF"] --> TIFF
    TIFF --> Validate["校验单帧、8 位灰度<br/>及裁剪范围"]
    Validate --> Crop["按 crop_xyxy 裁剪"]
    Crop --> Resize["等比例双线性缩放<br/>512 × 768（宽 × 高）"]
    Resize --> Images["images/：新建灰度 TIFF"]
    Resize --> PGM["pgm/：同像素 P5 PGM"]
```

默认裁剪窗口为 `(0,0,750,1125)`，右／下边界不包含在裁剪区域内。预处理输出保留中性文件名及相对目录，并重新写入像素，移除原 TIFF 元数据。HEVC 输入宽高须为 32 的倍数。

对应实现：[RF 成像](imaging/beamform_rf.m)、[预处理](imaging/preprocess.py)。

### 2. HEVC 结构先验提取

`prior` 对 PGM 进行帧内编码，利用图像结构决定预测单元（PU）和变换单元（TU）的分块。后续去噪读取的是分块数组，图像像素来自预处理 TIFF。

```mermaid
flowchart TD
    PGM["P5 灰度 PGM"] --> CTU["按 32 × 32 编码树单元处理"]
    CTU --> Candidates["递归候选分块<br/>PLANAR／DC 帧内预测"]
    Candidates --> Transform["预测残差变换与量化"]
    CTU --> Gradient["计算块梯度能量"]
    Gradient -.->|调整系数清零阈值| Transform
    Transform --> Cost["结合像素／结构失真与码率<br/>评估候选"]
    Gradient -.->|调节失真与码率权重| Cost
    Cost --> Select["选择分块及预测模式"]
    Select --> Bitstream["熵编码并输出 .hevc 码流"]
    Select --> Arrays["导出 _tu.csv 与 _pu.csv"]
```

质量索引 0–4 对应 QP 4、10、16、22、28，默认索引为 0。TU／PU CSV 均无表头，每行是 `[block_size,x,y]`，坐标以左上角为原点、从 0 开始；相对目录和样本名与输入一致。

对应实现：[先验提取入口](hevc_prior/runner.py)、[编码器](hevc_prior/codec.c)。

### 3. 自适应尺度图与 Block K-SVD 去噪

`denoise` 将图像、TU 和 PU 结合，先生成每个像素的块尺度，再按实际出现的尺度分别学习字典和重建。字典训练图像由当前输入叠加合成脉冲／高斯噪声得到；最终稀疏重建使用当前输入图像的块。

```mermaid
flowchart TD
    TU["TU：块尺寸与位置"] --> Centers["提取中心、去重<br/>并保留较小 TU 尺寸"]
    Centers --> Interpolation["Delaunay 线性插值<br/>凸包外最近邻补全"]
    PU["PU 分块"] --> Brightness["计算各 PU 平均灰度"]
    Interpolation --> Zoom["根据 PU 灰度放大暗区尺度"]
    Brightness --> Zoom
    Zoom --> Smooth["均值平滑并归一化<br/>生成连续尺度图"]
    Smooth --> Discrete["匹配所选 profile 中<br/>的最近块尺寸"]
    Discrete --> TrainingPatches["按尺度分组并提取训练块"]
    Image["输入灰度图像"] --> Noise["合成脉冲噪声、膨胀<br/>及高斯噪声"]
    Image --> Brightness
    Noise --> TrainingPatches
    TrainingPatches --> Dictionary["每种尺度初始化字典"]
    Dictionary --> Learn["交替执行相关性稀疏编码<br/>与逐原子 SVD 更新"]
    Learn --> Stop{"达到收敛门限或迭代上限？"}
    Stop -->|否| Learn
    Stop -->|是| Encode["用各尺度字典<br/>对输入图像块稀疏编码"]
    Image --> Encode
    Discrete --> Encode
    Encode --> Reconstruct["字典乘系数还原图像块"]
    Reconstruct --> Fusion["重叠块高斯加权累加<br/>并归一化"]
    Fusion --> Output["images/ 去噪图像与 timing.csv"]
    Smooth --> Continuous["scale_maps/ 连续尺度图"]
    Discrete --> DiscreteOutput["scale_maps/ 离散尺度图"]
```

中心不足或共线时，尺度插值回退到最近邻。默认字典为 512 个原子、稀疏率 0.1、最多迭代 10 轮。连续尺度图用于后续尺度分区评价，离散尺度图决定逐像素块尺寸。

| `--profile` | 块边长（像素） | 用途 |
| --- | --- | --- |
| `cascade` | 4、8、12 | USR-Mamba 推理前级 |
| `denoising` | 4、8、12、16、20、24 | 独立去噪与固定块对照 |

`fixed` 跳过 TU／PU 尺度估计，对配置中的每种固定块尺寸分别学习、重建，并使用等权重融合；同一图像的各固定尺度共享同一份合成训练噪声。结果写入 `images/fixed_04/` 等目录。

对应实现：[批量去噪](block_ksvd/batch_denoise.m)、[尺度图](block_ksvd/scale_map.m)、[字典学习](block_ksvd/dictionary_learning.m)、[稀疏编码](block_ksvd/sparse_encode.m)、[重建与融合](block_ksvd/reconstruct_image.m)。

### 4. USR-Mamba 配对训练与独立验证

`train` 读取 `pairs.csv`，以预处理后的原始 32 线图像为输入、原始 128 线图像为参考。清单包含 `source_id,split,input,reference`，`split` 为 `train`、`val` 或 `test`；训练入口只使用前两者，并核验来源及文件划分。

```mermaid
flowchart TD
    Manifest["pairs.csv<br/>原始低／高线数 TIFF"] --> Check["核验来源、路径、尺寸<br/>及集合划分"]
    Check --> Train["train：四次成对裁切<br/>Lanczos 恢复尺寸<br/>水平翻转"]
    Train --> Normalize["灰度映射到 -1 至 1"]
    Normalize --> Generator["USR-Mamba 生成增强图像"]
    Generator --> L1["与对应 128 线参考计算 L1 损失"]
    Normalize --> Discriminator["条件 Patch 判别器<br/>真实配对与生成配对<br/>生成配对回放池"]
    Generator --> Discriminator
    Discriminator --> Adversarial["最小二乘对抗损失"]
    L1 --> Update["加权损失与 Adam<br/>交替更新生成器、判别器"]
    Adversarial --> Update
    Check --> Validation["val：完整配对图像<br/>不裁切、不翻转，灰度归一化"]
    Update --> Validate["每轮切换到评价模式<br/>无梯度计算验证 L1"]
    Validation --> Validate
    Validate --> Logs["losses.csv<br/>训练 L1、验证 L1 与学习率"]
    Validate --> Save["按保存周期及最终轮<br/>写入 usr_mamba_NNN.pt"]
    Parameters["实际实验参数"] --> Config["parameters.json"]
    Parameters --> Update
```

默认批量为 1、学习率为 0.0002、Adam 参数为 `(0.5,0.999)`；先恒定学习率训练 30 轮，再线性衰减训练 30 轮。生成器损失为 `10 × L1 + 1 × 最小二乘对抗损失`。每个训练来源得到四组确定的成对裁切，只裁左右和底部 0–32 像素、顶部不变，再通过水平翻转形成八个样本；验证集保持独立。权重默认每 5 轮及最终轮保存，记录模型结构与变体。

`--variant` 支持完整模型 `full`，以及纯卷积瓶颈 `conv_bottleneck`、去傅里叶分支 `no_fourier`、去 SSM 后残差精炼 `no_refinement`、行优先扫描 `raster_scan` 四种结构消融。

对应实现：[配对数据与划分](usr_mamba/dataset.py)、[训练与验证](usr_mamba/training.py)。

### 5. USR-Mamba 网络与低线数推理

`infer` 加载指定权重及其中记录的模型结构，对 `cascade` 输出的去噪 32 线图像进行增强。以下是默认完整模型的数据流，`H × W` 表示输入的高和宽。

```mermaid
flowchart TD
    Input["去噪 32 线 TIFF 与指定权重"] --> Normalize["灰度映射到 -1 至 1"]
    Normalize --> E1["7 × 7 卷积<br/>64 通道，H × W"]
    E1 --> E2["步长 2 卷积<br/>128 通道，H/2 × W/2"]
    E2 --> E3["步长 2 卷积<br/>256 通道，H/4 × W/4"]
    E3 --> Bottleneck["9 个瓶颈块<br/>第 1、5、9 块为双域 SSM<br/>其余为残差卷积"]
    Bottleneck --> D1["拼接对应编码特征并转置卷积<br/>128 通道，H/2 × W/2"]
    E3 -->|跳跃连接| D1
    D1 --> D2["拼接对应编码特征并转置卷积<br/>64 通道，H × W"]
    E2 -->|跳跃连接| D2
    D2 --> Head["拼接浅层特征<br/>7 × 7 卷积与 Tanh"]
    E1 -->|跳跃连接| Head
    Head --> Save["映射回 0 至 255 并裁限<br/>保存 8 位灰度 TIFF"]
```

双域 SSM 包含两条分支：空间特征经二维扫描、LayerNorm 和 Mamba 后还原；频域特征经二维 FFT 与中心化，将实部／虚部拼接后扫描建模，再逆变换回空间域。两分支融合后，与输入特征拼接融合，并进行残差卷积精炼。默认扫描顺序为从边界向中心逐层绕行的 `outside_in`。

推理采用评价模式并关闭梯度，输入宽高须能被默认下采样因子 4 整除。输出保留尺寸、样本名和相对目录。原始 32 线输入对照可使用相同权重另行执行 `infer`。

对应实现：[网络结构](usr_mamba/network.py)、[推理与图像保存](usr_mamba/inference.py)。

### 6. 定量评价与分组统计

`evaluate` 读取 `evaluation.csv`，每行指定参考图像、方法结果及可选的连续尺度图、ROI、TU 和掩码。结果与参考必须处于相同像素坐标系；原始输入、固定块去噪、自适应去噪和网络增强结果可分别建立清单进行比较。

```mermaid
flowchart TD
    Manifest["evaluation.csv<br/>样本、分组、结果、参考<br/>与可选标注"] --> Check["核验路径并读取对齐图像"]
    Check --> Whole["全图 CNR、ENL<br/>相对参考的 EPI 与 FSIM"]
    Check --> Region["结果图像与连续尺度图<br/>分区 CNR"]
    Check --> Width["结果图像与 ROI<br/>重心校正、区间扩张<br/>半高交点插值"]
    Width --> FWHM["横向／纵向 FWHM，单位为像素"]
    Check --> Partition["TU 与二值掩码<br/>按 TU 中心归类并统计尺寸"]
    Check --> LPIPS["可选 --lpips<br/>结果／参考灰度复制为三通道<br/>计算感知距离"]
    Whole --> Metrics["metrics.csv：逐样本指标"]
    Region --> Metrics
    FWHM --> Metrics
    Partition --> Metrics
    LPIPS --> Metrics
    Metrics --> Groups["按 group 对有效值取均值<br/>写入 group_means.csv"]
```

全图 `global_cnr` 使用均值／样本标准差，ENL 使用均值平方／样本方差；EPI 使用 Sobel 梯度相关。FSIM 使用多尺度 LoG 相位响应近似，具体口径见[复现指南](docs/reproduction.md)。尺度分区 CNR 默认将连续尺度图的 `[0,4)` 作为目标、`(8,12]` 作为背景、`[0,12]` 作为噪声统计范围。

缺少对应尺度图、ROI 或 TU／掩码时，相关指标保留为缺失值（NaN）；LPIPS 通过 `--lpips` 显式启用，需要额外特征评价依赖和公开特征权重。

对应实现：[评价入口](evaluation/runner.py)、[MATLAB 批量指标](evaluation/evaluate_batch.m)。

## 运行与复现

依次使用 `beamform`（可选）、`prepare`、`prior`、`denoise` 准备图像；通过 `train` 得到权重后执行 `infer`，最后通过 `evaluate` 汇总结果。已有兼容权重时可直接进入推理；`fixed` 及各网络变体用于独立对照。输入清单、标注和数据划分需事先准备。

环境安装、编码器构建和各阶段命令见[复现指南](docs/reproduction.md)，输入与清单格式见[数据契约](docs/data_contract.md)，已完成的合成验证及实测范围见[验证记录](docs/validation.md)。

归档包含流程源码、参数、必要许可证与论文，不包含采集数据、样本清单、ROI 实测标注、权重、实验结果、安装环境、编译产物或开发备份。运行目录默认排除在 Git 之外。

```sh
python -B run.py --help
python -B tools/verify_archive.py
```
