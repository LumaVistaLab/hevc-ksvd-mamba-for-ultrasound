# 复现指南

## 环境

需要 Python 3.10+、C99 编译器、MATLAB，以及适配 CUDA 的 PyTorch、Mamba 扩展。MATLAB 使用 Image Processing 和 Signal Processing 工具箱。成像、尺度图、字典学习、重建和传统指标在 MATLAB 中运行；预处理、模型与训练在 Python 中运行。

```powershell
python -m pip install -r requirements.txt
cmake -S . -B work/build
cmake --build work/build --config Release
```

CMake 需要 3.16+。支持 MSVC、GCC 或 Clang；多配置构建的编码器通常位于 `work/build/Release/hevc_prior.exe`，单配置构建通常位于 `work/build/hevc_prior`。MSVC 开发者终端也可直接编译：

```powershell
New-Item -ItemType Directory -Force work/build | Out-Null
cl /nologo /std:c11 /D_CRT_SECURE_NO_WARNINGS /TC hevc_prior/encode.c hevc_prior/codec.c /Fowork/build/ /Fework/build/hevc_prior.exe
```

以下 PowerShell 示例从项目根目录运行。把 `$Matlab`、`$Encoder` 与 `$Device` 设置为本机可用值；它们均是调用参数，不写入源码。`work/` 仅是示例运行位置，也可以替换为项目外目录。已存在的结果目录不会被覆盖。

```powershell
$Config = 'configs/paper.json'
$Matlab = 'matlab'
$Encoder = 'work/build/Release/hevc_prior.exe'
$Device = 'cuda:0'
```

MATLAB 不在 PATH 时，`$Matlab` 使用其可执行文件完整路径。网络使用真实的 Mamba 算子；不具备兼容扩展时应先完成环境配置。`conv_bottleneck` 变体不使用状态空间算子，可用于卷积结构核验。

## 1. 准备输入与划分

数据提供者先将图像及标注改为中性编号，身份关联表保存在归档之外。输入契约详见 [数据格式](data_contract.md)。不从文件名推断患者关系，不自动生成论文的数据划分。

```text
work/
  raw_images/
    low/sample_000001.tif
    high/sample_000001.tif
  pairs.csv
```

`pairs.csv` 表头为 `source_id,split,input,reference`。路径相对于训练时的 `--data-root`，例如预处理后使用 `images/low/sample_000001.tif`。每个来源只能出现一次，分为 `train`、`val`、`test`；训练与验证来源、文件均不重叠。来源是否相同应由数据提供者确认。

论文记录 400 对候选、排除 20 对测试同源图像后保留 380 对训练来源、40 对验证图像和 20 张测试图像。严格复现须取得与这些数量对应的权威来源划分；本程序不为凑齐数量而将验证来源并入训练。

## 2. 可选 RF 成像

已有 B 模式图像时跳过。MAT 文件需按数据契约提供中性的 `acquisition` 结构，包含实际频率、采样率、延迟和通道数组。

```powershell
python -B run.py --config $Config beamform --input work/rf/low --output work/raw_images/low --beams 32 --matlab $Matlab
python -B run.py --config $Config beamform --input work/rf/high --output work/raw_images/high --beams 128 --matlab $Matlab
```

流程为扫描线平移窗加权、传播延迟取样、通道求和、Hilbert 包络、对数压缩和双线性扫描转换。声速、阵元数、孔径系数、延迟偏移、显示高度与动态范围在 `imaging.rf` 中配置。输入的物理量均使用 Hz、米、秒对应的 SI 单位；`transmit_delay_cycles` 为发射周期数。

## 3. 图像预处理

```powershell
python -B run.py --config $Config prepare --input work/raw_images --output work/prepared
```

配置中的 `crop_xyxy` 是左、上、右、下边界，右/下边界不含在裁剪区域内。默认裁剪窗口为 `(0,0,750,1125)`，等比例双线性缩放为 512×768。输出 `images/` 与 `pgm/` 具有相同的中性相对目录和像素；TIFF 作者、描述等元数据不会传递到新图像。

预处理只接受单帧 8 位灰度 TIFF。输入尺寸和裁剪范围须匹配配置；程序不会猜测 16 位或浮点图像的强度映射。HEVC 输入宽高为 32 的倍数。

## 4. HEVC 结构先验

```powershell
python -B run.py --config $Config prior --input work/prepared/pgm --output work/priors --encoder $Encoder
```

使用 PLANAR/DC 帧内预测、梯度敏感率失真代价、结构失真和梯度相关清零阈值。`hevc` 配置包含质量索引、失真/码率权重、梯度门限、权重增益及量化门限。质量索引 0–4 对应 QP 4、10、16、22、28，默认索引为 0。

每个样本输出 `sample_000001.hevc`、`sample_000001_tu.csv`、`sample_000001_pu.csv`，相对目录与图像一致。CSV 为 `[block_size,x,y]`，坐标为左上角起算的 0-based 像素。数值数组不包含可执行代码。位流语法、变换和熵编码表属于编码器常量；研究用搜索权重通过配置传入。

## 5. Block K-SVD

独立去噪采用六种尺寸，网络前级采用三种尺寸：

```powershell
python -B run.py --config $Config denoise --images work/prepared/images --priors work/priors --output work/denoising --profile denoising --matlab $Matlab
python -B run.py --config $Config fixed --images work/prepared/images --output work/fixed --profile denoising --matlab $Matlab
python -B run.py --config $Config denoise --images work/prepared/images --priors work/priors --output work/cascade --profile cascade --matlab $Matlab
```

| 配置 | 尺寸 | 用途 |
| --- | --- | --- |
| `denoising` | 4、8、12、16、20、24 | 论文第 3 章独立去噪与固定块对照 |
| `cascade` | 4、8、12 | 论文第 4 章 USR-Mamba 前级 |

尺度图经过 TU 中心去重、Delaunay 插值、凸包外最近邻补全、PU 灰度放大、平滑和最近尺度选择。不同尺度分别学习字典、进行相关选原子和残差更新，再融合重建。自适应分块使用高斯权重，固定分块使用等权重；固定尺度共享同一张图像的训练噪声。

参数均位于 `block_ksvd`：512 字典原子、稀疏率 0.1、最多 10 轮、残差门限 46225、收敛门限 0.0009；脉冲噪声密度 0.003、膨胀核 3、高斯标准差 0.001；尺度放大 0.5、指数 6、平滑半径 12、高斯融合比例 0.5。保留逐次相关选择的稀疏编码规则，不额外加入联合最小二乘重估。

输出图像位于 `images/`，连续和离散尺度图位于 `scale_maps/`，每幅图耗时写入 `timing.csv`。固定块对照位于 `images/fixed_04/` 等目录。输入文件树可只包含待测试的低线数图像，无需对未使用图像进行处理。

## 6. USR-Mamba 训练与结构消融

```powershell
python -B run.py --config $Config train --manifest work/pairs.csv --data-root work/prepared --output work/models/full --variant full --device $Device
```

训练使用原始 32 线输入与对应原始 128 线参考。四次成对裁切只作用于左右和底部 0–32 像素，顶部保持不变，Lanczos 恢复原尺寸，再通过确定性水平翻转加倍。380 对有效来源对应每轮 3040 个样本。验证不裁切、不翻转、不参与梯度，并使用评价模式。

`usr_mamba` 定义通道、卷积核、瓶颈深度、SSM 位置与状态参数。默认编码通道 64/128/256，九个瓶颈块中的第 1、5、9 个为双域 SSM，状态维数 16、局部卷积 4、扩展倍数 2。空间分支和中心化傅里叶实/虚分支分别扫描、建模、还原，再融合并做残差卷积精炼。

`scan_order` 明确指定二维序列排列：`outside_in` 为从边界向中心逐层绕行，`inside_out` 为其逆序，`raster` 为行优先。参数预设使用 `outside_in`；不同排列的权重不能视作等价。网络在相同像素网格上学习波束线数间的增强映射，不把图像宽高直接扩大四倍。

`training` 包含批量、数据线程、Adam、损失权重、回放池、训练轮数和保存周期。默认批量 1、学习率 0.0002、Adam `(0.5,0.999)`、30 轮恒定和 30 轮线性衰减、L1 权重 10、对抗权重 1、回放容量 50。每次运行保存实际参数及验证损失，权重中不保存数据路径。

| `--variant` | 结构 |
| --- | --- |
| `full` | 完整 USR-Mamba |
| `conv_bottleneck` | 纯卷积瓶颈 |
| `no_fourier` | 移除傅里叶分支 |
| `no_refinement` | 移除 SSM 后的残差精炼 |
| `raster_scan` | 使用行优先扫描 |

保持清单与训练设置相同，为各变体分别指定结果目录。批量与线程可按设备资源修改；改变实验参数后应保存并报告实际配置。

## 7. 低线数图像推理

```powershell
python -B run.py --config $Config infer --input work/cascade/images/low --output work/enhanced --checkpoint work/models/full/usr_mamba_060.pt --device $Device
```

论文主路径使用前级去噪后的 32 线图像，参考为原生 128 线图像。原始 32 线对照可另指定输入和输出执行。图像的中性文件名、相对目录和尺寸保持一致。模型结构从所选权重读取，严格核验参数形状。

## 8. 评价与分组统计

提供 `evaluation.csv`，列为 `sample_id,group,reference,result,scale_map,rois,tu,mask`。路径相对 `--data-root`；可选标注留空时，对应指标为空，不自动编造 ROI。每一类方法可分别执行，或在不同清单中比较。

```powershell
python -B run.py --config $Config evaluate --manifest work/evaluation.csv --data-root work --output work/metrics --matlab $Matlab --device $Device
```

全图统计包括 CNR（均值/样本标准差）、ENL、Sobel 梯度相关 EPI、多尺度 LoG 相位响应近似的 FSIM。FSIM 的这个实现口径应与通用相位一致性实现区分。尺度图 CNR 使用连续图：目标 `[0,4)`、背景 `(8,12]`、噪声 `[0,12]`。FWHM 采用亮像素重心漂移、区间扩张和半高交点插值；TU 统计按中心像素归类。阈值、LoG 尺度、漂移和扩张比例在 `evaluation` 中配置。

LPIPS 需额外安装特征评价依赖，并准备公开特征权重：

```powershell
python -m pip install -r requirements-perceptual.txt
python -B run.py --config $Config evaluate --manifest work/evaluation.csv --data-root work --output work/metrics_lpips --matlab $Matlab --device $Device --lpips
```

默认使用 AlexNet 特征，灰度重复为三通道并映射到 `[-1,1]`。首次使用的依赖可能下载预训练特征权重，未包含在归档中。单图结果写入 `metrics.csv`，按 `group` 的均值写入 `group_means.csv`。

## 参数覆盖与核验

不改源码即可覆盖配置项，值按 JSON 解析，选项放在阶段名之前：

```powershell
python -B run.py --config $Config --set 'training.constant_epochs=1' --set 'training.decay_epochs=0' train --manifest work/pairs.csv --data-root work/prepared --output work/models/check --variant full --device $Device
python -B tools/verify_archive.py
```

完整输入、划分、标注和权重均需由数据持有人另行提供。合成数据验证确认实现与接口可运行，不能代替论文全量实验或证明论文表格数值。既有 Git 历史不属于运行数据或本归档文件清单；需要发布时应单独审查历史内容。

可用合成图像验证整条公开调用链，测试目录必须尚不存在：

```powershell
python -B tools/smoke_test.py --config $Config --matlab $Matlab --encoder $Encoder --device $Device --scratch work/synthetic_check
```

测试会生成三对 64×64 图像，依次运行预处理、HEVC、两类去噪、单轮训练、推理和评价。为控制测试成本，合成配置将字典改为 8 个原子、一轮迭代；这组结果不用于论文方法的性能比较。RF 成像与更多数值边界的实测范围见 [验证记录](validation.md)。
