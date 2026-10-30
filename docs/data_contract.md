# 数据契约

所有样本采用 `sample_000001` 形式的中性编号。目录与文件名使用小写字母、数字和下划线；不包含姓名、采集日期、设备账户或原始目录。身份关联表不放入项目。图像像素内若含有身份文字，需在输入前由数据提供者处理。

## 图像

使用单帧、8 位灰度 TIFF。32 线与 128 线配对必须配准并具有相同像素尺寸；文件名相同、低/高线数目录不同。网络输入宽高应能被编码下采样因子整除；默认因子为 4。HEVC 输入为 P5 PGM，宽高为 32 的倍数。

## 训练清单

CSV 使用 UTF-8，表头固定：

```csv
source_id,split,input,reference
sample_000001,train,images/low/sample_000001.tif,images/high/sample_000001.tif
sample_000002,val,images/low/sample_000002.tif,images/high/sample_000002.tif
sample_000003,test,images/low/sample_000003.tif,images/high/sample_000003.tif
```

示例仅表示格式。每个来源只出现一次，集合为 `train`、`val` 或 `test`。路径相对于 `--data-root`，使用正斜杠，禁止绝对路径、目录越界和文件别名跨集合。代码无法仅靠中性编号识别同一患者的不同帧，需先按来源完成划分。

## RF 输入

一个 MAT 文件表示一个来源，顶层变量为 `acquisition`：

| 字段 | 含义 |
| --- | --- |
| `channels` | 实数数组，形状为 `[samples_per_line × beam_count, elements]` |
| `center_frequency_hz` | 中心频率，Hz |
| `sampling_frequency_hz` | 采样频率，Hz |
| `samples_per_line` | 每条扫描线的接收样本数 |
| `transmit_delay_cycles` | 发射延迟的周期数，标量 |

扫描线按行连续排列，阵元按列排列；默认阵元数 128，须与配置一致。`--beams` 指定线数。物理元数据由采集格式适配后提供，程序不读取设备私有字段或推断单位。

## HEVC 先验和尺度图

TU/PU 为无表头 CSV，三列依次为块边长、横坐标、纵坐标，均为整数，原点位于左上角。尺度图 CSV 为与图像同形状的数值矩阵；连续图用于尺度阈值评价，离散图用于块选择。二维坐标和矩阵行列不可互换。

## 评价清单与标注

```csv
sample_id,group,reference,result,scale_map,rois,tu,mask
sample_000003,point,prepared/images/high/sample_000003.tif,enhanced/sample_000003.tif,cascade/scale_maps/low/sample_000003_continuous.csv,annotations/sample_000003_rois.csv,priors/low/sample_000003_tu.csv,annotations/sample_000003_mask.png
```

`reference` 和 `result` 必填，其余标注可留空。`group` 是中性的结构组名，例如 `point`、`band`、`spot`、`void`、`groove`。ROI CSV 每行是 `[x,y,width,height]`，0-based，宽高为像素数；二值掩码与图像同尺寸，正值表示 ROI。每张图可以有多个 ROI，输出同时提供均值；没有 ROI 时 FWHM 为空。

所有路径相对于评价的 `--data-root`。同一清单中的参考、结果、尺度图和标注须处于同一像素坐标系，不允许为对齐而在评价时静默缩放。
