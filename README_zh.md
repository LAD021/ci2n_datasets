# CI2N 数据集

本仓库包含用于电路图分析和电子元件识别任务的综合数据集。数据集分为四个主要类别，每个类别在电路理解和验证中都有特定用途。

## 数据集概览

```
├─device_identification     # 元件检测和分类
├─device_orientation        # 设备方向分类
├─jumper_identification     # 线路/连接检测
└─validation               # 基于图的电路验证
```

## 数据集描述

### 1. 设备识别数据集

**用途**：电路图中电子元件的目标检测和分类。

**数据内容**：
- **图像文件**：900张PNG格式的电路图（0.png - 899.png）
- **标注文件**：对应的JSON格式标注文件（0.json - 99.json）
- **元件类型**：电阻器(resistor)、电容器(capacitor)、电流源(current)、电压源、端口(port)等

**数据格式示例**：
标注文件采用labelme格式，包含元件的边界框信息：
```json
{
  "shapes": [
    {
      "label": "resistor",
      "points": [[597.0, 298.0], [668.0, 339.0]],
      "shape_type": "rectangle"
    },
    {
      "label": "port",
      "points": [[437.0, 19.0], [456.0, 38.0]],
      "shape_type": "rectangle"
    }
  ],
  "imageHeight": 400,
  "imageWidth": 800
}
```

### 2. 设备方向数据集

**用途**：电路图中电子元件方向的分类。

**数据内容**：
按设备类型和方向组织的图像数据：
- **设备类型**：
  - `train_amp_d/`：放大器方向分类（4个方向）
  - `train_amp_m/`：放大器镜像分类（2个状态）
  - `train_bjt_d/`：BJT晶体管方向分类（4个方向）
  - `train_bjt_m/`：BJT晶体管镜像分类（2个状态）
  - `train_diode/`：二极管方向分类（4个方向）
  - `train_mos/`：MOSFET方向分类（4个方向）
  - `train_switch/`：开关方向分类
  - `train_voltagelines/`：电压线方向分类

**数据组织结构**：
```
train_diode/
├── d/          # 向下方向
│   ├── d1.jpg
│   ├── d2.jpg
│   └── ...
├── l/          # 向左方向
├── r/          # 向右方向
└── u/          # 向上方向

train_amp_m/
├── 0/          # 正常状态
└── 1/          # 镜像状态
```

**分类标签**：
- **4类方向**：`d`（向下）、`l`（向左）、`r`（向右）、`u`（向上）
- **2类镜像**：`0`（正常）、`1`（镜像）

### 3. 跳线识别数据集

**用途**：电路图中线路、连接和跳线的检测与分割。

**数据内容**：
- `images/train/`：训练图像集合
- `images/val/`：验证图像集合
- `labels/`：对应的标注文件

**数据来源**：
- **书籍图像**：来自教科书的电路图（文件名前缀：`book_images_`）
- **论文图像**：来自学术论文的图表（文件名前缀：`paper_images_`）
- **爬虫图像**：网络爬取的电路图（文件名前缀：`spider_images_`）
- **手动截图**：基于截图的标注（文件名前缀：`manual_`）

**数据组织结构**：
```
jumper_identification/
├── images/
│   ├── train/
│   │   ├── book_images_001.png
│   │   ├── paper_images_001.jpg
│   │   ├── spider_images_001.png
│   │   └── manual_001.jpg
│   └── val/
│       ├── book_images_val_001.png
│       └── ...
└── labels/
    ├── train/
    └── val/
```

**标注格式**：
采用YOLO OBB（Oriented Bounding Box，定向边界框）格式，每行包含一个跳线对象的标注信息：
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

其中：
- `class_id`：跳线类别ID
- `x1 y1 x2 y2 x3 y3 x4 y4`：四个角点的归一化坐标（0-1之间）

**类别定义**：
- **0**：水平跳线
- **1**：垂直跳线
- **2**：其他方向跳线

**标注示例**：
```
2 0.780811 0.743961 0.782371 0.74686 0.777691 0.752174 0.776131 0.749758
1 0.331000 0.019000 0.493000 0.260000 0.776000 0.070000 0.613000 -0.171000
```

### 4. 验证数据集

**用途**：使用图编辑距离（GED）指标进行基于图的电路拓扑验证。

**数据内容**：
- `golden/`：标准答案电路表示（JSON格式，65个文件）
- `images/`：对应的电路图图像（PNG格式，127个文件）
- `calc_ged.py`：GED计算脚本
- `utils_ged.py`：图工具和HeteroGraph实现
- `my_networkx.py`：自定义NetworkX扩展

**电路网表格式示例**：
标准答案文件采用JSON格式，包含电路的网表描述：
```json
{
  "ckt_netlist": [
    {
      "component_type": "Res",
      "port_connection": {
        "Neg": "net3",
        "Pos": "VDD"
      }
    },
    {
      "component_type": "NMOS",
      "port_connection": {
        "Drain": "net8",
        "Gate": "net3",
        "Source": "net1"
      }
    }
  ],
  "ckt_type": "DISO-Amplifier"
}
```

**支持的元件类型**：
- `Res`：电阻器
- `NMOS`：N型MOSFET
- `PMOS`：P型MOSFET
- `Capacitor`：电容器
- `Inductor`：电感器
- `Current_Source`：电流源
- `Voltage_Source`：电压源

## 验证工具依赖

验证数据集提供了GED计算工具，需要以下依赖：
- `networkx`：图操作和GED计算
- `torch`：深度学习框架
- `dgl`：深度图库
- `fire`：命令行接口
- `loguru`：日志记录
- `pydantic`：数据验证

安装方式：
```bash
cd validation/
pip install -r requirements.txt
# 或使用uv
uv sync
```

## 数据集统计

| 数据集 | 图像数量 | 标注数量 | 主要内容 | 数据格式 |
|--------|----------|----------|----------|----------|
| 设备识别 | 900张PNG | 100个JSON | 电子元件边界框检测 | labelme格式JSON |
| 设备方向 | 1000+张JPG/PNG | - | 元件方向分类 | 文件夹结构分类 |
| 跳线识别 | 500+张PNG/JPG | 对应标注文件 | 线路连接检测 | 图像+标注文件 |
| 验证 | 127张PNG | 65个JSON | 电路拓扑网表 | JSON网表格式 |

## 引用

如果您在研究中使用这些数据集，请引用：

```bibtex
@dataset{ci2n_datasets,
  title={CI2N: Circuit Diagram Analysis Datasets},
  author={[作者姓名]},
  year={2024},
  url={https://github.com/[仓库地址]}
}
```

## 许可证

详见LICENSE文件。

## 贡献

欢迎贡献！请在提交拉取请求之前阅读贡献指南。