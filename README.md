# CI2N Datasets

[中文版 README](README_zh.md) | [English README](README.md)

This repository contains comprehensive datasets for circuit diagram analysis and electronic component identification tasks. The datasets are organized into four main categories, each serving specific purposes in circuit understanding and validation.

## Dataset Overview

```
├─device_identification     # Component detection and classification
├─device_orientation        # Device orientation classification
├─jumper_identification     # Wire/connection detection
└─validation               # Graph-based circuit validation
```

## Dataset Descriptions

### 1. Device Identification Dataset

**Purpose**: Object detection and classification of electronic components in circuit diagrams.

**Data Content**:
- **Image Files**: 900 PNG format circuit diagrams (0.png - 899.png)
- **Annotation Files**: Corresponding JSON format annotation files (0.json - 99.json)
- **Component Types**: Resistors, capacitors, current sources, voltage sources, ports, etc.

**Data Format Example**:
Annotation files use labelme format, containing component bounding box information:
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

### 2. Device Orientation Dataset

**Purpose**: Classification of electronic component orientations in circuit diagrams.

**Data Content**:
Image data organized by device type and orientation:
- **Device Types**:
  - `train_amp_d/`: Amplifier orientation classification (4 orientations)
  - `train_amp_m/`: Amplifier mirroring classification (2 states)
  - `train_bjt_d/`: BJT transistor orientation classification (4 orientations)
  - `train_bjt_m/`: BJT transistor mirroring classification (2 states)
  - `train_diode/`: Diode orientation classification (4 orientations)
  - `train_mos/`: MOSFET orientation classification (4 orientations)
  - `train_switch/`: Switch orientation classification
  - `train_voltagelines/`: Voltage line orientation classification

**Data Organization Structure**:
```
train_diode/
├── d/          # Down orientation
│   ├── d1.jpg
│   ├── d2.jpg
│   └── ...
├── l/          # Left orientation
├── r/          # Right orientation
└── u/          # Up orientation

train_amp_m/
├── 0/          # Normal state
└── 1/          # Mirrored state
```

**Classification Labels**:
- **4-class orientations**: `d` (down), `l` (left), `r` (right), `u` (up)
- **2-class mirroring**: `0` (normal), `1` (mirrored)

**Data Format**:
- Images in JPG/PNG format
- Folder structure indicates class labels
- File naming: `{orientation}{number}.jpg` (e.g., `d1.jpg`, `l5.jpg`)

### 3. Jumper Identification Dataset

**Purpose**: Detection and segmentation of wires, connections, and jumpers in circuit diagrams.

**Data Content**:
- `images/train/`: Training image collection from multiple sources
- `images/val/`: Validation image collection
- `labels/`: Corresponding annotation files for wire detection

**Data Sources**:
- **Book images**: Circuit diagrams extracted from textbooks
- **Paper images**: Figures from academic papers and publications
- **Spider images**: Web-crawled circuit diagrams from online sources
- **Manual captures**: Screenshot-based annotations and manual drawings

**Data Organization Structure**:
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

**Annotation Format**:
Uses YOLO OBB (Oriented Bounding Box) format, where each line contains annotation information for one jumper object:
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Where:
- `class_id`: Jumper category ID
- `x1 y1 x2 y2 x3 y3 x4 y4`: Normalized coordinates (0-1) of four corner points

**Class Definitions**:
- **0**: Horizontal jumpers
- **1**: Vertical jumpers
- **2**: Other direction jumpers

**Annotation Examples**:
```
2 0.780811 0.743961 0.782371 0.74686 0.777691 0.752174 0.776131 0.749758
1 0.331000 0.019000 0.493000 0.260000 0.776000 0.070000 0.613000 -0.171000
```

### 4. Validation Dataset

**Purpose**: Graph-based validation of circuit topology using Graph Edit Distance (GED) metrics.

**Data Content**:
- `golden/`: Ground truth circuit representations (JSON format, 65 files)
- `images/`: Corresponding circuit diagram images (PNG format, 127 files)
- `calc_ged.py`: GED calculation script
- `utils_ged.py`: Graph utilities and HeteroGraph implementation
- `my_networkx.py`: Custom NetworkX extensions

**Circuit Netlist Format Example**:
Ground truth files use JSON format containing circuit netlist descriptions:
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

**Supported Component Types**:
- `Res`: Resistor
- `NMOS`: N-type MOSFET
- `PMOS`: P-type MOSFET
- `Capacitor`: Capacitor
- `Inductor`: Inductor
- `Current_Source`: Current source
- `Voltage_Source`: Voltage source

## Validation Tool Dependencies

The validation dataset provides GED calculation tools, requiring the following dependencies:
- `networkx`: Graph operations and GED calculation
- `torch`: Deep learning framework
- `dgl`: Deep Graph Library
- `fire`: Command line interface
- `loguru`: Logging
- `pydantic`: Data validation

Installation:
```bash
cd validation/
pip install -r requirements.txt
# Or use uv
uv sync
```

## Dataset Statistics

| Dataset | Image Count | Annotation Count | Main Content | Data Format |
|---------|-------------|------------------|--------------|-------------|
| Device Identification | 900 PNG | 100 JSON | Electronic component bounding box detection | labelme format JSON |
| Device Orientation | 1000+ JPG/PNG | - | Component orientation classification | Folder structure classification |
| Jumper Identification | 500+ PNG/JPG | Corresponding annotation files | Wire connection detection | Image + annotation files |
| Validation | 127 PNG | 65 JSON | Circuit topology netlist | JSON netlist format |

## Citation

If you use these datasets in your research, please cite:

```bibtex
@dataset{ci2n_datasets,
  title={CI2N: Circuit Diagram Analysis Datasets},
  author={[Author Names]},
  year={2024},
  url={https://github.com/[repository]}
}
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.