# ReadMe

This aaa

## Environment Setup

We use uv to set up the invironment.

### Install uv

See [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

### Initialize

```shell

uv sync

```

## File Structure

```
├─golden/  # golden netlists
├─images/  # validation set
├─calc_ged.py  # entry file
├─device_def.toml  # definition of devices and ports
├─my_networkx.py  # a modify for ged alogrithm in networkx, which can accelerate for 50%
└─utils_ged.py  # ged algorithm
```

## Run
Put your result in a input directory, then rename all your answers as in golden_dir.

```shell
uv run python calc_ged.py --input_dir <input_dir> --golden_dir golden --output_dir <output_dir> --timeout <timout>
```

The report will be in `<output_dir>`