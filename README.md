# drive-c-dataset
DRIVE-C is a controlled degradation dataset designed for evaluating perception robustness, degradation awareness, and sensor health estimation in autonomous driving systems.

The dataset combines **real-world driving data** with **physics-inspired synthetic degradations** applied across multiple severity levels, enabling reproducible and structured robustness evaluation. 

Please refer ```/dataset/README.md``` file for link to download the dataset.

---

## Highlights

- 10 real-world driving scenarios  
- 12 corruption types (weather, optical, sensor, compression)  
- 5 severity levels (s1–s5) + clean  
- 610 video clips (~78K frames)  
- Includes **Global Sensor Health Index (GSHI)** ground truth and predictions  

---

## License

- Code is licensed under the MIT License.
- Dataset is licensed under the Creative Commons Attribution 4.0 (CC BY 4.0) License.

Please cite the dataset and associated paper when used.

## 📂 Repository Structure

```
drive-c-dataset/
README.md
LICENSE # MIT (code)
requirements.txt
scripts/ # dataset generation & processing
simulation/ # corruption models
configs/ # configuration files
dataset/ # readme and dataset source
```

## Environment

Recommended Python version:
- Python 3.10+

Main Python packages:
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- pillow
- faiss
- openpyxl (optional)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Instruction to build the new dataset

Inside each script, use this pattern:

```
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]   # DRIVE-C-DATASET
DATASET_ROOT = PROJECT_ROOT / "dataset"
CONFIG_ROOT = PROJECT_ROOT / "configs"

sys.path.insert(0, str(PROJECT_ROOT))
```

That way all scripts work no matter where you launch them from, as long as you run them inside the repo.

## Recommended dataset folder layout

Please refer ```/dataset/README.md``` file for link to download the dataset.

Inside dataset/, use this structure:

```
dataset/
  clean_clips/
  corrupted/
  final_metadata.csv
  samples_metadata.csv
  samples_metadata_with_gshi.csv
  samples_metadata_with_gshi_pred.csv
  scenario_metadata.csv
  gshi_pred_sanity_report.txt
  gshi_pred_per_corruption_stats.csv
  figures/
```

### Recommended config layout
```
configs/
  taxonomy/
    camera_issues.yaml
```

That camera_issues.yaml should be the same one used for gshi_gt.

## Quick Start

From the repository root:

```bash
pip install -r requirements.txt

python scripts/unified_generate_drivec.py
python scripts/add_gshi_gt.py
python scripts/add_gshi_pred.py --ckpt checkpoints/epoch_021_best.pth
python scripts/make_final_metadata.py
python scripts/analyze_gshi_pred.py
python scripts/make_benchmark_figures.py
```

## Generate corrupted clips and base metadata

This script should read clean source clips, source metadata, corruption configs and
write:
dataset/clean_clips/
dataset/corrupted/
dataset/samples_metadata.csv


Run for complete dataset:
```
python scripts/unified_generate_drivec.py
```

Run for a small test:
```
python scripts/unified_generate_drivec.py --scenarios S01 --severity-names s3
```

## Add ground-truth GSHI

This should read:

dataset/samples_metadata.csv
configs/taxonomy/camera_issues.yaml

and write:

dataset/samples_metadata_with_gshi.csv

Run:
```
python scripts/add_gshi_gt.py
```

If you want to be explicit:
```
python scripts/add_gshi_gt.py \
  --input-csv dataset/samples_metadata.csv \
  --output-csv dataset/samples_metadata_with_gshi.csv \
  --taxonomy-yaml configs/taxonomy/camera_issues.yaml
```

## Add predicted GSHI and degradation predictions

This should read:

dataset/samples_metadata_with_gshi.csv
model checkpoint

and write:

dataset/samples_metadata_with_gshi_pred.csv

Run:
```
python scripts/add_gshi_pred.py \
  --ckpt /path/to/checkpoint.pth
```

## Merge GT and predictions into final metadata

This should read:

dataset/samples_metadata_with_gshi.csv
dataset/samples_metadata_with_gshi_pred.csv

and write:

dataset/final_metadata.csv

Run:
```
python scripts/make_final_metadata.py
```

Explicit version:
```
python scripts/make_final_metadata.py \
  --gt-csv dataset/samples_metadata_with_gshi.csv \
  --pred-csv dataset/samples_metadata_with_gshi_pred.csv \
  --out-csv dataset/final_metadata.csv
```

## Run sanity analysis on predictions

This should read:

dataset/final_metadata.csv

and write:

dataset/gshi_pred_sanity_report.txt
dataset/gshi_pred_per_corruption_stats.csv

Run:
```
python scripts/analyze_gshi_pred.py
```