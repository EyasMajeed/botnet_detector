# Botnet Detector — Preprocessing Pipeline

## Overview

This pipeline preprocesses three network traffic datasets into training-ready CSV files for the two-stage botnet detection system:

| Output File | Source Dataset | Model Target | Task |
|---|---|---|---|
| `stage1_iot_vs_noniot.csv` | UNSW-NB15 + CICIDS2017 | Stage-1 Classifier | IoT vs Non-IoT |
| `stage2_iot_botnet.csv` | IoT-23 | Stage-2 IoT CNN-LSTM | Benign vs Botnet |
| `stage2_noniot_botnet.csv` | CTU-13 | Stage-2 Non-IoT CNN-LSTM | Benign vs Botnet |

## Unified Feature Schema

All output CSVs share the same **56-feature schema** (Appendix A of the report):
- **40 Flow-Level Features**: duration, packets, bytes, IATs, flags, ports, protocol
- **6 Time-Window Features**: bytes/sec window, periodicity score, burst rate
- **9 Packet-Level Features**: TTL variation, DNS count, payload stats (zeros when unavailable)
- **1 TLS Flag**: tls_features_available (binary indicator)
- **Labels**: `class_label` (benign/botnet) and `device_type` (iot/noniot)

All features are MinMax-normalized to [0, 1].

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place datasets (see directory structure below)

# 3. Run all preprocessing
python run_all_preprocessing.py

# Or run individually:
python -m src.ingestion.preprocess_ctu13
python -m src.ingestion.preprocess_iot23
python -m src.ingestion.preprocess_unsw_cicids
```

## Dataset Setup

### Directory Structure

```
data/raw/
├── ctu13/
│   ├── capture20110810.binetflow     # Binetflow format
│   ├── capture20110811.binetflow
│   └── ...
│   # OR Kaggle CSV version:
│   └── ctu13_dataset.csv
│
├── iot23/
│   ├── CTU-IoT-Malware-Capture-1-1/
│   │   └── conn.log.labeled          # Zeek conn.log format
│   ├── CTU-IoT-Malware-Capture-3-1/
│   │   └── conn.log.labeled
│   └── ...
│   # OR combined CSV:
│   └── iot23_combined.csv
│
└── unsw_cicids2017/
    ├── unsw/                          # UNSW-NB15 CSV files
    │   ├── UNSW-NB15_1.csv
    │   ├── UNSW-NB15_2.csv
    │   ├── UNSW-NB15_3.csv
    │   └── UNSW-NB15_4.csv
    └── cicids/                        # CICIDS2017 CICFlowMeter CSVs
        ├── Monday-WorkingHours.pcap_Flow.csv
        ├── Tuesday-WorkingHours.pcap_Flow.csv
        └── ...
```

### Download Links

| Dataset | URL |
|---|---|
| CTU-13 | https://www.stratosphereips.org/datasets-ctu13 |
| IoT-23 | https://www.stratosphereips.org/datasets-iot23 |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CICIDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html |

### Converting PCAPs to CSVs

If your datasets are in PCAP format, convert them first:

```bash
# Option A: Use CICFlowMeter (recommended for CICIDS2017)
java -jar CICFlowMeter.jar input.pcap output_dir/

# Option B: Use the included PCAP parser
python -m src.ingestion.pcap_parser -i path/to/pcaps/ -o output.csv

# Option C: Use tshark directly
tshark -r file.pcap -T fields \
  -e frame.time_epoch -e ip.src -e ip.dst \
  -e tcp.srcport -e tcp.dstport -e ip.proto \
  -e frame.len -E separator=, > output.csv
```

## Configuration

Edit `src/config.py` to customize:
- `MAX_ROWS_PER_DATASET`: Set to an integer (e.g., 500000) for faster testing
- `TIME_WINDOW_SEC`: Sliding window width (default: 10s)
- `IOT_IPS`: Add known IoT device IPs for better Stage-1 labeling
- Column mappings for each dataset format

## Pipeline Steps

Each preprocessor follows the same 5-step pipeline:

1. **Load** — Read raw files (binetflow, conn.log, CSV)
2. **Map Columns** — Rename dataset-specific columns to unified schema
3. **Derive Features** — Compute missing stats (bytes/sec, packet lengths, IATs, flags)
4. **Normalize Labels** — Convert to binary (benign/botnet) + device type (iot/noniot)
5. **Feature Pipeline** — Time-window features → schema alignment → cleaning → MinMax scaling

## Output Format

Each output CSV has columns in this exact order:

```
flow_duration, total_fwd_packets, total_bwd_packets, ...,  # 40 flow features
bytes_per_second_window, ..., burst_rate,                    # 6 time-window features
tcp_flag_sequence_entropy, ..., handshake_packet_count,      # 9 packet features
tls_version, ..., tls_features_available,                    # 6 TLS features
class_label, device_type                                     # labels
```

All feature values are float32 in [0, 1]. Labels are strings.
