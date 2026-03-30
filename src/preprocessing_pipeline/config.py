"""
config.py — Central configuration for the Botnet Detection preprocessing pipeline.

Defines:
  - Directory paths for raw / processed data
  - The unified feature schema (Appendix A of the report)
  - Label mappings and dataset metadata
"""

from pathlib import Path

# ──────────────────────────────────────────────
# 1.  DIRECTORY LAYOUT
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Each dataset gets its own subfolder under raw/
RAW_CTU13    = RAW_DIR / "ctu13"
RAW_IOT23    = RAW_DIR / "iot23"
RAW_UNSW_CIC = RAW_DIR / "unsw_cicids2017"

# Processed output paths (training-ready CSVs)
OUT_STAGE1   = PROCESSED_DIR / "stage1_iot_vs_noniot.csv"
OUT_STAGE2_IOT    = PROCESSED_DIR / "stage2_iot_botnet.csv"
OUT_STAGE2_NONIOT = PROCESSED_DIR / "stage2_noniot_botnet.csv"

# ──────────────────────────────────────────────
# 2.  UNIFIED FEATURE SCHEMA  (matches Appendix A)
# ──────────────────────────────────────────────
# A. Flow-Level Statistical Features
FLOW_FEATURES = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_pkt_len_min",
    "fwd_pkt_len_max",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_min",
    "bwd_pkt_len_max",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "fwd_iat_min",
    "fwd_iat_max",
    "fwd_iat_mean",
    "fwd_iat_std",
    "bwd_iat_min",
    "bwd_iat_max",
    "bwd_iat_mean",
    "bwd_iat_std",
    "flow_iat_min",
    "flow_iat_max",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_header_length",
    "bwd_header_length",
    "protocol",
    "src_port",
    "dst_port",
    "flag_FIN",
    "flag_SYN",
    "flag_RST",
    "flag_PSH",
    "flag_ACK",
    "flag_URG",
    "flow_active_time",
    "flow_idle_time",
]

# B. Time-Window Features
TIME_WINDOW_FEATURES = [
    "bytes_per_second_window",
    "packets_per_second_window",
    "connections_per_window",
    "distinct_dst_ips_window",
    "periodicity_score",
    "burst_rate",
]

# C. Packet-Level Features (filled with 0 when not available)
PACKET_FEATURES = [
    "tcp_flag_sequence_entropy",
    "ttl_variation",
    "dns_query_count",
    "payload_size_min",
    "payload_size_max",
    "payload_size_mean",
    "packet_size_variation",
    "retransmission_count",
    "handshake_packet_count",
]

# D. TLS Metadata (filled with 0 / -1 when not available)
TLS_FEATURES = [
    "tls_version",
    "cipher_suite_id",
    "tls_handshake_count",
    "certificate_length",
    "repeated_handshake_attempts",
    "tls_features_available",      # binary flag
]

# E. Labels
LABEL_COLS = [
    "class_label",       # "benign" or "botnet"
    "device_type",       # "iot" or "noniot"
]

# Full ordered feature vector (what the models receive)
ALL_FEATURES = FLOW_FEATURES + TIME_WINDOW_FEATURES + PACKET_FEATURES + TLS_FEATURES

# ──────────────────────────────────────────────
# 3.  COLUMN MAPPINGS PER DATASET
# ──────────────────────────────────────────────

# CTU-13 conn.log / binetflow columns → unified names
CTU13_COL_MAP = {
    "Duration":    "flow_duration",
    "SrcBytes":    "total_fwd_bytes",
    "TotBytes":    "_total_bytes",       # will compute bwd = tot - src
    "TotPkts":     "_total_pkts",
    "SrcPkts":     "total_fwd_packets",
    "Proto":       "protocol",
    "Sport":       "src_port",
    "Dport":       "dst_port",
    "State":       "_state",
    "Label":       "_raw_label",
}

# IoT-23 conn.log columns → unified names
IOT23_COL_MAP = {
    "duration":    "flow_duration",
    "orig_bytes":  "total_fwd_bytes",
    "resp_bytes":  "total_bwd_bytes",
    "orig_pkts":   "total_fwd_packets",
    "resp_pkts":   "total_bwd_packets",
    "proto":       "protocol",
    "id.orig_p":   "src_port",
    "id.resp_p":   "dst_port",
    "conn_state":  "_state",
    "label":       "_raw_label",
    "detailed-label": "_detailed_label",
}

# CICFlowMeter output columns → unified names
# (Used for UNSW+CICIDS2017 after running CICFlowMeter on PCAPs)
CICFLOW_COL_MAP = {
    "Flow Duration":            "flow_duration",
    "Total Fwd Packets":        "total_fwd_packets",
    "Total Backward Packets":   "total_bwd_packets",
    "Total Length of Fwd Packets":  "total_fwd_bytes",
    "Total Length of Bwd Packets":  "total_bwd_bytes",
    "Fwd Packet Length Min":    "fwd_pkt_len_min",
    "Fwd Packet Length Max":    "fwd_pkt_len_max",
    "Fwd Packet Length Mean":   "fwd_pkt_len_mean",
    "Fwd Packet Length Std":    "fwd_pkt_len_std",
    "Bwd Packet Length Min":    "bwd_pkt_len_min",
    "Bwd Packet Length Max":    "bwd_pkt_len_max",
    "Bwd Packet Length Mean":   "bwd_pkt_len_mean",
    "Bwd Packet Length Std":    "bwd_pkt_len_std",
    "Flow Bytes/s":             "flow_bytes_per_sec",
    "Flow Packets/s":           "flow_pkts_per_sec",
    "Fwd IAT Min":              "fwd_iat_min",
    "Fwd IAT Max":              "fwd_iat_max",
    "Fwd IAT Mean":             "fwd_iat_mean",
    "Fwd IAT Std":              "fwd_iat_std",
    "Bwd IAT Min":              "bwd_iat_min",
    "Bwd IAT Max":              "bwd_iat_max",
    "Bwd IAT Mean":             "bwd_iat_mean",
    "Bwd IAT Std":              "bwd_iat_std",
    "Flow IAT Min":             "flow_iat_min",
    "Flow IAT Max":             "flow_iat_max",
    "Flow IAT Mean":            "flow_iat_mean",
    "Flow IAT Std":             "flow_iat_std",
    "Fwd Header Length":        "fwd_header_length",
    "Bwd Header Length":        "bwd_header_length",
    "Protocol":                 "protocol",
    "Source Port":              "src_port",
    "Destination Port":         "dst_port",
    "FIN Flag Count":           "flag_FIN",
    "SYN Flag Count":           "flag_SYN",
    "RST Flag Count":           "flag_RST",
    "PSH Flag Count":           "flag_PSH",
    "ACK Flag Count":           "flag_ACK",
    "URG Flag Count":           "flag_URG",
    "Active Mean":              "flow_active_time",
    "Idle Mean":                "flow_idle_time",
    "Label":                    "_raw_label",
}

# UNSW-NB15 CSV columns → unified names
UNSW_COL_MAP = {
    "dur":         "flow_duration",
    "spkts":       "total_fwd_packets",
    "dpkts":       "total_bwd_packets",
    "sbytes":      "total_fwd_bytes",
    "dbytes":      "total_bwd_bytes",
    "proto":       "protocol",
    "sport":       "src_port",
    "dsport":      "dst_port",
    "sttl":        "_src_ttl",
    "dttl":        "_dst_ttl",
    "ct_srv_src":  "_ct_srv_src",
    "ct_srv_dst":  "_ct_srv_dst",
    "label":       "_raw_label",
    "attack_cat":  "_attack_cat",
}


# ──────────────────────────────────────────────
# 4.  PROTOCOL ENCODING
# ──────────────────────────────────────────────
PROTO_MAP = {
    "tcp": 6, "udp": 17, "icmp": 1,
    "igmp": 2, "arp": 0, "ipv6-icmp": 58,
    "6": 6, "17": 17, "1": 1,
}

# ──────────────────────────────────────────────
# 5.  PROCESSING PARAMETERS
# ──────────────────────────────────────────────
RANDOM_SEED   = 42
TIME_WINDOW_SEC = 10          # sliding-window width for time-window features
MAX_ROWS_PER_DATASET = 1000000   # set to int to subsample (useful for testing)
