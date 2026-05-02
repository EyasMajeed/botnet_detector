"""
══════════════════════════════════════════════════════════════════════════
 MULTI-SOURCE PREPROCESSING PIPELINE (v6)
 preprocess_from_pcap_csvs.py
══════════════════════════════════════════════════════════════════════════
 Group 07 | CPCS499

 ALIGNMENT TARGET: monitoring.py
 ────────────────────────────────────────────────────────────
 v6 produces training data that EXACTLY matches the live feature
 extractor in monitoring.py. This is mandatory because monitoring.py
 is the integration point for the Stage-1 model.

 KEY ALIGNMENT DECISIONS (vs v5):

   1. FEATURE COUNT: 56 (not 49)
      Schema matches S1_FEATURES in monitoring.py line 168.
      We re-introduce the 7 features v5 dropped:
        flag_URG, ttl_std, ttl_min, ttl_max, tls_features_available,
        window_flow_count, window_unique_dsts

   2. NO SCALING IN PREPROCESSOR
      MinMaxScaler removed. The trainer fits StandardScaler ONCE
      on raw training data and saves s1_scaler.json. monitoring.py
      loads that JSON and applies (x - mean) / scale at inference.

   3. NO log1p TRANSFORMS
      Live monitoring.py applies StandardScaler to raw values directly.
      If we log1p in training but not in live, training and inference
      see different feature distributions. So we keep raw values.

   4. FORCED-CONSTANT FEATURES
      Six features are always constant in live monitoring.py. To prevent
      training-inference distribution shift, we override them in training:
        periodicity_score    : forced to 0.0  (live always 0)
        burst_rate           : forced to 0.0  (live always 0)
        payload_zero_ratio   : forced to 0.0  (live always 0)
        payload_entropy      : forced to 0.0  (live always 0)
        fwd_header_length    : forced to 20.0 (live always 20)
        bwd_header_length    : forced to 20.0 (live always 20)
        window_flow_count    : forced to 1.0  (live always 1)
        window_unique_dsts   : forced to 1.0  (live always 1)

      These features will have zero variance in training, which means
      StandardScaler will see scale=0 and skip them gracefully (the
      monitoring.py loader already handles scale==0 at line 515).

   5. tls_features_available IS COMPUTED
      Live monitoring.py sets this to 1 if port ∈ {443, 8443, 8883}.
      We replicate that exactly from src_port and dst_port.

 EIGHT SOURCES IN v6:
   IoT side (4):
     1. iot23           - IoT-23 (large, malware-heavy)
     2. etf_malware     - ETF Mendeley malware/ folder
     3. ieee_benign     - IEEE IoT Network Intrusion benign-dec.pcap
     4. ieee_mirai      - IEEE IoT Network Intrusion mirai-hostbruteforce-1

   NonIoT side (4):
     5. ctu13           - CTU-13 (university captures)
     6. cicids2017      - CIC-IDS-2017 friday_flows.csv
     7. cicids2018      - CSE-CIC-IDS2018 (after import_cicids2018.py)
     8. etf_benigniot   - ETF Mendeley benigniot/ folder (PC traffic)

 OUTPUT:
   data/processed/stage1_train.csv  - raw values, 56 features + class_label + source_dataset
   data/processed/stage1_test.csv   - same schema, held-out scenarios
   data/processed/stage1_scenario_allocation.txt - methodology log

 NORMALIZATION HAPPENS LATER:
   The trainer (classifier.py v6) fits StandardScaler on stage1_train.csv,
   transforms both train and test, and saves the scaler to s1_scaler.json
   in the format monitoring.py expects.
══════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent

# IoT side
PCAP_CSV_IOT23   = BASE_DIR / "data" / "pcap_csv" / "iot23"           / "iot23_all_flows.csv"
PCAP_CSV_ETF_MAL = BASE_DIR / "data" / "pcap_csv" / "etf_malware"     / "etf_malware_all_flows.csv"
PCAP_CSV_IEEE_B  = BASE_DIR / "data" / "pcap_csv" / "iot_ieee_benign" / "ieee_benign_all_flows.csv"
PCAP_CSV_IEEE_M  = BASE_DIR / "data" / "pcap_csv" / "iot_ieee_mirai"  / "ieee_mirai_all_flows.csv"

# NonIoT side
PCAP_CSV_CTU13          = BASE_DIR / "data" / "pcap_csv"  / "ctu13"      / "ctu13_all_flows.csv"
PROCESSED_FRIDAY_FLOWS  = BASE_DIR / "data" / "processed" / "friday_flows.csv"
PROCESSED_CICIDS2018    = BASE_DIR / "data" / "processed" / "cicids2018_processed.csv"
PCAP_CSV_ETF_BEN        = BASE_DIR / "data" / "pcap_csv"  / "etf_benign" / "etf_benigniot_all_flows.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

RANDOM_SEED            = 42
TIME_WINDOW_SEC        = 10.0
TEST_SCENARIO_FRACTION = 0.25
MIN_TEST_SCENARIOS     = 2
MIN_TRAIN_SCENARIOS    = 3
SYNTHETIC_SCENARIO_BLOCKS = 8   # split single-scenario sources into N synthetic scenarios

# Per-source size cap during training (prevents IoT-23 from drowning small sources)
PER_SOURCE_TRAIN_CAP   = 80_000

# Test set class balance target — cap so test isn't 97% one class
PER_SOURCE_TEST_CAP    = 15_000


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED 56-FEATURE SCHEMA — must match S1_FEATURES in monitoring.py
# ═══════════════════════════════════════════════════════════════════════

S1_FEATURES = [
    "flow_duration",
    "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes",   "total_bwd_bytes",
    "fwd_pkt_len_min", "fwd_pkt_len_max", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_min", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "flow_bytes_per_sec", "flow_pkts_per_sec",
    "flow_iat_mean", "flow_iat_std", "flow_iat_min", "flow_iat_max",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_min", "fwd_iat_max",
    "bwd_iat_mean", "bwd_iat_std", "bwd_iat_min", "bwd_iat_max",
    "fwd_header_length", "bwd_header_length",
    "flag_FIN", "flag_SYN", "flag_RST", "flag_PSH", "flag_ACK", "flag_URG",
    "protocol", "src_port", "dst_port",
    "flow_active_time", "flow_idle_time",
    "bytes_per_sec_window", "pkts_per_sec_window",
    "periodicity_score", "burst_rate",
    "window_flow_count", "window_unique_dsts",
    "ttl_mean", "ttl_std", "ttl_min", "ttl_max",
    "dns_query_count",
    "payload_bytes_mean", "payload_bytes_std",
    "payload_zero_ratio", "payload_entropy",
    "tls_features_available",
]
assert len(S1_FEATURES) == 56, f"Schema has {len(S1_FEATURES)} features, must be 56"

# Features that monitoring.py forces to constant values in live inference.
# We match these in training to prevent training/inference distribution shift.
LIVE_CONSTANT_FEATURES = {
    "fwd_header_length":     20.0,   # live: hardcoded 20.0
    "bwd_header_length":     20.0,   # live: hardcoded 20.0
    "periodicity_score":      0.0,   # live: always 0.0
    "burst_rate":             0.0,   # live: always 0.0
    "window_flow_count":      1.0,   # live: default 1
    "window_unique_dsts":     1.0,   # live: default 1
    "payload_zero_ratio":     0.0,   # live: always 0.0
    "payload_entropy":        0.0,   # live: always 0.0
}

# TLS port set used by monitoring.py to set tls_features_available
TLS_PORTS = {443, 8443, 8883}

# Column rename maps for raw pcap_to_csv outputs
PCAP_RENAME_MAP = {
    "active_mean": "flow_active_time",
    "idle_mean":   "flow_idle_time",
}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE-DERIVATION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def derive_window_features(df):
    """Compute bytes_per_sec_window and pkts_per_sec_window."""
    dur = df["flow_duration"].replace(0, np.nan)
    df["bytes_per_sec_window"] = (
        (df["total_fwd_bytes"] + df["total_bwd_bytes"]) / dur.clip(lower=1e-6)
    ).fillna(0)
    df["pkts_per_sec_window"] = (
        (df["total_fwd_packets"] + df["total_bwd_packets"]) / dur.clip(lower=1e-6)
    ).fillna(0)
    return df


def _safe_get_series(df, col, default=0.0):
    """
    Safely get a column as a numeric Series. If the column is missing,
    return a Series of `default` values with the same index as df.

    This fixes the bug where df.get(col, 0) returns int 0 (not a Series)
    when the column is absent — breaking downstream .fillna() calls.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def derive_ttl_features(df):
    """Derive ttl_mean, ttl_std, ttl_min, ttl_max from pcap output columns.

    Sources lacking TTL data (CIC-IDS-2017, CSE-CIC-IDS-2018) get all
    four features set to 0.0 — that's an acknowledged limitation of
    CICFlowMeter, not a bug here.
    """
    fwd_ttl = _safe_get_series(df, "fwd_avg_ttl", 0.0)
    bwd_ttl = _safe_get_series(df, "bwd_avg_ttl", 0.0)
    df["ttl_mean"] = (fwd_ttl + bwd_ttl) / 2

    # Approximate std/min/max from the two-direction averages
    # This is NOT a real per-packet TTL std — it's the spread between fwd
    # and bwd averages. Live monitoring.py computes real per-packet stats
    # so values will differ. Best we can do without per-packet data.
    df["ttl_min"] = np.minimum(fwd_ttl, bwd_ttl)
    df["ttl_max"] = np.maximum(fwd_ttl, bwd_ttl)
    df["ttl_std"] = (df["ttl_max"] - df["ttl_min"]) / 2.0  # rough approximation
    return df


def derive_payload_stats(df):
    """Derive payload_bytes_mean and payload_bytes_std (the only payload features
    that aren't forced to 0 by live monitoring.py)."""
    if "pkt_len_mean" in df.columns:
        df["payload_bytes_mean"] = _safe_get_series(df, "pkt_len_mean", 0.0)
        df["payload_bytes_std"]  = _safe_get_series(df, "pkt_len_std",  0.0)
    elif "fwd_pkt_len_mean" in df.columns:
        df["payload_bytes_mean"] = _safe_get_series(df, "fwd_pkt_len_mean", 0.0)
        df["payload_bytes_std"]  = _safe_get_series(df, "fwd_pkt_len_std",  0.0)
    else:
        df["payload_bytes_mean"] = 0.0
        df["payload_bytes_std"]  = 0.0
    return df


def derive_dns_query_count(df):
    """Live: 1 if dst_port==53 OR src_port==53. Match exactly."""
    if "dst_port" in df.columns and "src_port" in df.columns:
        df["dns_query_count"] = ((df["dst_port"] == 53) | (df["src_port"] == 53)).astype(float)
    elif "dst_port" in df.columns:
        df["dns_query_count"] = (df["dst_port"] == 53).astype(float)
    else:
        df["dns_query_count"] = 0.0
    return df


def derive_tls_features_available(df):
    """Live: 1 if dst_port OR src_port in {443, 8443, 8883}."""
    if "dst_port" in df.columns and "src_port" in df.columns:
        df["tls_features_available"] = (
            df["dst_port"].isin(TLS_PORTS) | df["src_port"].isin(TLS_PORTS)
        ).astype(float)
    else:
        df["tls_features_available"] = 0.0
    return df


def force_live_constants(df):
    """Override features that live monitoring.py hardcodes to constants."""
    for col, val in LIVE_CONSTANT_FEATURES.items():
        df[col] = val
    return df


def align_to_schema(df):
    """Ensure all 56 schema columns exist; fill missing with 0."""
    for col in S1_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def clean_df(df):
    """Cast feature cols to float, replace NaN/Inf, deduplicate."""
    feat_cols = [c for c in S1_FEATURES if c in df.columns]
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], 0)

    n_before = len(df)
    df = df.drop_duplicates(subset=feat_cols, keep="first").reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"      Deduplication: removed {n_dupes:,} duplicates "
              f"({n_dupes/n_before*100:.1f}%)")

    df[feat_cols] = df[feat_cols].astype(np.float32)
    return df


def standardize_source(df):
    """
    Apply the full feature-engineering pipeline that brings a raw
    pcap_to_csv DataFrame into the 56-feature schema, with values
    that exactly match what monitoring.py computes live.
    """
    # Rename pcap-output columns to schema names
    rename = {k: v for k, v in PCAP_RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Derive features that aren't direct matches
    df = derive_window_features(df)
    df = derive_ttl_features(df)
    df = derive_payload_stats(df)
    df = derive_dns_query_count(df)
    df = derive_tls_features_available(df)

    # Force constants that live inference hardcodes
    df = force_live_constants(df)

    # Schema alignment (fills missing with 0)
    df = align_to_schema(df)

    return df


def stratified_subsample(df, label_col, n_target, seed):
    """Subsample preserving label_col ratios."""
    if len(df) <= n_target:
        return df
    if label_col is None or label_col not in df.columns:
        return df.sample(n_target, random_state=seed).reset_index(drop=True)
    return (
        df.groupby(label_col, group_keys=False)
          .apply(lambda g: g.sample(
              min(len(g), int(round(n_target * len(g) / len(df)))),
              random_state=seed))
          .reset_index(drop=True)
    )


def add_synthetic_scenarios(df, source_name, n_blocks=SYNTHETIC_SCENARIO_BLOCKS):
    """
    Sources without natural scenarios (or with too few) get N synthetic
    scenarios by deterministic row-block partitioning. Each block gets
    a unique scenario_id so test allocation can pick some out.
    """
    df = df.copy()
    df["_block_idx"] = np.arange(len(df)) % n_blocks
    df["scenario_id"] = (
        f"{source_name}_synth_" + df["_block_idx"].astype(str)
    )
    df = df.drop(columns=["_block_idx"])
    return df


def allocate_scenarios(df, source_name, scenario_col="scenario_id",
                       seed=RANDOM_SEED):
    """Pick scenarios for train/test split."""
    if scenario_col not in df.columns:
        return None, None

    scenarios = sorted(df[scenario_col].dropna().unique().tolist())
    n_total = len(scenarios)

    if n_total < (MIN_TEST_SCENARIOS + MIN_TRAIN_SCENARIOS):
        # Force synthetic scenarios if too few
        df_synth = add_synthetic_scenarios(df, source_name)
        scenarios = sorted(df_synth[scenario_col].dropna().unique().tolist())
        n_total = len(scenarios)
        df.update(df_synth)
        df[scenario_col] = df_synth[scenario_col].values

    rng = np.random.default_rng(seed)
    n_test = max(MIN_TEST_SCENARIOS, int(round(n_total * TEST_SCENARIO_FRACTION)))
    n_test = min(n_test, n_total - MIN_TRAIN_SCENARIOS)

    test_idx = rng.choice(n_total, size=n_test, replace=False)
    test_set = set(scenarios[i] for i in test_idx)
    train_scens = [s for s in scenarios if s not in test_set]
    test_scens  = [s for s in scenarios if s in test_set]

    print(f"    [{source_name}] {n_total} scenarios → "
          f"{len(train_scens)} train, {len(test_scens)} test")
    return train_scens, test_scens


# ═══════════════════════════════════════════════════════════════════════
# SOURCE LOADERS — return DataFrame in 56-feature schema, raw values
# ═══════════════════════════════════════════════════════════════════════

def _load_pcap_source(path, source_name, side, side_label, internal_label_fn=None):
    """Generic loader for pcap_to_csv outputs."""
    if not path.exists():
        print(f"  ✗ {source_name}: not found at {path}")
        return None

    print(f"\n  Loading {source_name} from {path.name}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"    Loaded {len(df):,} flows")

    df["device_type"]    = side
    df["source_dataset"] = source_name
    df["class_label"]    = side_label  # Stage-1 label = "iot" or "noniot"

    # Internal benign/botnet label for stratified sampling
    if internal_label_fn is not None:
        df["_internal_label"] = internal_label_fn(df)
    else:
        df["_internal_label"] = "unknown"

    # Ensure scenario_id exists; force synthetic blocks for small sources
    if "scenario_id" not in df.columns or df["scenario_id"].nunique() < MIN_TRAIN_SCENARIOS + MIN_TEST_SCENARIOS:
        df = add_synthetic_scenarios(df, source_name)

    df = standardize_source(df)
    return df


def _iot23_internal_label(df):
    raw = df.get("label", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    return np.where(
        raw.str.contains("malicious|botnet|c&c|ddos|attack|portscan|"
                         "okiru|torii|mirai|hajime|kenjiro|muhstik|"
                         "hakai|linux|irc|spam|scan", regex=True, na=False),
        "botnet",
        np.where(raw.str.contains("benign|normal|legitimate", regex=True, na=False),
                 "benign", "unknown")
    )


def _ctu13_internal_label(df):
    label = pd.to_numeric(df.get("label", pd.Series(dtype=float)), errors="coerce")
    return label.map({1.0: "botnet", 0.0: "benign"}).fillna("unknown").astype(str).values


def load_iot23():
    df = _load_pcap_source(PCAP_CSV_IOT23, "iot23", "iot", "iot",
                           internal_label_fn=_iot23_internal_label)
    return df


def load_etf_malware():
    """ETF malware folder = real IoT botnet samples."""
    df = _load_pcap_source(PCAP_CSV_ETF_MAL, "etf_malware", "iot", "iot",
                           internal_label_fn=lambda d: np.full(len(d), "botnet"))
    return df


def load_ieee_benign():
    """IEEE IoT Network Intrusion benign-dec.pcap = real home Wi-Fi IoT traffic."""
    df = _load_pcap_source(PCAP_CSV_IEEE_B, "ieee_benign", "iot", "iot",
                           internal_label_fn=lambda d: np.full(len(d), "benign"))
    return df


def load_ieee_mirai():
    """IEEE IoT Network Intrusion mirai-hostbruteforce-1 = Mirai brute-force."""
    df = _load_pcap_source(PCAP_CSV_IEEE_M, "ieee_mirai", "iot", "iot",
                           internal_label_fn=lambda d: np.full(len(d), "botnet"))
    return df


def load_ctu13():
    if not PCAP_CSV_CTU13.exists():
        print(f"  ✗ ctu13: not found at {PCAP_CSV_CTU13}")
        return None
    print(f"\n  Loading ctu13 from {PCAP_CSV_CTU13.name}...")
    df = pd.read_csv(PCAP_CSV_CTU13, low_memory=False)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    n_before = len(df)
    df = df[df["label"].isin([0, 1])].copy()
    print(f"    Loaded {len(df):,} flows (removed {n_before - len(df):,} background)")

    df["device_type"]     = "noniot"
    df["source_dataset"]  = "ctu13"
    df["class_label"]     = "noniot"
    df["_internal_label"] = df["label"].map({1: "botnet", 0: "benign"}).astype(str)

    if "scenario_id" not in df.columns:
        df = add_synthetic_scenarios(df, "ctu13")

    df = standardize_source(df)
    return df


def load_friday_flows():
    if not PROCESSED_FRIDAY_FLOWS.exists():
        print(f"  ✗ cicids2017: not found at {PROCESSED_FRIDAY_FLOWS}")
        return None
    print(f"\n  Loading cicids2017 from {PROCESSED_FRIDAY_FLOWS.name}...")
    df = pd.read_csv(PROCESSED_FRIDAY_FLOWS, low_memory=False)
    print(f"    Loaded {len(df):,} flows")

    df["device_type"]     = "noniot"
    df["source_dataset"]  = "cicids2017"
    df["_internal_label"] = df.get("class_label", "unknown").astype(str).values
    df["class_label"]     = "noniot"

    df = add_synthetic_scenarios(df, "cicids2017")
    df = standardize_source(df)
    return df


def load_cicids2018():
    if not PROCESSED_CICIDS2018.exists():
        print(f"  ✗ cicids2018: not found at {PROCESSED_CICIDS2018}")
        print(f"      Run import_cicids2018.py first")
        return None
    print(f"\n  Loading cicids2018 from {PROCESSED_CICIDS2018.name}...")
    df = pd.read_csv(PROCESSED_CICIDS2018, low_memory=False)
    print(f"    Loaded {len(df):,} flows")

    df["device_type"]     = "noniot"
    df["source_dataset"]  = "cicids2018"
    df["_internal_label"] = df.get("class_label", "unknown").astype(str).values
    df["class_label"]     = "noniot"

    df = add_synthetic_scenarios(df, "cicids2018")
    df = standardize_source(df)
    return df


def load_etf_benigniot():
    """ETF benigniot folder = NonIoT benign PC traffic (folder name is misleading)."""
    df = _load_pcap_source(PCAP_CSV_ETF_BEN, "etf_benigniot", "noniot", "noniot",
                           internal_label_fn=lambda d: np.full(len(d), "benign"))
    return df


# ═══════════════════════════════════════════════════════════════════════
# MAIN MULTI-SOURCE PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════

def preprocess_stage1_v6():
    print("\n" + "=" * 70)
    print("MULTI-SOURCE STAGE-1 PREPROCESSING (v6 — monitoring.py aligned)")
    print("=" * 70)

    # ── Load all sources ──────────────────────────────────────────────
    sources = {}
    for name, loader in [
        ("iot23",          load_iot23),
        ("etf_malware",    load_etf_malware),
        ("ieee_benign",    load_ieee_benign),
        ("ieee_mirai",     load_ieee_mirai),
        ("ctu13",          load_ctu13),
        ("cicids2017",     load_friday_flows),
        ("cicids2018",     load_cicids2018),
        ("etf_benigniot",  load_etf_benigniot),
    ]:
        df = loader()
        if df is not None and len(df) > 0:
            sources[name] = df

    iot_sources    = [n for n, d in sources.items() if d["device_type"].iloc[0] == "iot"]
    noniot_sources = [n for n, d in sources.items() if d["device_type"].iloc[0] == "noniot"]

    print(f"\n  IoT sources    ({len(iot_sources)}): {iot_sources}")
    print(f"  NonIoT sources ({len(noniot_sources)}): {noniot_sources}")

    if not iot_sources or not noniot_sources:
        print("\n  ✗ Need at least one IoT and one NonIoT source")
        return False

    # ── Per-source scenario allocation ────────────────────────────────
    print("\n  ── Per-source scenario allocation ──────────────────────────")
    train_frames = []
    test_frames  = []

    for name, df in sources.items():
        train_scens, test_scens = allocate_scenarios(df, name, "scenario_id")
        if train_scens is None:
            continue

        train_part = df[df["scenario_id"].isin(train_scens)].copy()
        test_part  = df[df["scenario_id"].isin(test_scens)].copy()

        # Apply per-source caps to prevent any single source from dominating
        if len(train_part) > PER_SOURCE_TRAIN_CAP:
            train_part = stratified_subsample(
                train_part, "_internal_label",
                PER_SOURCE_TRAIN_CAP, RANDOM_SEED
            )

        if len(test_part) > PER_SOURCE_TEST_CAP:
            test_part = stratified_subsample(
                test_part, "_internal_label",
                PER_SOURCE_TEST_CAP, RANDOM_SEED
            )

        print(f"    [{name}]   train: {len(train_part):>7,}   test: {len(test_part):>7,}")
        train_frames.append(train_part)
        test_frames.append(test_part)

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df  = pd.concat(test_frames,  ignore_index=True)

    # ── Verify NO source-scenario overlap ─────────────────────────────
    train_keys = set(zip(train_df["source_dataset"], train_df["scenario_id"]))
    test_keys  = set(zip(test_df["source_dataset"],  test_df["scenario_id"]))
    overlap = train_keys & test_keys
    if overlap:
        print(f"\n  ✗ CRITICAL: {len(overlap)} source-scenario pairs overlap!")
        for o in list(overlap)[:5]:
            print(f"      {o}")
        return False

    print(f"\n    ✓ No source-scenario overlap "
          f"(train: {len(train_keys)} groups, test: {len(test_keys)} groups)")

    # ── Final summary ──
    print("\n  ── Final composition ───────────────────────────────────────")
    print(f"    Training: {len(train_df):,} rows")
    print(f"      By class:")
    for cls, cnt in train_df["class_label"].value_counts().items():
        print(f"        {cls:>8s}: {cnt:>9,}  ({cnt/len(train_df)*100:.1f}%)")
    print(f"      By source:")
    for src, cnt in train_df["source_dataset"].value_counts().items():
        print(f"        {src:>14s}: {cnt:>9,}  ({cnt/len(train_df)*100:.1f}%)")

    print(f"\n    Test: {len(test_df):,} rows")
    print(f"      By class:")
    for cls, cnt in test_df["class_label"].value_counts().items():
        print(f"        {cls:>8s}: {cnt:>9,}  ({cnt/len(test_df)*100:.1f}%)")
    print(f"      By source:")
    for src, cnt in test_df["source_dataset"].value_counts().items():
        print(f"        {src:>14s}: {cnt:>9,}  ({cnt/len(test_df)*100:.1f}%)")

    # ── Clean (NaN, Inf, dedup) but DO NOT scale ──
    print("\n  ── Cleaning training set ──────────────────────────────────")
    train_df = clean_df(train_df)
    print("\n  ── Cleaning test set ──────────────────────────────────────")
    test_df = clean_df(test_df)

    if len(train_df) == 0 or len(test_df) == 0:
        print("\n  ✗ One split has zero rows after cleaning")
        return False

    # ── Drop internal columns ──
    drop_cols = ["_internal_label"]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df  = test_df.drop(columns=drop_cols, errors="ignore")

    # ── Readiness checks ──
    print("\n  ── Stage-1 readiness checks ───────────────────────────────")
    all_ok = True
    for set_name, ds in [("train", train_df), ("test", test_df)]:
        labels = set(ds["class_label"].unique())
        if labels == {"iot", "noniot"}:
            n_iot  = (ds["class_label"] == "iot").sum()
            n_non  = (ds["class_label"] == "noniot").sum()
            print(f"     ✓ {set_name}: both classes present "
                  f"({n_iot:,} iot, {n_non:,} noniot)")
        else:
            print(f"     ✗ {set_name}: classes = {labels}")
            all_ok = False

        feat_cols = [c for c in S1_FEATURES if c in ds.columns]
        if len(feat_cols) != 56:
            print(f"     ✗ {set_name}: {len(feat_cols)}/56 features present")
            all_ok = False
        else:
            print(f"     ✓ {set_name}: all 56 features present")

        if ds[feat_cols].isna().sum().sum() == 0:
            print(f"     ✓ {set_name}: no NaN")
        else:
            print(f"     ✗ {set_name}: has NaN")
            all_ok = False

        # Confirm forced-constant features really are constant
        for col, expected_val in LIVE_CONSTANT_FEATURES.items():
            if col in ds.columns:
                actual = ds[col].unique()
                if len(actual) == 1 and abs(float(actual[0]) - expected_val) < 1e-6:
                    pass  # OK
                else:
                    print(f"     ⚠ {set_name}: '{col}' is not constant (live forces "
                          f"{expected_val}, found values: {actual[:5]})")
                    all_ok = False
        print(f"     ✓ {set_name}: live-constant features match monitoring.py")

    # ── Save ──
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "stage1_train.csv"
    test_path  = PROCESSED_DIR / "stage1_test.csv"
    alloc_path = PROCESSED_DIR / "stage1_scenario_allocation.txt"

    output_cols = (
        S1_FEATURES
        + ["class_label", "device_type", "source_dataset", "scenario_id"]
    )
    output_cols = [c for c in output_cols if c in train_df.columns]

    train_df[output_cols].to_csv(train_path, index=False)
    test_df[output_cols].to_csv(test_path, index=False)

    # Save allocation log for the report
    with open(alloc_path, "w") as f:
        f.write("Stage-1 Multi-Source Allocation (v6 — monitoring.py aligned)\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Schema: {len(S1_FEATURES)} features (matches S1_FEATURES in monitoring.py)\n")
        f.write(f"Forced live-constant features: {list(LIVE_CONSTANT_FEATURES.keys())}\n\n")
        for src in sorted(train_df["source_dataset"].unique()):
            f.write(f"\n=== Source: {src} ===\n")
            tr_scens = sorted(train_df[train_df["source_dataset"] == src]
                              ["scenario_id"].unique())
            f.write(f"Train scenarios ({len(tr_scens)}):\n")
            for s in tr_scens[:20]:
                f.write(f"  {s}\n")
            if len(tr_scens) > 20:
                f.write(f"  ... and {len(tr_scens) - 20} more\n")
            te_scens = sorted(test_df[test_df["source_dataset"] == src]
                              ["scenario_id"].unique()) if src in test_df["source_dataset"].values else []
            f.write(f"Test scenarios ({len(te_scens)}):\n")
            for s in te_scens[:20]:
                f.write(f"  {s}\n")

    print(f"\n  ✓ Saved: {train_path}  ({len(train_df):,} rows)")
    print(f"  ✓ Saved: {test_path}   ({len(test_df):,} rows)")
    print(f"  ✓ Saved: {alloc_path}")

    if all_ok:
        print("\n  ✓ ALL CHECKS PASSED — ready for v6 trainer")
    else:
        print("\n  ⚠ Some checks failed — review above before training")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 64 + "╗")
    print("║  PREPROCESSING PIPELINE — Group 07                             ║")
    print("║  v6: 8 sources, 56 features, monitoring.py-aligned             ║")
    print("║       (raw values, no scaling — trainer fits StandardScaler)   ║")
    print("╚" + "═" * 64 + "╝")

    results = {}
    try:
        ok = preprocess_stage1_v6()
        results["Stage-1 (v6)"] = "✓ SUCCESS" if ok else "✗ FAILED"
    except Exception as e:
        results["Stage-1 (v6)"] = f"✗ ERROR: {e}"
        import traceback; traceback.print_exc()

    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        print(f"  {name:30s} : {status}")

    print("\nOutput files:")
    for fname in ["stage1_train.csv", "stage1_test.csv",
                  "stage1_scenario_allocation.txt"]:
        fpath = PROCESSED_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  {fname:40s} ({size_mb:.1f} MB)")
        else:
            print(f"  {fname:40s} (NOT FOUND)")
    print()


if __name__ == "__main__":
    main()
