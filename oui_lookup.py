"""
═══════════════════════════════════════════════════════════════════════
 OUI LOOKUP MODULE
 oui_lookup.py
═══════════════════════════════════════════════════════════════════════
 Group 07 | CPCS499 | Path 3: MAC-based fingerprinting

 PURPOSE:
   Deterministic device-type identification via MAC OUI lookup, used as
   a Stage-1 pre-classifier in monitoring.py. Only fires when we have
   HIGH confidence in the vendor → device-type mapping. Otherwise we
   fall back to the ML model.

 WHY A PRE-CLASSIFIER:
   Path 1 (threshold tuning) reduced false-positive CDN routings but
   couldn't fix actual IoT routing on real consumer networks. The reason:
   the ML model never learned device-type-invariant features — it learned
   per-source fingerprints from the 8 academic datasets.

   MAC OUI is a vendor-assigned identifier baked into the hardware.
   When we see a packet from b0:fc:0d:..., we know with certainty that
   the device manufacturer is Amazon, regardless of what the ML model
   thinks. This is more reliable than statistical inference for known
   vendors.

 USAGE:
   from oui_lookup import OUIClassifier

   clf = OUIClassifier()
   verdict, confidence = clf.classify("38:f9:d3:00:00:00")
   # → ("noniot", 0.95)  for Apple

   verdict, confidence = clf.classify("b0:fc:0d:99:00:00")
   # → ("iot", 0.90)  for Amazon (Echo/Alexa)

   verdict, confidence = clf.classify("0a:11:22:33:44:55")
   # → (None, 0.0)  privacy MAC, fall back to ML

 DESIGN PHILOSOPHY:
   Conservative > greedy. We'd rather say "I don't know, use ML" than
   give a confident wrong answer. The mapping below covers ~80% of
   common consumer devices but leaves edge cases for the model.

 REQUIREMENTS:
   pip install manuf
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from typing import Optional, Tuple

try:
    from manuf import manuf as _manuf
    _MANUF_OK = True
except ImportError:
    _MANUF_OK = False


# ═══════════════════════════════════════════════════════════════════════
# VENDOR → DEVICE TYPE MAPPING
# ═══════════════════════════════════════════════════════════════════════
#
# Each entry is: vendor_substring → (device_type, confidence, note)
#
# Matching rules:
#   - Substring match against vendor name (case-insensitive)
#   - First match wins (order matters — most specific FIRST)
#   - Confidence 0.95+ = override ML always
#   - Confidence 0.80-0.94 = override ML if model is uncertain (P<0.70)
#   - Confidence < 0.80 = informational only, ML decides
#
# IMPORTANT: this is intentionally conservative. We only confidently
# classify vendors where the device type is essentially unambiguous.
# Phones, tablets, "smart" peripherals get the lower-confidence treatment.

VENDOR_RULES = [
    # ── Strong NonIoT (workstations, laptops, network infrastructure) ─────
    # These vendors make almost exclusively desktop/laptop/server hardware.
    # Note that "Apple" is high-confidence noniot for traditional Mac/iPhone
    # but if it ever turns out you have an Apple TV or HomePod, those may
    # also be classified as noniot — which is acceptable behavior since
    # Apple's IoT-ish devices behave more like consumer electronics than
    # like Mirai-vulnerable IoT.
    ("Apple",                ("noniot", 0.95, "Mac/iPhone/iPad/Apple TV/HomePod")),
    ("Microsoft",            ("noniot", 0.95, "Windows PC / Surface / Xbox")),
    ("Dell",                 ("noniot", 0.95, "Dell workstation/laptop")),
    ("Hewlett",              ("noniot", 0.90, "HP laptop/desktop (also printers)")),
    ("Lenovo",               ("noniot", 0.95, "Lenovo laptop/desktop")),
    ("ASUSTek",              ("noniot", 0.85, "ASUS laptop/desktop (also some routers)")),
    ("Acer",                 ("noniot", 0.95, "Acer laptop/desktop")),
    ("MSI",                  ("noniot", 0.90, "MSI laptop/desktop")),
    ("LiteOn",               ("noniot", 0.90, "LiteOn laptop OEM")),
    ("Quanta",               ("noniot", 0.85, "Quanta laptop OEM")),
    ("Compal",               ("noniot", 0.85, "Compal laptop OEM")),
    ("VMware",               ("noniot", 0.95, "VMware virtual machine")),
    ("VirtualBox",           ("noniot", 0.95, "VirtualBox virtual machine")),
    ("Parallels",            ("noniot", 0.95, "Parallels virtual machine")),

    # ── Network infrastructure (routers, switches) ────────────────────────
    # Routers occupy a gray area — they're "always on" embedded devices,
    # but they don't behave like IoT in feature space (they aggregate
    # rather than originate IoT-style traffic). Treating them as noniot
    # is more useful for botnet detection because router-originated flows
    # are infrastructure, not target.
    ("Cisco",                ("noniot", 0.85, "Cisco network infrastructure")),
    ("Juniper",              ("noniot", 0.95, "Juniper enterprise network")),
    ("Aruba",                ("noniot", 0.90, "Aruba enterprise WiFi")),
    ("Ubiquiti",             ("noniot", 0.85, "Ubiquiti home/SMB networking")),
    ("Netgear",              ("noniot", 0.85, "Netgear home router")),
    ("D-Link",               ("noniot", 0.80, "D-Link router (some IoT)")),
    ("TP-Link",              ("noniot", 0.75, "TP-Link router (also smart bulbs/cams)")),

    # ── Strong IoT (smart home devices) ───────────────────────────────────
    # These vendors essentially only make IoT devices.
    ("Amazon",               ("iot",    0.85, "Amazon Echo/Alexa (also Kindle = noniot)")),
    ("AmazonTe",             ("iot",    0.85, "Amazon Technologies (Echo line)")),
    ("Sonos",                ("iot",    0.95, "Sonos smart speaker")),
    ("Roku",                 ("iot",    0.95, "Roku streaming device")),
    ("Nest",                 ("iot",    0.95, "Nest thermostat/cam (Google)")),
    ("Ring",                 ("iot",    0.95, "Ring doorbell/cam (Amazon)")),
    ("Wyze",                 ("iot",    0.95, "Wyze cam/sensor")),
    ("Ecobee",               ("iot",    0.95, "Ecobee thermostat")),
    ("Philips",              ("iot",    0.85, "Philips Hue lights (also TVs)")),
    ("LIFX",                 ("iot",    0.95, "LIFX smart bulbs")),
    ("TPMS",                 ("iot",    0.90, "tire pressure / generic sensors")),
    ("EZVIZ",                ("iot",    0.95, "EZVIZ IP camera")),
    ("Hikvision",            ("iot",    0.95, "Hikvision IP camera")),
    ("Dahua",                ("iot",    0.95, "Dahua IP camera")),
    ("Foscam",               ("iot",    0.95, "Foscam IP camera")),
    ("Nintendo",             ("iot",    0.85, "Nintendo Switch (semi-IoT)")),
    ("Sony",                 ("iot",    0.70, "Sony — TVs, PlayStation (mixed)")),
    ("Raspberr",             ("iot",    0.80, "Raspberry Pi (often IoT projects)")),
    ("Espressif",            ("iot",    0.95, "ESP32/ESP8266 chips (IoT only)")),

    # ── Smart TVs (treat as IoT — they have similar attack surface) ───────
    ("Samsung",              ("iot",    0.65, "Samsung — phones (noniot) and TVs/SmartThings (iot)")),
    ("LG",                   ("iot",    0.75, "LG — TVs (iot) and phones (noniot)")),
    ("Vizio",                ("iot",    0.95, "Vizio TV")),
    ("Roku",                 ("iot",    0.95, "Roku")),

    # ── Phone/tablet vendors (mostly noniot but used as IoT for some lines) ─
    ("Google",               ("iot",    0.65, "Google — Pixel (noniot) and Nest/Chromecast (iot)")),
    ("Xiaomi",               ("iot",    0.60, "Xiaomi — phones (noniot) and IoT lots")),
    ("Huawei",               ("noniot", 0.65, "Huawei — phones (noniot) and routers")),
    ("OnePlus",              ("noniot", 0.90, "OnePlus phone")),
    ("Motorola",             ("noniot", 0.85, "Motorola phone")),

    # ── Embedded chipset OEMs (skip — too generic) ────────────────────────
    # These chips appear in EVERYTHING. We can't tell if a Realtek MAC is
    # in a router, a NIC, or an IoT camera. Confidence stays low; ML
    # decides.
    ("Realtek",              (None,     0.0,  "embedded chip — could be anything")),
    ("Broadcom",             (None,     0.0,  "embedded chip — could be anything")),
    ("MediaTek",             (None,     0.0,  "embedded chip — phones/IoT/routers")),
    ("Qualcomm",             (None,     0.0,  "embedded chip — phones/IoT")),
    ("Atheros",              (None,     0.0,  "embedded chip — usually WiFi NIC")),
    ("Espressif",            ("iot",    0.95, "ESP32/ESP8266 — IoT only")),  # exception
    ("Texas Instruments",    (None,     0.0,  "embedded chip — generic")),
]


# ═══════════════════════════════════════════════════════════════════════
# IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════

class OUIClassifier:
    """
    Look up MAC vendor and produce (device_type, confidence) verdict.

    Verdicts:
      ("noniot", c) — high-confidence NonIoT vendor (workstation, etc.)
      ("iot",    c) — high-confidence IoT vendor (smart home device)
      (None,    0)  — unknown / unmapped / privacy MAC → fall back to ML

    Confidence is in [0, 1]. Caller decides how to use it.
    """

    def __init__(self):
        if not _MANUF_OK:
            raise ImportError(
                "manuf package required for MAC-based classification.\n"
                "Install with: pip install manuf"
            )
        self._parser = _manuf.MacParser()

    @staticmethod
    def is_locally_administered(mac: str) -> bool:
        """
        Detect whether a MAC is locally administered (e.g. randomized
        privacy MAC from iOS/Android). The locally-administered bit is
        bit 0x02 of the first octet.
        """
        try:
            first_octet = int(mac.split(":")[0], 16)
            return bool(first_octet & 0x02)
        except (ValueError, IndexError):
            return False

    @staticmethod
    def is_multicast(mac: str) -> bool:
        """
        Detect multicast MAC (bit 0x01 of first octet). Multicast traffic
        wouldn't be a "device" in our sense.
        """
        try:
            first_octet = int(mac.split(":")[0], 16)
            return bool(first_octet & 0x01)
        except (ValueError, IndexError):
            return False

    def classify(self, mac: str) -> Tuple[Optional[str], float]:
        """
        Look up the MAC's vendor and return a device-type verdict.

        Returns:
          (device_type, confidence) where device_type ∈ {"iot", "noniot", None}
        """
        if not mac or mac.count(":") != 5:
            return None, 0.0

        # Privacy MACs and multicast cannot be vendor-classified.
        if self.is_locally_administered(mac):
            return None, 0.0
        if self.is_multicast(mac):
            return None, 0.0

        # Look up vendor in IEEE OUI database
        try:
            vendor = self._parser.get_manuf(mac)
        except Exception:
            return None, 0.0
        if not vendor:
            return None, 0.0

        # Match against our vendor rules
        vendor_lower = vendor.lower()
        for needle, (device_type, confidence, _note) in VENDOR_RULES:
            if needle.lower() in vendor_lower:
                return device_type, confidence

        # No rule matched — vendor is in IEEE database but we don't know
        # what kind of device they make. ML will decide.
        return None, 0.0

    def vendor_name(self, mac: str) -> Optional[str]:
        """Get the raw vendor name for logging/debugging. May return None."""
        try:
            return self._parser.get_manuf(mac)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════
# CLI / SELF-TEST
# ═══════════════════════════════════════════════════════════════════════

def _self_test():
    """Run a sanity check on common vendors. Used by the test entry point."""
    print("OUIClassifier self-test")
    print("=" * 60)

    test_cases = [
        ("38:f9:d3:00:00:00", "Apple Mac/iPhone",            "noniot"),
        ("dc:a6:32:00:00:00", "Raspberry Pi",                "iot"),
        ("b0:fc:0d:99:00:00", "Amazon Echo",                 "iot"),
        ("00:50:56:c0:00:01", "VMware virtual machine",      "noniot"),
        ("0a:11:22:33:44:55", "Locally administered MAC",     None),
        ("01:00:5e:00:00:01", "Multicast MAC",                None),
        ("aa:bb:cc:dd:ee:ff", "Likely random / unmapped",     None),
        ("",                  "Empty MAC",                    None),
    ]

    clf = OUIClassifier()

    for mac, label, expected in test_cases:
        verdict, conf = clf.classify(mac)
        vendor = clf.vendor_name(mac) if mac else None
        match = "✓" if verdict == expected else "✗"
        print(f"  {match} {mac:18s} → vendor={str(vendor):12s} "
              f"verdict={str(verdict):8s} conf={conf:.2f}  ({label})")
    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _self_test()
    elif len(sys.argv) > 1:
        clf = OUIClassifier()
        for mac in sys.argv[1:]:
            verdict, conf = clf.classify(mac)
            vendor = clf.vendor_name(mac)
            print(f"{mac}  vendor={vendor}  verdict={verdict}  conf={conf}")
    else:
        print("Usage:")
        print("  python3 oui_lookup.py --test            # run self-test")
        print("  python3 oui_lookup.py <MAC> [<MAC>...]  # look up specific MACs")
