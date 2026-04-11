"""
setup_live_capture.py  —  Run this ONCE before using real live capture
=======================================================================
This script:
    1. Checks if Scapy is installed, installs it if not
    2. Lists all available network interfaces on your machine
    3. Tests that Scapy can actually sniff on your chosen interface
    4. Prints the exact line to paste into live_capture.py

Run as:
    Windows (Admin CMD):   python setup_live_capture.py
    Linux/macOS (sudo):    sudo python setup_live_capture.py

You only need to run this once.
"""

import subprocess
import sys
import os
import platform

OS = platform.system()   # "Windows", "Linux", "Darwin"

# ── ANSI colours ──────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"; E = "\033[0m"

def ok(msg):  print(f"{G}✔ {msg}{E}")
def err(msg): print(f"{R}✘ {msg}{E}")
def info(msg):print(f"{B}ℹ {msg}{E}")
def warn(msg):print(f"{Y}⚠ {msg}{E}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Check / install Scapy
# ══════════════════════════════════════════════════════════════════════════════
def check_scapy():
    print("\n── Step 1: Checking Scapy ──────────────────────────────────────")
    try:
        import scapy
        ok(f"Scapy {scapy.__version__} is already installed.")
        return True
    except ImportError:
        warn("Scapy not found. Installing now...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "scapy", "--quiet"]
            )
            ok("Scapy installed successfully.")
            return True
        except subprocess.CalledProcessError:
            err("pip install failed. Try manually: pip install scapy")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — List interfaces
# ══════════════════════════════════════════════════════════════════════════════
def list_interfaces():
    print("\n── Step 2: Network Interfaces on this machine ──────────────────")

    from scapy.all import get_if_list, get_if_addr

    ifaces = get_if_list()
    valid  = []

    for i, iface in enumerate(ifaces):
        try:
            ip = get_if_addr(iface)
        except Exception:
            ip = "unknown"

        # Skip loopback and obviously virtual/useless ones
        skip = any(x in iface.lower() for x in ["lo", "loopback", "npcap loopback"])
        tag  = f"  [{i}]  {iface:<30}  IP: {ip}"

        if ip == "0.0.0.0" or skip:
            print(f"{Y}{tag}   (skipped — no IP / loopback){E}")
        else:
            print(f"{G}{tag}{E}")
            valid.append((iface, ip))

    if not valid:
        err("No usable interfaces found with a real IP address.")
        print("     Make sure you are connected to a network.")
    else:
        print()
        ok(f"Found {len(valid)} usable interface(s).")

    return valid


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Permissions check
# ══════════════════════════════════════════════════════════════════════════════
def check_permissions():
    print("\n── Step 3: Permission Check ─────────────────────────────────────")

    if OS == "Windows":
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        if is_admin:
            ok("Running as Administrator — good.")
        else:
            err("NOT running as Administrator.")
            warn("Right-click CMD → 'Run as administrator', then run this script again.")
            warn("Scapy on Windows REQUIRES admin rights for raw socket access.")
        return is_admin

    else:  # Linux / macOS
        is_root = os.geteuid() == 0
        if is_root:
            ok("Running as root — good.")
        else:
            err("NOT running as root.")
            warn("On Linux/macOS, either:")
            warn("  sudo python setup_live_capture.py")
            warn("  OR grant your Python binary cap_net_raw capability:")
            warn("  sudo setcap cap_net_raw+eip $(which python3)")
        return is_root


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Quick sniff test
# ══════════════════════════════════════════════════════════════════════════════
def test_sniff(iface: str) -> bool:
    print(f"\n── Step 4: Testing live sniff on '{iface}' (3 seconds) ─────────")
    from scapy.all import sniff, IP, conf
    conf.verb = 0

    captured = []
    try:
        pkts = sniff(iface=iface, timeout=3, store=True, count=20)
        ip_pkts = [p for p in pkts if IP in p]
        captured = ip_pkts
        if ip_pkts:
            ok(f"Captured {len(pkts)} packets ({len(ip_pkts)} IP) in 3 seconds.")
            info("Sample flows:")
            seen = set()
            for p in ip_pkts[:5]:
                key = (p[IP].src, p[IP].dst)
                if key not in seen:
                    seen.add(key)
                    print(f"     {p[IP].src:>16}  →  {p[IP].dst}")
        else:
            warn(f"Captured {len(pkts)} packets but none had an IP layer.")
            warn("Try generating some traffic (open a browser, ping something).")
    except PermissionError:
        err("Permission denied — see Step 3 above.")
        return False
    except OSError as e:
        err(f"Interface error: {e}")
        return False

    return bool(captured)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Print the config line
# ══════════════════════════════════════════════════════════════════════════════
def print_config(iface: str):
    print("\n── Step 5: What to change in your code ─────────────────────────")
    print()
    print("In  app/live_capture.py,  change the LiveCaptureThread constructor:")
    print()
    print(f'{Y}  # BEFORE (demo mode):{E}')
    print(f'  self._thread = LiveCaptureThread(demo_mode=True)')
    print()
    print(f'{G}  # AFTER (real capture on your interface):{E}')
    print(f'  self._thread = LiveCaptureThread(interface="{iface}", demo_mode=False)')
    print()
    print("In  app/monitor_page.py,  find the same line and make the same change.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  BotDetect — Live Capture Setup")
    print("  Group 07 · CPCS499")
    print("=" * 60)

    # Step 1
    if not check_scapy():
        sys.exit(1)

    # Step 2
    valid_ifaces = list_interfaces()
    if not valid_ifaces:
        sys.exit(1)

    # Step 3
    has_perm = check_permissions()

    # Pick interface — ask user if multiple
    if len(valid_ifaces) == 1:
        chosen_iface, chosen_ip = valid_ifaces[0]
        info(f"Using the only available interface: {chosen_iface} ({chosen_ip})")
    else:
        print()
        info("Multiple interfaces found. Which one do you want to use?")
        for i, (iface, ip) in enumerate(valid_ifaces):
            print(f"  [{i}]  {iface}  ({ip})")
        while True:
            try:
                idx = int(input(f"\nEnter number [0-{len(valid_ifaces)-1}]: ").strip())
                if 0 <= idx < len(valid_ifaces):
                    chosen_iface, chosen_ip = valid_ifaces[idx]
                    break
            except (ValueError, KeyboardInterrupt):
                pass
        ok(f"Selected: {chosen_iface} ({chosen_ip})")

    # Step 4 — only test if we have permissions
    test_ok = False
    if has_perm:
        test_ok = test_sniff(chosen_iface)
    else:
        warn("Skipping sniff test — no permissions. Fix Step 3 first.")

    # Step 5 — always show config
    print_config(chosen_iface)

    # Summary
    print("─" * 60)
    if test_ok:
        ok("Everything looks good! Live capture is ready.")
        ok(f"Interface to use:  {chosen_iface}")
    elif has_perm:
        warn("Test captured no packets. Make sure you have network activity.")
        warn("The config line above is still correct — try it and see.")
    else:
        err("Fix admin/root permissions, then run this script again.")

    print()


if __name__ == "__main__":
    main()
