"""Collect CSI from a real ESP32 over serial or UDP.

Two transports, same output (a CSV of amplitude per subcarrier, optionally with
a --label column for supervised recording):

  serial : ESP32 RX prints `CSI_DATA,...` lines over USB  (needs pyserial)
  udp    : ESP32 RX firmware streams CSI frames to this host:port

Examples
--------
  # record 60s of "walk" from a board on /dev/tty.usbserial-0001
  python -m src.live_capture serial --port /dev/tty.usbserial-0001 \
         --label walk --seconds 60 --out data/raw/walk_01.csv

  # just stream UDP frames to a file until Ctrl-C
  python -m src.live_capture udp --port 5566 --out data/raw/session.csv

See firmware/README.md for flashing the boards (one TX, one RX).
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
import numpy as np
import pandas as pd

from . import config, csi_parser

_COLS = [f"sc{i}" for i in range(config.NUM_SUBCARRIERS)]


def _row(csi, label):
    amp = csi_parser.amplitude(csi)
    d = {c: float(v) for c, v in zip(_COLS, amp)}
    if label is not None:
        d = {"label": label, **d}
    return d


def capture_serial(port, baud, seconds, label):
    try:
        import serial
    except ImportError:
        sys.exit("pyserial not installed — `pip install pyserial`")
    ser = serial.Serial(port, baud, timeout=1)
    rows, t0 = [], time.time()
    print(f"   reading {port}@{baud} for {seconds}s …  (Ctrl-C to stop early)")
    try:
        while time.time() - t0 < seconds:
            line = ser.readline().decode("utf-8", "ignore")
            csi = csi_parser.parse_esp_line(line)
            if csi is not None:
                rows.append(_row(csi, label))
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
    return rows


def capture_udp(host, port, seconds, label):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(1.0)
    rows, t0 = [], time.time()
    print(f"   listening udp://{host}:{port} for {seconds}s …  (Ctrl-C to stop)")
    try:
        while time.time() - t0 < seconds:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            csi = csi_parser.parse_esp_line(data.decode("utf-8", "ignore"))
            if csi is not None:
                rows.append(_row(csi, label))
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transport", choices=["serial", "udp"])
    ap.add_argument("--port", default=None, help="serial device or UDP port")
    ap.add_argument("--baud", type=int, default=config.SERIAL_BAUD)
    ap.add_argument("--host", default=config.UDP_HOST)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--label", default=None, help="activity label for supervised capture")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.transport == "serial":
        rows = capture_serial(args.port or config.SERIAL_PORT, args.baud,
                              args.seconds, args.label)
    else:
        rows = capture_udp(args.host, int(args.port or config.UDP_PORT),
                           args.seconds, args.label)

    if not rows:
        sys.exit("No CSI frames captured — check wiring/firmware and the TX packet rate.")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    rate = len(rows) / args.seconds
    print(f"==> wrote {args.out}: {len(df):,} frames (~{rate:.0f} Hz)")


if __name__ == "__main__":
    main()
