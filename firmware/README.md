# ESP32 CSI firmware (hardware path)

You need **two ESP32 boards**: one **transmitter (TX)** that floods packets at a
steady rate, and one **receiver (RX)** that extracts CSI from each packet and
prints/streams it to your host. ESP32-S3 is recommended (more RAM, faster UART).

## 1. Install ESP-IDF
```bash
git clone -b v5.1 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh && . ./export.sh
```

## 2. Get esp-csi (Espressif's official CSI examples)
```bash
git clone https://github.com/espressif/esp-csi.git
cd esp-csi/examples/get-started
```
- `csi_send/`  → flash to the **TX** board (sends packets at a fixed rate)
- `csi_recv/`  → flash to the **RX** board (prints `CSI_DATA,...` lines)

```bash
# in each example dir, with the board plugged in:
idf.py set-target esp32s3
idf.py -p /dev/tty.usbserial-XXXX flash monitor
```

## 3. Confirm the RX is printing CSI
You should see lines like:
```
CSI_DATA,STA,aa:bb:cc:dd:ee:ff,-41,11,1,...,[3 -1 4 0 -2 5 ...]
```
The bracketed list is int8 I/Q pairs — exactly what `src/csi_parser.parse_esp_line()`
decodes.

## 4. Capture from the host
```bash
# 60 seconds of labelled "walk" over USB serial
python -m src.live_capture serial --port /dev/tty.usbserial-XXXX \
       --label walk --seconds 60 --out data/raw/walk_01.csv
```
Record several sessions per class, ideally from a few positions/people, then point
training at your collected CSVs (extend `src/datasets.py` with a `load_captured()`
that reads `data/raw/*.csv` — the format already matches the replay stream).

## Tips for clean data
- Keep TX → RX line-of-sight fixed; ~1–3 m apart works well indoors.
- Drive the TX at **~100 packets/s** (set in `csi_send`); verify the printed Hz.
- One environment per model to start — CSI does **not** transfer across rooms.
- Amplitude-only is the robust default; raw ESP32 **phase** needs the linear-fit
  removal in `csi_parser.sanitize_phase()` before it's usable.
