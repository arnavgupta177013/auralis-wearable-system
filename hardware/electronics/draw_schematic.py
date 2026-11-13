"""Connection / power-distribution schematic for the wearable (design draft, redraw in KiCad for fabrication).
Run: python draw_schematic.py -> schematic_connections.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

NAVY, BLUE, GREEN, AMBER, RED, GREY, VIOLET = "#1e3a8a", "#0284c7", "#16a34a", "#d97706", "#dc2626", "#64748b", "#7c3aed"
fig, ax = plt.subplots(figsize=(11.6, 6.6), dpi=200)
ax.set_xlim(0, 124); ax.set_ylim(-1.2, 70); ax.axis("off")

def box(x, y, w, h, title, sub=None, ec=NAVY, fc="#f8fafc", tc=NAVY, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=fs, fontweight="bold", color=tc, family="DejaVu Sans")
    if sub:
        ax.text(x + w / 2, y + h - 5.2, sub, ha="center", va="top", fontsize=7.2, color=GREY, linespacing=1.3)

def wire(pts, c=NAVY, lw=1.6, ls="-", arrow=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=c, lw=lw, ls=ls, solid_capstyle="round")
    if arrow:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=c, lw=lw))

# ---------------- POWER CHAIN (left) ----------------
ax.text(1, 68.2, "POWER DISTRIBUTION", fontsize=9, fontweight="bold", color=BLUE)
box(1, 56, 22, 9, "USB-C 5 V", "charge input only")
box(1, 41, 22, 11, "TP4056 + DW01A/8205", "1-cell charger, over-charge /\nover-discharge / short protect", ec=AMBER)
box(1, 28, 22, 9, "Li-Po 3.7 V 400 mAh", "IEC 62133-2 cell, PCM protected", ec=AMBER)
box(1, 14, 22, 10, "AP2112K-3.3 LDO", "600 mA, 3V3 rail, 55 µA Iq", ec=GREEN)
box(1, 1.5, 22, 8, "VBAT sense", "100k / 100k divider → GPIO35", ec=GREY)
for y0, y1 in ((56, 52), (41, 37), (28, 24)):
    wire([(12, y0), (12, y1)], c=AMBER, arrow=True)
wire([(12, 14), (12, 9.5)], c=GREY, arrow=True)
ax.text(24, 26, "SW1\nslide\nswitch", fontsize=6.5, color=GREY, va="center")

# ---------------- ESP32 (centre) ----------------
box(40, 3, 30, 61, "ESP32-WROOM-32E", "Dual-core 240 MHz · BLE · Wi-Fi\nEdge-AI (TFLite-Micro)", ec=NAVY, fc="#eff6ff", fs=9.8)
pins = [("3V3", 54.5), ("GPIO21  SDA", 52.5), ("GPIO22  SCL", 48.5), ("GPIO4   INT_PPG", 43.5), ("GPIO19  INT_IMU", 39.5),
        ("GPIO34  (free)", 33.5), ("GPIO35  VBAT_SENS", 27.5), ("GPIO25  HAPTIC", 18.5), ("GPIO27  EVENT_BTN", 10.5), ("GPIO26  STATUS_LED", 7), ("GND", 4.3)]

for name, y in pins:
    ax.text(41.2, y, name, fontsize=7.4, color=NAVY, va="center", family="DejaVu Sans Mono")
    ax.plot([70, 74], [y, y], color=NAVY, lw=1.6)
wire([(12, 1.5), (12, 1.5)], c=GREY)
wire([(23, 5.5), (36, 5.5), (36, 27.5), (40, 27.5)], c=GREY, arrow=True)
wire([(23, 19), (32, 19), (32, 54.5), (40, 54.5)], c=GREEN, arrow=True)
ax.text(25, 56, "3V3 rail", fontsize=7, color=GREEN)

# ---------------- SENSORS (right) ----------------
ax.text(76, 68.4, "SENSING & I/O", fontsize=9, fontweight="bold", color=BLUE)
for y, t, s2, c in [(57.5, "MAX30102  (0x57)", "INT → GPIO4  ·  bench: MAX30100 (ID 0x11)", VIOLET),
                    (49.0, "ADS1115  (0x48)", "EDA AIN0 · NTC AIN1 · 3V3 AIN3", AMBER),
                    (40.5, "MPU6050  (0x68)", "INT → GPIO19  ·  bench: MPU6500 (ID 0x70)", BLUE)]:
    box(88, y, 34, 7.0, t, s2, ec=c, fs=8.2)
    wire([(84, y + 3.5), (88, y + 3.5)], c=NAVY, lw=2.2)
wire([(84, 44), (84, 61)], c=NAVY, lw=2.6)
wire([(74, 52.5), (79, 52.5), (79, 50.5), (84, 50.5)], c=NAVY, lw=1.4)
wire([(74, 48.5), (79, 48.5), (79, 50.5)], c=NAVY, lw=1.4)
ax.text(79.3, 55.0, "I²C 400 kHz", fontsize=6.3, color=NAVY, ha="center", va="bottom")
ax.text(79.3, 62.6, "2.2 kΩ pull-ups\nSDA/SCL → 3V3", fontsize=5.8, color=RED, ha="center", va="bottom")
ax.text(75, 43.5, "INT →", fontsize=6.3, color=RED, va="center"); ax.text(75, 39.5, "INT →", fontsize=6.3, color=RED, va="center")

box(88, 28.3, 34, 10.5, "EDA divider → ADS1115 AIN0", "3V3 → 330 kΩ → electrodes → GND\nµS ratiometric · ≤ 10 µA DC", ec=GREEN, fs=8.2)
wire([(122, 33.5), (123.6, 33.5), (123.6, 52.5), (122.3, 52.5)], c=GREEN, lw=1.6)
ax.annotate("", xy=(122.2, 52.5), xytext=(123.6, 52.5), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
box(88, 18.9, 34, 7.0, "Electrodes: 2× 316L SS Ø6", "skin contact · battery-only · no mains link", ec=GREEN, fs=8.2)
wire([(105, 28.3), (105, 25.9)], c=GREEN, lw=1.6, arrow=True)
box(88, 8.4, 34, 9.8, "Haptic driver", "GPIO25 → 1 kΩ → AO3400 MOSFET\ncoin ERM Ø8 + SS14 flyback diode", ec=RED, fs=8.2)
wire([(74, 18.5), (80, 18.5), (80, 12.3), (88, 12.3)], c=RED, lw=1.5, arrow=True)
box(88, 0.2, 34, 7.4, "Event button + status LED", "button = shared time-marker for validation", ec=GREY, fs=8.2)
wire([(74, 10.5), (78, 10.5), (78, 5.5), (88, 4.9)], c=GREY, lw=1.3, arrow=True)
wire([(74, 7), (76, 7), (76, 2.4), (88, 2.4)], c=GREY, lw=1.1)

fig.savefig("schematic_connections.png", bbox_inches="tight", facecolor="white")
print("ok")
