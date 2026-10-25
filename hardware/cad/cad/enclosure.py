"""
Parametric CAD of the wrist-worn multi-parameter physiological monitor (Stage-2 design).
Project : Multi-Parameter Physiological Signal-Based Wearable Device for Panic & Anxiety Detection
Tool    : CadQuery 2.x (OpenCascade) -> STEP (native, re-importable into SolidWorks/Fusion) + STL
Axes    : X = strap direction, Y = across wrist, Z = up (Z=0 is the skin-contact plane)
Units   : mm
Run     : python enclosure.py   -> writes ./out/*.step, ./out/*.stl, ./out/parts.json
"""
import json, os
import cadquery as cq

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# ---------------- key parameters ----------------
L, W, R = 44.0, 38.0, 6.0          # case footprint and corner radius
FLOOR = 1.6                        # skin-side floor thickness
WALL = 1.8                         # side wall
H_CASE = 13.0                      # bottom case height
COVER_T = 1.5                      # top cover plate
LUG_T, LUG_H, LUG_PIN = 3.0, 7.0, 1.6   # lug thickness (Y), height (Z), spring-bar dia
LUG_X = 25.0                       # pin centre from case centre
STRAP_W = 20.0                     # strap width -> inner lug faces at +-10

DENS = {"pa12": 1.01e-3, "316l": 8.0e-3, "pcb": 1.9e-3}   # g/mm^3

parts = {}   # name -> (solid, colour hex, material key or fixed mass)

# ---------------- bottom case ----------------
body = (cq.Workplane("XY").rect(L, W).extrude(H_CASE).edges("|Z").fillet(R))
pocket = (cq.Workplane("XY").workplane(offset=FLOOR)
          .rect(L - 2 * WALL, W - 2 * WALL).extrude(H_CASE).edges("|Z").fillet(R - WALL))
body = body.cut(pocket)
# optical window (PPG), thermal pad slot, two electrode holes
body = body.cut(cq.Workplane("XY").center(2, 0).circle(5.0).extrude(FLOOR))
body = body.cut(cq.Workplane("XY").center(15, 0).rect(5.5, 5.5).extrude(FLOOR))
for sy in (-9, 9):
    body = body.cut(cq.Workplane("XY").center(-14, sy).circle(3.0).extrude(FLOOR))
# lugs (two per end), pin hole along Y, rounded tip
for sx in (-1, 1):
    for sy in (-1, 1):
        yc = sy * (STRAP_W / 2 + LUG_T / 2)
        lug = (cq.Workplane("XZ").workplane(offset=-(yc + LUG_T / 2))
               .moveTo(sx * (L / 2 - 2.0), 0)
               .lineTo(sx * LUG_X, 0).lineTo(sx * LUG_X, LUG_H)
               .lineTo(sx * (L / 2 - 2.0), LUG_H).close().extrude(LUG_T))
        tip = (cq.Workplane("XZ").workplane(offset=-(yc + LUG_T / 2))
               .center(sx * LUG_X, LUG_H / 2).circle(LUG_H / 2).extrude(LUG_T))
        lug = lug.union(tip)
        lug = lug.cut(cq.Workplane("XZ").workplane(offset=-(yc + LUG_T / 2 + 0.1))
                      .center(sx * LUG_X, LUG_H / 2).circle(LUG_PIN / 2).extrude(LUG_T + 0.2))
        body = body.union(lug)
parts["bottom_case"] = (body, "#52627a", "pa12")

# ---------------- top cover with button + LED windows ----------------
cover = (cq.Workplane("XY").workplane(offset=H_CASE).rect(L, W).extrude(COVER_T)
         .edges("|Z").fillet(R).faces(">Z").edges().fillet(0.6))
cover = cover.cut(cq.Workplane("XY").workplane(offset=H_CASE).center(-12, 0).circle(3.2).extrude(COVER_T))
cover = cover.cut(cq.Workplane("XY").workplane(offset=H_CASE).center(10, 8).circle(1.2).extrude(COVER_T))
parts["top_cover"] = (cover, "#7a8aa3", "pa12")
cap = cq.Workplane("XY").workplane(offset=H_CASE - 0.5).center(-12, 0).circle(2.8).extrude(COVER_T + 1.3)
parts["event_button"] = (cap, "#38bdf8", "pa12")

# ---------------- electronics stack ----------------
Z_PCB = 3.0
pcb = (cq.Workplane("XY").workplane(offset=Z_PCB).rect(L - 2 * WALL - 0.4, W - 2 * WALL - 0.4)
       .extrude(1.6).edges("|Z").fillet(R - WALL - 0.2))
parts["carrier_pcb"] = (pcb, "#15803d", "pcb")
# sensors on skin side of PCB (z 1.6 -> 3.0)
ppg = cq.Workplane("XY").workplane(offset=FLOOR).center(2, 0).rect(8, 8).extrude(Z_PCB - FLOOR)
parts["MAX30102_PPG"] = (ppg, "#7c3aed", 0.15)
tmp = cq.Workplane("XY").workplane(offset=FLOOR).center(15, 0).rect(4, 4).extrude(Z_PCB - FLOOR)
parts["MAX30205_temp"] = (tmp, "#f59e0b", 0.05)
for sy, nm in ((-9, "EDA_electrode_1"), (9, "EDA_electrode_2")):
    e = cq.Workplane("XY").workplane(offset=-0.8).center(-14, sy).circle(2.9).extrude(Z_PCB + 0.8 - 0.0)
    parts[nm] = (e, "#cbd5e1", "316l")
imu = cq.Workplane("XY").workplane(offset=Z_PCB + 1.6).center(-4, -10).rect(4, 4).extrude(0.9)
parts["MPU6050_IMU"] = (imu, "#2563eb", 0.05)
esp = cq.Workplane("XY").workplane(offset=Z_PCB + 1.6).center(-1, 0).rect(25.5, 18).extrude(3.1)
parts["ESP32_WROOM_32E"] = (esp, "#94a3b8", 3.0)
motor = cq.Workplane("XY").workplane(offset=Z_PCB + 1.6).center(13, -9).circle(4.0).extrude(2.7)
parts["haptic_motor"] = (motor, "#ef4444", 0.8)
bat = (cq.Workplane("XY").workplane(offset=Z_PCB + 1.6 + 3.3).center(0, 0).rect(35, 25).extrude(5.0)
       .edges("|Z").fillet(1.0))
parts["LiPo_400mAh"] = (bat, "#60a5fa", 9.0)

# ---------------- strap stubs (visual context only) ----------------
for sx, nm in ((-1, "strap_L"), (1, "strap_R")):
    s = (cq.Workplane("XY").workplane(offset=LUG_H / 2 - 1.3)
         .center(sx * (LUG_X + 10), 0).rect(26, STRAP_W - 0.4).extrude(2.6).edges("|Z").fillet(1.0))
    parts[nm] = (s, "#0f766e", 7.5)

# ---------------- exports + mass table ----------------
summary = {}
for name, (solid, col, mat) in parts.items():
    cq.exporters.export(solid, os.path.join(OUT, f"{name}.stl"), tolerance=0.02, angularTolerance=0.1)
    cq.exporters.export(solid, os.path.join(OUT, f"{name}.step"))
    vol = solid.val().Volume()
    mass = vol * DENS[mat] if isinstance(mat, str) else float(mat)
    summary[name] = {"color": col, "volume_mm3": round(vol, 1), "mass_g": round(mass, 2)}

asm = cq.Assembly()
for name, (solid, col, _) in parts.items():
    asm.add(solid, name=name)
asm.save(os.path.join(OUT, "wearable_assembly.step"))

total = sum(v["mass_g"] for v in summary.values())
summary["_total_mass_g"] = round(total, 1)
summary["_envelope_mm"] = [L, W, H_CASE + COVER_T]
json.dump(summary, open(os.path.join(OUT, "parts.json"), "w"), indent=1)
print(json.dumps({k: v["mass_g"] if isinstance(v, dict) else v for k, v in summary.items()}, indent=1))
