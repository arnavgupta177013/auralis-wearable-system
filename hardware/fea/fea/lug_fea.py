"""
Linear-static FEA of the strap lug (critical load path of the wearable case).
Mesh   : gmsh 4.x, 2nd-order tetrahedra (global size refined over 4 levels -> convergence study)
Solver : scikit-fem (P2 tets), small-strain linear elasticity
Model  : lug (3 mm thick, 7 mm tall, 1.6 mm pin hole, R3.5 tip) + 3 mm wall stub, wall end face fixed
Loads  : 50 N pull per lug through the spring-bar, applied as pressure on the loaded half of the pin hole
         case A: pull along strap axis (+X)     case B: pull 30 deg below the axis (towards the wrist, -Z)
Material: PA12 (MJF / SLS) E = 1.7 GPa, nu = 0.40, tensile yield ~ 48 MPa (vendor datasheet range 45-50 MPa)
Output : out/convergence.csv, out/vonmises_caseA.png, out/vonmises_caseB.png, out/summary.json
NOTE   : peak at the fixed face is a boundary singularity -> stress is reported over the lug (x > 23.5 mm)
         i.e. excluding the artificial constraint region; hand-calcs are printed alongside.
"""
import json, os, math, csv
import numpy as np
import gmsh
import skfem
from skfem import MeshTet2, Basis, ElementTetP2, ElementVector, asm, solve, condense, FacetBasis
from skfem.models.elasticity import linear_elasticity, lame_parameters
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)
E, NU, YIELD = 1700.0, 0.40, 48.0   # MPa (N/mm^2)
F = 50.0                            # N
T, H, PIN, X0, XP = 3.0, 7.0, 1.6, 22.0, 25.0   # thickness, height, hole dia, root x, hole centre x
XW = 19.0                           # wall stub start (fixed face)

def build_mesh(h):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("lug"); occ = gmsh.model.occ
    # profile in XZ extruded along Y (thickness T): stub box + box to hole centre + half-disc tip
    box = occ.addBox(XW, 0, 0, XP - XW, T, H)
    disc = occ.addCylinder(XP, 0, H / 2, 0, T, 0, H / 2)
    hole = occ.addCylinder(XP, -0.1, H / 2, 0, T + 0.2, 0, PIN / 2)
    u, _ = occ.fuse([(3, box)], [(3, disc)]); c, _ = occ.cut(u, [(3, hole)])
    occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h / 4)
    # refine around the hole
    f = gmsh.model.mesh.field
    f.add("Ball", 1); f.setNumber(1, "Radius", 2.2); f.setNumber(1, "Thickness", 1.5)
    f.setNumber(1, "VIn", h / 3); f.setNumber(1, "VOut", h); f.setNumber(1, "XCenter", XP); f.setNumber(1, "YCenter", T / 2); f.setNumber(1, "ZCenter", H / 2)
    f.setAsBackgroundMesh(1)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.model.mesh.generate(3)
    path = os.path.join(OUT, f"lug_{h:.2f}.msh"); gmsh.write(path); gmsh.finalize()
    return skfem.Mesh.load(path) if False else MeshTet2.load(path)

def solve_case(mesh, angle_deg):
    basis = Basis(mesh, ElementVector(ElementTetP2()))
    lam, mu = lame_parameters(E, NU)
    K = asm(linear_elasticity(lam, mu), basis)
    # fixed face x = XW
    fixed = basis.get_dofs(lambda x: np.isclose(x[0], XW, atol=1e-6))
    # loaded facets: hole surface, half facing the pull direction d (pin pushes the material on that side)
    d = np.array([math.cos(math.radians(angle_deg)), 0.0, -math.sin(math.radians(angle_deg))])
    on_hole = lambda x: np.abs(np.hypot(x[0] - XP, x[2] - H / 2) - PIN / 2) < 0.05
    facets = mesh.facets_satisfying(on_hole)
    fb = FacetBasis(mesh, basis.elem, facets=facets)
    from skfem import LinearForm
    mask_of = lambda w: ((w.x[0] - XP) * d[0] + (w.x[2] - H / 2) * d[2] > 0).astype(float)
    area = asm(LinearForm(lambda v, w: mask_of(w) * v[1] * 0 + mask_of(w) * 1.0 * (v[0] * 0 + 1)), fb) if False else None
    # loaded area from a scalar basis
    from skfem import ElementTetP2 as _P2
    sb = Basis(mesh, _P2()); sfb = FacetBasis(mesh, _P2(), facets=facets)
    area = asm(LinearForm(lambda v, w: mask_of(w) * v), sfb).sum()
    tau = F / area                                   # uniform traction magnitude (N/mm^2) acting along d
    b = asm(LinearForm(lambda v, w: tau * mask_of(w) * (v[0] * d[0] + v[2] * d[2])), fb)
    u = solve(*condense(K, b, D=fixed))
    # von Mises at element quadrature points -> nodal average via P1 projection (use P0 per element mean)
    ux = u[basis.nodal_dofs]  # (3, nnodes) for P2 nodes only
    # stress via gradient at quadrature points, averaged per element
    from skfem import Basis as B0, ElementTetP0
    dbasis = Basis(mesh, ElementTetP0(), quadrature=basis.quadrature)
    vm = np.zeros(mesh.t.shape[1])
    grad = basis.interpolate(u).grad   # (3,3,nel,nq)
    eps = 0.5 * (grad + np.swapaxes(grad, 0, 1))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    sig = 2 * mu * eps
    for i in range(3): sig[i, i] += lam * tr
    s = np.sqrt(0.5 * ((sig[0,0]-sig[1,1])**2 + (sig[1,1]-sig[2,2])**2 + (sig[2,2]-sig[0,0])**2) + 3 * (sig[0,1]**2 + sig[1,2]**2 + sig[0,2]**2))
    vm_el = s.max(axis=1)   # peak over quad points per element
    centroid_x = mesh.p[0][mesh.t].mean(axis=0)
    mask = centroid_x > 23.5
    return dict(vm_el=vm_el, peak_all=float(vm_el.max()), peak_lug=float(vm_el[mask].max()), ndof=int(basis.N), nel=int(mesh.t.shape[1]),
                umax=float(np.abs(u[basis.nodal_dofs]).max()), mesh=mesh, mask=mask)

levels = [1.6, 1.1, 0.8]
rows = []
for h in levels:
    m = build_mesh(h)
    ra = solve_case(m, 0.0); rb = solve_case(m, 30.0)
    rows.append((h, ra["nel"], ra["ndof"], ra["peak_lug"], rb["peak_lug"], ra["umax"]))
    print(f"h={h:.2f} elems={ra['nel']} dof={ra['ndof']}  vM(A)={ra['peak_lug']:.2f}  vM(B)={rb['peak_lug']:.2f} MPa  umax={ra['umax']:.4f} mm", flush=True)
    last = (m, ra, rb)

with open(os.path.join(OUT, "convergence.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["h_mm", "elements", "dof", "vonMises_caseA_MPa", "vonMises_caseB_MPa", "umax_caseA_mm"]); w.writerows(rows)

# convergence plot
fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
ax.plot([r[2] for r in rows], [r[3] for r in rows], "o-", color="#0284c7", label="Case A: axial 50 N")
ax.plot([r[2] for r in rows], [r[4] for r in rows], "s-", color="#16a34a", label="Case B: 30° to wrist")
ax.axhline(YIELD, color="#dc2626", ls="--", lw=1); ax.text(rows[0][2], YIELD * 0.94, "PA12 yield ≈ 48 MPa", color="#dc2626", fontsize=7)
ax.set_xlabel("Degrees of freedom"); ax.set_ylabel("Peak von Mises in lug (MPa)"); ax.set_xscale("log")
ax.set_ylim(0, 55); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="center right"); ax.set_title("Mesh convergence (P2 tetrahedra)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "convergence.png")); plt.close(fig)

# contour plots on the finest mesh
m, ra, rb = last
for tag, r in (("caseA", ra), ("caseB", rb)):
    p = m.p; t = m.t[:4]   # corner nodes only for plotting
    cen = p[:, t].mean(axis=1)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=220)
    sel = np.argsort(cen[1])            # back-to-front along Y so the front face shows
    vmax = max(ra["peak_lug"], rb["peak_lug"])
    sc = ax.scatter(cen[0][sel], cen[2][sel], c=np.minimum(r["vm_el"][sel], vmax), cmap="turbo", s=9, marker="s", vmin=0, vmax=vmax, linewidths=0)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(XP + PIN / 2 * np.cos(th), H / 2 + PIN / 2 * np.sin(th), color="k", lw=0.8)
    ax.plot([XW, XP, XP + H / 2 * 0 ], [0, 0, 0], color="k", lw=0.6)
    ax.axvline(XW, color="#dc2626", lw=1.2); ax.text(XW + 0.1, H + 0.35, "fixed face", color="#dc2626", fontsize=6)
    ax.annotate("", xy=(XP + 2.6, H / 2), xytext=(XP + 0.9, H / 2), arrowprops=dict(arrowstyle="-|>", color="k", lw=1)) if tag == "caseA" else None
    ax.text(XP + 2.7, H / 2 + 0.3, "50 N", fontsize=6.5) if tag == "caseA" else ax.text(XP + 1.3, H / 2 - 1.6, "50 N at 30°", fontsize=6.5)
    ax.set_aspect("equal"); ax.set_xlim(XW - 0.5, XP + 5.5); ax.set_ylim(-0.8, H + 1.2); ax.set_axis_off()
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01); cb.set_label("von Mises (MPa)", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_title(f"{'Axial pull 50 N' if tag=='caseA' else '50 N at 30° to wrist'}: peak {r['peak_lug']:.1f} MPa (front view, X-Z)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"vonmises_{tag}.png")); plt.close(fig)

# hand calculations
bearing = F / (PIN * T)
net_tension = F / ((H - PIN) * T)
ligament = F / (2 * ((H / 2) - PIN / 2) * T)   # tear-out across the two ligaments above/below hole (conservative: axial to tip)
summary = {
    "mesh_levels": [dict(h_mm=r[0], elements=r[1], dof=r[2], vM_A=round(r[3], 2), vM_B=round(r[4], 2)) for r in rows],
    "final_peak_vonMises_caseA_MPa": round(rows[-1][3], 1), "final_peak_vonMises_caseB_MPa": round(rows[-1][4], 1),
    "FoS_yield_caseA": round(YIELD / rows[-1][3], 2), "FoS_yield_caseB": round(YIELD / rows[-1][4], 2),
    "max_deflection_mm": round(rows[-1][5], 4),
    "hand_calc": {"pin_bearing_MPa": round(bearing, 2), "net_section_tension_MPa": round(net_tension, 2), "tip_ligament_shear_tearout_MPa": round(F / (2 * (XP + H / 2 - XP - PIN / 2) * T), 2)},
    "mesh_stats_final": {"elements": rows[-1][1], "dof": rows[-1][2]},
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
