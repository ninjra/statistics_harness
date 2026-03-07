from __future__ import annotations

from typing import Any, Callable

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import BudgetTimer, stable_id
from statistic_harness.core.stat_plugins.ideaspace import (
    ENERGY_MAXIMIZE_KEYS,
    ENERGY_MINIMIZE_KEYS,
    _default_energy_weights,
    _energy_gap_for_metrics,
    _energy_terms,
)
from statistic_harness.core.stat_plugins.references import default_references_for_plugin
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import read_json, write_json


def _basic_metrics(df: pd.DataFrame, sample_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows_seen": int(sample_meta.get("rows_seen", len(df))),
        "rows_used": int(sample_meta.get("rows_used", len(df))),
        "cols_used": int(len(df.columns)),
        "sampled": bool(sample_meta.get("sampled", False)),
    }


def _na_result(
    plugin_id: str,
    summary: str,
    gating_reason: str,
    df: pd.DataFrame,
    sample_meta: dict[str, Any],
) -> PluginResult:
    return PluginResult(
        status="na",
        summary=summary,
        metrics=_basic_metrics(df, sample_meta),
        findings=[],
        artifacts=[],
        error=None,
        references=default_references_for_plugin(plugin_id),
        debug={"gating_reason": gating_reason},
    )


def _artifact(ctx: Any, plugin_id: str, name: str, payload: Any) -> PluginArtifact:
    artifact_dir = ctx.artifacts_dir(plugin_id)
    path = artifact_dir / name
    write_json(path, payload)
    return PluginArtifact(path=str(path.relative_to(ctx.run_dir)), type="json", description=name)


def _artifact_html(ctx: Any, plugin_id: str, name: str, html: str) -> PluginArtifact:
    artifact_dir = ctx.artifacts_dir(plugin_id)
    path = artifact_dir / name
    path.write_text(html, encoding="utf-8")
    return PluginArtifact(path=str(path.relative_to(ctx.run_dir)), type="html", description=name)


def _round6(v: float) -> float:
    return round(float(v), 6)


def _load_energy_state_vector(ctx: Any) -> dict[str, Any] | None:
    p = ctx.run_dir / "artifacts" / "analysis_ideaspace_energy_ebm_v1" / "energy_state_vector.json"
    if not p.exists():
        return None
    return read_json(p)


def _load_route_plan(ctx: Any) -> dict[str, Any] | None:
    p = ctx.run_dir / "artifacts" / "analysis_ebm_action_verifier_v1" / "route_plan.json"
    if not p.exists():
        return None
    data = read_json(p)
    if not isinstance(data, dict):
        return None
    if data.get("decision") != "modeled":
        return None
    return data


def _load_verified_actions(ctx: Any) -> list[dict[str, Any]] | None:
    p = ctx.run_dir / "artifacts" / "analysis_ebm_action_verifier_v1" / "verified_actions.json"
    if not p.exists():
        return None
    data = read_json(p)
    if isinstance(data, dict):
        return data.get("actions") or data.get("verified_actions") or []
    if isinstance(data, list):
        return data
    return None


def _extract_entities(esv: dict[str, Any]) -> list[dict[str, Any]]:
    entities = esv.get("entities") or []
    return sorted(
        [e for e in entities if isinstance(e, dict) and isinstance(e.get("entity_key"), str)],
        key=lambda e: (-float(e.get("energy_total", 0.0)), str(e.get("entity_key", ""))),
    )


def _energy_keys_from_esv(esv: dict[str, Any]) -> list[str]:
    keys = esv.get("energy_keys")
    if isinstance(keys, list) and keys:
        return [str(k) for k in keys]
    all_keys = list(ENERGY_MINIMIZE_KEYS) + list(ENERGY_MAXIMIZE_KEYS)
    return all_keys


def _build_feature_matrix(entities: list[dict[str, Any]], energy_keys: list[str]) -> tuple[np.ndarray, list[int]]:
    rows = []
    valid_indices = []
    for i, e in enumerate(entities):
        obs = e.get("observed") or e
        row = []
        valid = True
        for k in energy_keys:
            v = obs.get(k)
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                valid = False
                break
            row.append(float(v))
        if valid:
            rows.append(row)
            valid_indices.append(i)
    if not rows:
        return np.empty((0, len(energy_keys))), []
    return np.array(rows, dtype=np.float64), valid_indices


def _build_ideal_matrix(entities: list[dict[str, Any]], energy_keys: list[str], valid_indices: list[int]) -> np.ndarray:
    rows = []
    for i in valid_indices:
        ideal = entities[i].get("ideal") or {}
        row = []
        for k in energy_keys:
            v = ideal.get(k)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                row.append(float(v))
            else:
                obs = (entities[i].get("observed") or entities[i]).get(k, 0.0)
                row.append(float(obs) if isinstance(obs, (int, float)) else 0.0)
        rows.append(row)
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# HTML generators
# ---------------------------------------------------------------------------

def _landscape_html(
    grid_x: list[float],
    grid_y: list[float],
    grid_z: list[list[float]],
    entity_points: list[dict[str, Any]],
    ideal_points: list[dict[str, Any]],
    route_path: list[dict[str, Any]],
    pca_variance: list[float],
) -> str:
    data_js = json.dumps({
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
        "entities": entity_points,
        "ideals": ideal_points,
        "route": route_path,
        "variance": pca_variance,
    })
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Kona 3D Energy Landscape</title>
<style>
body{{margin:0;background:#1a1a2e;color:#e0e0e0;font-family:monospace;display:flex;flex-direction:column;align-items:center}}
canvas{{border:1px solid #333;margin:20px}}
.legend{{display:flex;gap:20px;padding:10px}}
.legend span{{display:flex;align-items:center;gap:5px}}
.dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
</style></head><body>
<h2>Kona 3D Energy Landscape</h2>
<div class="legend">
<span><span class="dot" style="background:#22c55e"></span>Low Energy</span>
<span><span class="dot" style="background:#ef4444"></span>High Energy</span>
<span><span class="dot" style="background:#3b82f6"></span>Ideal</span>
<span><span class="dot" style="background:#facc15"></span>Route</span>
</div>
<canvas id="c" width="800" height="600"></canvas>
<script>
var D={data_js};
var cv=document.getElementById("c"),cx=cv.getContext("2d");
var W=cv.width,H=cv.height,ox=W/2,oy=H*0.75;
var sc=1.8;
function iso(x,y,z){{return[ox+(x-y)*sc*30, oy-(x+y)*sc*15-z*sc*40]}}
function lerp(a,b,t){{return a+(b-a)*t}}
function eColor(t){{var r=Math.round(lerp(34,239,t)),g=Math.round(lerp(197,68,t)),b=Math.round(lerp(94,68,t));return"rgb("+r+","+g+","+b+")"}}
cx.fillStyle="#1a1a2e";cx.fillRect(0,0,W,H);
var gx=D.grid_x,gy=D.grid_y,gz=D.grid_z;
if(gx.length>0&&gy.length>0){{
var zmin=1e9,zmax=-1e9;
for(var i=0;i<gz.length;i++)for(var j=0;j<gz[i].length;j++){{if(gz[i][j]<zmin)zmin=gz[i][j];if(gz[i][j]>zmax)zmax=gz[i][j]}}
var zr=Math.max(zmax-zmin,1e-9);
for(var i=0;i<gz.length-1;i++)for(var j=0;j<gz[i].length-1;j++){{
var t=(gz[i][j]-zmin)/zr;
cx.strokeStyle=eColor(t);cx.globalAlpha=0.3;
var p1=iso(gx[j],gy[i],gz[i][j]),p2=iso(gx[j+1],gy[i],gz[i][j+1]);
cx.beginPath();cx.moveTo(p1[0],p1[1]);cx.lineTo(p2[0],p2[1]);cx.stroke();
var p3=iso(gx[j],gy[i+1],gz[i+1][j]);
cx.beginPath();cx.moveTo(p1[0],p1[1]);cx.lineTo(p3[0],p3[1]);cx.stroke();
}}
cx.globalAlpha=1.0;
}}
for(var i=0;i<D.entities.length;i++){{var e=D.entities[i];var zr2=Math.max(zmax-zmin,1e-9);var t2=(e.energy_total-zmin)/zr2;
var p=iso(e.pc1,e.pc2,e.energy_total);cx.fillStyle=eColor(Math.min(1,Math.max(0,t2)));cx.beginPath();cx.arc(p[0],p[1],5,0,Math.PI*2);cx.fill();
cx.fillStyle="#fff";cx.font="9px monospace";cx.fillText(e.label,p[0]+7,p[1]+3);}}
for(var i=0;i<D.ideals.length;i++){{var d=D.ideals[i];var p=iso(d.pc1,d.pc2,d.energy_total);
cx.fillStyle="#3b82f6";cx.beginPath();cx.moveTo(p[0],p[1]-8);for(var s=1;s<10;s++){{var a=s*Math.PI/5-Math.PI/2,r=s%2?3:8;cx.lineTo(p[0]+r*Math.cos(a),p[1]+r*Math.sin(a))}}cx.closePath();cx.fill();}}
if(D.route.length>1){{cx.strokeStyle="#facc15";cx.lineWidth=2.5;cx.beginPath();
for(var i=0;i<D.route.length;i++){{var r=D.route[i];var p=iso(r.pc1,r.pc2,r.energy);if(i===0)cx.moveTo(p[0],p[1]);else cx.lineTo(p[0],p[1])}}cx.stroke();
for(var i=0;i<D.route.length;i++){{var r=D.route[i];var p=iso(r.pc1,r.pc2,r.energy);cx.fillStyle="#facc15";cx.beginPath();cx.arc(p[0],p[1],4,0,Math.PI*2);cx.fill();
cx.fillStyle="#fff";cx.font="9px monospace";cx.fillText(r.step_index+"",p[0]+6,p[1]-4)}}}}
cx.fillStyle="#888";cx.font="11px monospace";
cx.fillText("PC1 ("+Math.round(D.variance[0]*100)+"% var)",W-160,H-10);
cx.fillText("PC2 ("+Math.round(D.variance[1]*100)+"% var)",10,H-10);
cx.fillText("Energy (Z)",10,20);
</script></body></html>"""


def _kohonen_html(
    umatrix: list[list[float]],
    energy_grid: list[list[float]],
    entity_bmus: list[dict[str, Any]],
    ideal_bmus: list[dict[str, Any]],
    route_bmus: list[dict[str, Any]],
    som_shape: list[int],
) -> str:
    data_js = json.dumps({
        "umatrix": umatrix,
        "energy_grid": energy_grid,
        "entities": entity_bmus,
        "ideals": ideal_bmus,
        "route": route_bmus,
        "shape": som_shape,
    })
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Kohonen Energy Map</title>
<style>
body{{margin:0;background:#1a1a2e;color:#e0e0e0;font-family:monospace;display:flex;flex-direction:column;align-items:center}}
canvas{{border:1px solid #333;margin:20px}}
.legend{{display:flex;gap:20px;padding:10px}}
.legend span{{display:flex;align-items:center;gap:5px}}
.dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
#tooltip{{position:absolute;background:#222;border:1px solid #555;padding:4px 8px;display:none;pointer-events:none;font-size:11px}}
</style></head><body>
<h2>Kohonen Self-Organizing Map — Energy Space</h2>
<div class="legend">
<span><span class="dot" style="background:#2d2d44"></span>Low U-dist (basin)</span>
<span><span class="dot" style="background:#e0e0e0"></span>High U-dist (boundary)</span>
<span><span class="dot" style="background:#3b82f6"></span>Ideal</span>
<span><span class="dot" style="background:#facc15"></span>Route</span>
</div>
<canvas id="c" width="700" height="700"></canvas>
<div id="tooltip"></div>
<script>
var D={data_js};
var cv=document.getElementById("c"),cx=cv.getContext("2d"),tt=document.getElementById("tooltip");
var S=D.shape,cs=Math.floor(Math.min(650/S[0],650/S[1])),pad=25;
cv.width=S[1]*cs+pad*2;cv.height=S[0]*cs+pad*2;
var umin=1e9,umax=-1e9;
for(var i=0;i<D.umatrix.length;i++)for(var j=0;j<D.umatrix[i].length;j++){{if(D.umatrix[i][j]<umin)umin=D.umatrix[i][j];if(D.umatrix[i][j]>umax)umax=D.umatrix[i][j]}}
var ur=Math.max(umax-umin,1e-9);
for(var i=0;i<S[0];i++)for(var j=0;j<S[1];j++){{
var t=(D.umatrix[i][j]-umin)/ur;
var g=Math.round(45+t*180);
cx.fillStyle="rgb("+g+","+g+","+(g+20)+")";
cx.fillRect(pad+j*cs,pad+i*cs,cs-1,cs-1);
}}
var emin=1e9,emax=-1e9;
for(var i=0;i<D.entities.length;i++){{var e=D.entities[i].energy_total;if(e<emin)emin=e;if(e>emax)emax=e}}
var er=Math.max(emax-emin,1e-9);
for(var i=0;i<D.entities.length;i++){{var e=D.entities[i];
var px=pad+e.bmu_y*cs+cs/2,py=pad+e.bmu_x*cs+cs/2;
var t2=(e.energy_total-emin)/er;
var r=Math.round(34+t2*205),g2=Math.round(197-t2*129),b=Math.round(94-t2*26);
cx.fillStyle="rgb("+r+","+g2+","+b+")";cx.beginPath();cx.arc(px,py,Math.max(4,cs/4),0,Math.PI*2);cx.fill();
cx.strokeStyle="#fff";cx.lineWidth=1;cx.stroke();
}}
for(var i=0;i<D.ideals.length;i++){{var d=D.ideals[i];
var px=pad+d.bmu_y*cs+cs/2,py=pad+d.bmu_x*cs+cs/2;
cx.fillStyle="#3b82f6";cx.beginPath();
cx.moveTo(px,py-cs/3);for(var s=1;s<10;s++){{var a=s*Math.PI/5-Math.PI/2,rv=s%2?cs/6:cs/3;cx.lineTo(px+rv*Math.cos(a),py+rv*Math.sin(a))}}cx.closePath();cx.fill();
}}
if(D.route.length>1){{cx.strokeStyle="#facc15";cx.lineWidth=2;cx.beginPath();
for(var i=0;i<D.route.length;i++){{var r=D.route[i];var px=pad+r.bmu_y*cs+cs/2,py=pad+r.bmu_x*cs+cs/2;
if(i===0)cx.moveTo(px,py);else cx.lineTo(px,py)}}cx.stroke();}}
cv.addEventListener("mousemove",function(ev){{
var rect=cv.getBoundingClientRect(),mx=ev.clientX-rect.left,my=ev.clientY-rect.top;
var found=null;
for(var i=0;i<D.entities.length;i++){{var e=D.entities[i];
var px=pad+e.bmu_y*cs+cs/2,py=pad+e.bmu_x*cs+cs/2;
if(Math.abs(mx-px)<cs/2&&Math.abs(my-py)<cs/2){{found=e;break}}
}}
if(found){{tt.style.display="block";tt.style.left=(ev.clientX+10)+"px";tt.style.top=(ev.clientY+10)+"px";
tt.textContent=found.entity_key+" | energy: "+found.energy_total.toFixed(4)+" | qerr: "+found.quantization_error.toFixed(4)}}
else{{tt.style.display="none"}}
}});
</script></body></html>"""


# ---------------------------------------------------------------------------
# Handler 1: 3D Energy Landscape
# ---------------------------------------------------------------------------

def _kona_3d_landscape_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if not config.get("enabled", False):
        return _na_result(plugin_id, "3D landscape disabled", "disabled_by_config", df, sample_meta)

    esv = _load_energy_state_vector(ctx)
    if esv is None:
        return _na_result(plugin_id, "Missing energy state vector", "missing_energy_artifact", df, sample_meta)

    entities = _extract_entities(esv)
    energy_keys = _energy_keys_from_esv(esv)
    min_entities = int(config.get("min_entities_for_surface", 4))

    feature_matrix, valid_indices = _build_feature_matrix(entities, energy_keys)
    if len(valid_indices) < min_entities:
        return _na_result(
            plugin_id,
            f"Insufficient entities ({len(valid_indices)} < {min_entities})",
            "insufficient_entities",
            df,
            sample_meta,
        )

    if timer.exceeded():
        return _na_result(plugin_id, "Budget exceeded before PCA", "budget_exceeded", df, sample_meta)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    run_seed = int(ctx.run_seed)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    n_components = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=run_seed)
    X_2d = pca.fit_transform(X_scaled)

    energies = np.array([float(entities[i].get("energy_total", 0.0)) for i in valid_indices])

    grid_resolution = int(config.get("grid_resolution", 50))
    margin_pct = float(config.get("grid_margin_pct", 0.15))

    if X_2d.shape[1] >= 2:
        pc1 = X_2d[:, 0]
        pc2 = X_2d[:, 1]
    else:
        pc1 = X_2d[:, 0]
        pc2 = np.zeros_like(pc1)

    margin_x = (pc1.max() - pc1.min()) * margin_pct if pc1.max() > pc1.min() else 0.5
    margin_y = (pc2.max() - pc2.min()) * margin_pct if pc2.max() > pc2.min() else 0.5

    grid_x = np.linspace(float(pc1.min() - margin_x), float(pc1.max() + margin_x), grid_resolution)
    grid_y = np.linspace(float(pc2.min() - margin_y), float(pc2.max() + margin_y), grid_resolution)

    from scipy.interpolate import griddata

    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    points_2d = np.column_stack([pc1, pc2])
    grid_z = griddata(points_2d, energies, (grid_xx, grid_yy), method="cubic", fill_value=float(np.nanmax(energies)))
    grid_z = np.nan_to_num(grid_z, nan=float(np.nanmax(energies)))

    entity_points = []
    for idx_pos, orig_idx in enumerate(valid_indices):
        entity_points.append({
            "entity_key": str(entities[orig_idx].get("entity_key", "")),
            "pc1": _round6(float(pc1[idx_pos])),
            "pc2": _round6(float(pc2[idx_pos])),
            "energy_total": _round6(float(energies[idx_pos])),
            "label": str(entities[orig_idx].get("entity_key", "")),
        })

    ideal_matrix = _build_ideal_matrix(entities, energy_keys, valid_indices)
    ideal_scaled = scaler.transform(ideal_matrix)
    ideal_2d = pca.transform(ideal_scaled)

    ideal_points = []
    for idx_pos, orig_idx in enumerate(valid_indices):
        ideal_e = entities[orig_idx].get("ideal") or {}
        obs_e = entities[orig_idx].get("observed") or entities[orig_idx]
        weights = _default_energy_weights(esv.get("weights"))
        e_ideal = _energy_gap_for_metrics(ideal_e, ideal_e, weights)
        ideal_points.append({
            "pc1": _round6(float(ideal_2d[idx_pos, 0]) if ideal_2d.shape[1] >= 1 else 0.0),
            "pc2": _round6(float(ideal_2d[idx_pos, 1]) if ideal_2d.shape[1] >= 2 else 0.0),
            "energy_total": _round6(float(e_ideal)),
            "label": "IDEAL",
        })

    # Deduplicate ideal points to unique positions
    seen_ideals: set[tuple[float, float]] = set()
    unique_ideals = []
    for ip in ideal_points:
        key = (ip["pc1"], ip["pc2"])
        if key not in seen_ideals:
            seen_ideals.add(key)
            unique_ideals.append(ip)
    ideal_points = unique_ideals

    route_path: list[dict[str, Any]] = []
    if config.get("include_route_overlay", True):
        route_plan = _load_route_plan(ctx)
        if route_plan is not None:
            steps = route_plan.get("steps") or []
            for step_idx, step in enumerate(steps):
                modeled_after = step.get("modeled_metrics_after") or {}
                if not modeled_after:
                    continue
                step_row = []
                valid_step = True
                for k in energy_keys:
                    v = modeled_after.get(k)
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        step_row.append(float(v))
                    else:
                        valid_step = False
                        break
                if not valid_step:
                    continue
                step_arr = np.array([step_row], dtype=np.float64)
                step_scaled = scaler.transform(step_arr)
                step_2d = pca.transform(step_scaled)
                weights = _default_energy_weights(esv.get("weights"))
                ideal_ref = entities[0].get("ideal") or {}
                step_energy = _energy_gap_for_metrics(modeled_after, ideal_ref, weights)
                route_path.append({
                    "step_index": step_idx,
                    "pc1": _round6(float(step_2d[0, 0])),
                    "pc2": _round6(float(step_2d[0, 1]) if step_2d.shape[1] >= 2 else 0.0),
                    "energy": _round6(float(step_energy)),
                    "label": str(step.get("lever_id", f"step_{step_idx}")),
                })

    pca_variance = [_round6(float(v)) for v in pca.explained_variance_ratio_]

    landscape_json = {
        "schema_version": "kona_3d_landscape.v1",
        "pca_components": [[_round6(float(c)) for c in row] for row in pca.components_.tolist()],
        "pca_explained_variance_ratio": pca_variance,
        "scaler_min": [_round6(float(v)) for v in scaler.data_min_.tolist()],
        "scaler_scale": [_round6(float(v)) for v in scaler.scale_.tolist()],
        "grid": {
            "x": [_round6(float(v)) for v in grid_x.tolist()],
            "y": [_round6(float(v)) for v in grid_y.tolist()],
            "z": [[_round6(float(v)) for v in row] for row in grid_z.tolist()],
        },
        "entities": entity_points,
        "ideal_points": ideal_points,
        "route_path": route_path,
    }

    html = _landscape_html(
        landscape_json["grid"]["x"],
        landscape_json["grid"]["y"],
        landscape_json["grid"]["z"],
        entity_points,
        ideal_points,
        route_path,
        pca_variance,
    )

    artifacts = [
        _artifact(ctx, plugin_id, "kona_3d_landscape.json", landscape_json),
        _artifact_html(ctx, plugin_id, "kona_3d_landscape.html", html),
    ]

    findings = [{
        "id": stable_id(f"{plugin_id}:landscape"),
        "kind": "kona_3d_landscape",
        "measurement_type": "measured",
        "pca_explained_variance_total": _round6(sum(pca_variance)),
        "entity_count": len(valid_indices),
        "ideal_count": len(ideal_points),
        "route_steps": len(route_path),
        "grid_resolution": grid_resolution,
    }]

    return PluginResult(
        status="ok",
        summary=f"3D landscape: {len(valid_indices)} entities, PCA {sum(pca_variance):.0%} variance",
        metrics=_basic_metrics(df, sample_meta),
        findings=findings,
        artifacts=artifacts,
        error=None,
        references=default_references_for_plugin(plugin_id),
        debug={},
    )


# ---------------------------------------------------------------------------
# Handler 2: Kohonen SOM Energy Map
# ---------------------------------------------------------------------------

def _kohonen_energy_map_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if not config.get("enabled", False):
        return _na_result(plugin_id, "Kohonen map disabled", "disabled_by_config", df, sample_meta)

    try:
        from minisom import MiniSom
    except ImportError:
        return _na_result(plugin_id, "minisom not installed", "minisom_not_installed", df, sample_meta)

    esv = _load_energy_state_vector(ctx)
    if esv is None:
        return _na_result(plugin_id, "Missing energy state vector", "missing_energy_artifact", df, sample_meta)

    entities = _extract_entities(esv)
    energy_keys = _energy_keys_from_esv(esv)
    min_entities = int(config.get("min_entities", 5))

    feature_matrix, valid_indices = _build_feature_matrix(entities, energy_keys)
    if len(valid_indices) < min_entities:
        return _na_result(
            plugin_id,
            f"Insufficient entities ({len(valid_indices)} < {min_entities})",
            "insufficient_entities",
            df,
            sample_meta,
        )

    if timer.exceeded():
        return _na_result(plugin_id, "Budget exceeded before SOM training", "budget_exceeded", df, sample_meta)

    from sklearn.preprocessing import MinMaxScaler

    run_seed = int(ctx.run_seed)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    n_features = X_scaled.shape[1]

    n = len(valid_indices)
    som_max_side = int(config.get("som_max_side", 20))
    side = max(3, min(som_max_side, int(math.sqrt(5 * math.sqrt(n)))))
    som_shape = [side, side]

    som = MiniSom(
        x=side,
        y=side,
        input_len=n_features,
        sigma=max(1.0, side / 2.0),
        learning_rate=0.5,
        neighborhood_function="gaussian",
        random_seed=run_seed,
    )
    som.pca_weights_init(X_scaled)

    max_iter = int(config.get("som_max_iterations", 50000))
    iterations = min(500 * n, max_iter)
    som.train(X_scaled, num_iteration=iterations, random_order=False)

    entity_bmus = []
    for idx_pos, orig_idx in enumerate(valid_indices):
        bmu = som.winner(X_scaled[idx_pos])
        entity_bmus.append({
            "entity_key": str(entities[orig_idx].get("entity_key", "")),
            "bmu_x": int(bmu[0]),
            "bmu_y": int(bmu[1]),
            "energy_total": _round6(float(entities[orig_idx].get("energy_total", 0.0))),
            "quantization_error": _round6(float(np.linalg.norm(X_scaled[idx_pos] - som.get_weights()[bmu[0], bmu[1]]))),
        })

    ideal_matrix = _build_ideal_matrix(entities, energy_keys, valid_indices)
    ideal_scaled = scaler.transform(ideal_matrix)
    seen_ideal_bmus: set[tuple[int, int]] = set()
    ideal_bmus = []
    for iv in ideal_scaled:
        bmu = som.winner(iv)
        key = (int(bmu[0]), int(bmu[1]))
        if key not in seen_ideal_bmus:
            seen_ideal_bmus.add(key)
            ideal_bmus.append({"bmu_x": key[0], "bmu_y": key[1], "label": "IDEAL"})

    umatrix = som.distance_map()

    weights_arr = som.get_weights()
    energy_grid = np.zeros(som_shape)
    esv_weights = _default_energy_weights(esv.get("weights"))
    # Use the first entity's ideal as reference for energy computation
    ref_ideal = entities[0].get("ideal") or {}
    for i in range(side):
        for j in range(side):
            neuron_scaled = weights_arr[i, j].reshape(1, -1)
            neuron_metrics_arr = scaler.inverse_transform(neuron_scaled)[0]
            neuron_metrics = {k: float(neuron_metrics_arr[ki]) for ki, k in enumerate(energy_keys)}
            energy_grid[i, j] = _energy_gap_for_metrics(neuron_metrics, ref_ideal, esv_weights)

    route_bmus: list[dict[str, Any]] = []
    if config.get("include_route_overlay", True):
        route_plan = _load_route_plan(ctx)
        if route_plan is not None:
            steps = route_plan.get("steps") or []
            for step_idx, step in enumerate(steps):
                modeled_after = step.get("modeled_metrics_after") or {}
                if not modeled_after:
                    continue
                step_row = []
                valid_step = True
                for k in energy_keys:
                    v = modeled_after.get(k)
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        step_row.append(float(v))
                    else:
                        valid_step = False
                        break
                if not valid_step:
                    continue
                step_arr = np.array([step_row], dtype=np.float64)
                step_scaled = scaler.transform(step_arr)
                bmu = som.winner(step_scaled[0])
                route_bmus.append({
                    "step_index": step_idx,
                    "bmu_x": int(bmu[0]),
                    "bmu_y": int(bmu[1]),
                    "label": str(step.get("lever_id", f"step_{step_idx}")),
                })

    qe = float(som.quantization_error(X_scaled))
    te = float(som.topographic_error(X_scaled))

    kohonen_json = {
        "schema_version": "kohonen_energy_map.v1",
        "som_shape": som_shape,
        "sigma": _round6(max(1.0, side / 2.0)),
        "learning_rate": 0.5,
        "iterations": iterations,
        "quantization_error": _round6(qe),
        "topographic_error": _round6(te),
        "umatrix": [[_round6(float(umatrix[i, j])) for j in range(side)] for i in range(side)],
        "energy_grid": [[_round6(float(energy_grid[i, j])) for j in range(side)] for i in range(side)],
        "entity_bmus": entity_bmus,
        "ideal_bmus": ideal_bmus,
        "route_path_bmus": route_bmus,
        "feature_names": energy_keys,
    }

    html = _kohonen_html(
        kohonen_json["umatrix"],
        kohonen_json["energy_grid"],
        entity_bmus,
        ideal_bmus,
        route_bmus,
        som_shape,
    )

    artifacts = [
        _artifact(ctx, plugin_id, "kohonen_energy_map.json", kohonen_json),
        _artifact_html(ctx, plugin_id, "kohonen_energy_map.html", html),
    ]

    findings = [{
        "id": stable_id(f"{plugin_id}:kohonen"),
        "kind": "kohonen_energy_map",
        "measurement_type": "measured",
        "som_shape": som_shape,
        "quantization_error": _round6(qe),
        "topographic_error": _round6(te),
        "entity_count": len(valid_indices),
        "ideal_bmu_count": len(ideal_bmus),
        "route_steps_mapped": len(route_bmus),
    }]

    return PluginResult(
        status="ok",
        summary=f"Kohonen SOM {side}x{side}: {len(valid_indices)} entities, QE={qe:.4f}, TE={te:.4f}",
        metrics=_basic_metrics(df, sample_meta),
        findings=findings,
        artifacts=artifacts,
        error=None,
        references=default_references_for_plugin(plugin_id),
        debug={},
    )


# ---------------------------------------------------------------------------
# Handler 3: Weight Learner
# ---------------------------------------------------------------------------

def _kona_weight_learner_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if not config.get("enabled", False):
        return _na_result(plugin_id, "Weight learner disabled", "disabled_by_config", df, sample_meta)

    esv = _load_energy_state_vector(ctx)
    if esv is None:
        return _na_result(plugin_id, "Missing energy state vector", "missing_energy_artifact", df, sample_meta)

    actions = _load_verified_actions(ctx)
    if actions is None:
        return _na_result(plugin_id, "Missing verified actions", "missing_actions_artifact", df, sample_meta)

    energy_keys = _energy_keys_from_esv(esv)
    entities = _extract_entities(esv)
    if not entities:
        return _na_result(plugin_id, "No entities in energy state vector", "no_entities", df, sample_meta)

    ref_ideal = entities[0].get("ideal") or {}
    ref_observed = entities[0].get("observed") or entities[0]

    calibration_actions = []
    impacts = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        est_pct = action.get("estimated_improvement_pct")
        if not isinstance(est_pct, (int, float)) or not math.isfinite(float(est_pct)):
            continue
        confidence = float(action.get("confidence", 0.5))
        if not math.isfinite(confidence):
            confidence = 0.5
        impact = confidence * float(est_pct) / 100.0
        modeled = action.get("modeled_metrics_after") or {}
        observed = action.get("observed_metrics") or ref_observed
        if not modeled:
            continue
        calibration_actions.append({
            "lever_id": str(action.get("lever_id", "")),
            "observed_metrics": observed,
            "ideal_metrics": ref_ideal,
            "modeled_metrics_after": modeled,
            "estimated_improvement_pct": float(est_pct),
            "confidence": confidence,
        })
        impacts.append(impact)

    min_cal = int(config.get("min_calibration_actions", 3))
    if len(calibration_actions) < min_cal:
        return _na_result(
            plugin_id,
            f"Insufficient calibration data ({len(calibration_actions)} < {min_cal})",
            "insufficient_calibration_data",
            df,
            sample_meta,
        )

    if timer.exceeded():
        return _na_result(plugin_id, "Budget exceeded before optimization", "budget_exceeded", df, sample_meta)

    from scipy.optimize import minimize
    from scipy.stats import spearmanr

    default_weights = _default_energy_weights(esv.get("weights"))
    key_list = sorted(energy_keys)

    def _compute_deltas(w_dict: dict[str, float]) -> list[float]:
        deltas = []
        for ca in calibration_actions:
            e_before = _energy_gap_for_metrics(ca["observed_metrics"], ca["ideal_metrics"], w_dict)
            e_after = _energy_gap_for_metrics(ca["modeled_metrics_after"], ca["ideal_metrics"], w_dict)
            deltas.append(float(e_before - e_after))
        return deltas

    impacts_arr = np.array(impacts, dtype=np.float64)

    def objective(w_raw: np.ndarray) -> float:
        w_exp = np.exp(w_raw)
        w_norm = w_exp / w_exp.sum()
        w_dict = {k: float(w_norm[i]) for i, k in enumerate(key_list)}
        deltas = _compute_deltas(w_dict)
        if len(set(deltas)) < 2 or len(set(impacts)) < 2:
            return 0.0
        corr, _ = spearmanr(deltas, impacts_arr)
        if not math.isfinite(corr):
            return 0.0
        return -corr

    w0 = np.array([math.log(max(default_weights.get(k, 1.0), 1e-9)) for k in key_list], dtype=np.float64)
    max_iter = int(config.get("max_optimizer_iterations", 200))

    result = minimize(
        objective,
        w0,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-8},
    )

    learned_raw = np.exp(result.x)
    learned_norm = learned_raw / learned_raw.sum()
    learned_weights = {k: _round6(float(learned_norm[i])) for i, k in enumerate(key_list)}

    # Normalize default weights for fair comparison
    default_sum = sum(default_weights.get(k, 1.0) for k in key_list)
    default_norm = {k: _round6(float(default_weights.get(k, 1.0) / max(default_sum, 1e-9))) for k in key_list}

    default_deltas = _compute_deltas(default_norm)
    learned_deltas = _compute_deltas(learned_weights)

    default_corr_val = 0.0
    learned_corr_val = 0.0
    if len(set(default_deltas)) >= 2 and len(set(impacts)) >= 2:
        dc, _ = spearmanr(default_deltas, impacts_arr)
        default_corr_val = float(dc) if math.isfinite(dc) else 0.0
    if len(set(learned_deltas)) >= 2 and len(set(impacts)) >= 2:
        lc, _ = spearmanr(learned_deltas, impacts_arr)
        learned_corr_val = float(lc) if math.isfinite(lc) else 0.0

    improvement = _round6(learned_corr_val - default_corr_val)
    improved = improvement > 0.0

    # Find top weight change
    max_change_metric = key_list[0] if key_list else ""
    max_change_val = 0.0
    for k in key_list:
        delta = abs(learned_weights.get(k, 0.0) - default_norm.get(k, 0.0))
        if delta > max_change_val:
            max_change_val = delta
            max_change_metric = k

    weights_json = {
        "schema_version": "kona_weight_learner.v1",
        "default_weights": default_norm,
        "learned_weights": learned_weights,
        "calibration_actions": len(calibration_actions),
        "default_rank_correlation": _round6(default_corr_val),
        "learned_rank_correlation": _round6(learned_corr_val),
        "improvement": improvement,
        "improved": improved,
        "optimization_converged": bool(result.success),
        "optimization_iterations": int(result.nit),
    }

    # Build per-action comparison
    default_ranked = sorted(range(len(default_deltas)), key=lambda i: -default_deltas[i])
    learned_ranked = sorted(range(len(learned_deltas)), key=lambda i: -learned_deltas[i])
    default_rank_map = {i: rank + 1 for rank, i in enumerate(default_ranked)}
    learned_rank_map = {i: rank + 1 for rank, i in enumerate(learned_ranked)}

    per_action = []
    for i, ca in enumerate(calibration_actions):
        per_action.append({
            "lever_id": ca["lever_id"],
            "default_delta_energy": _round6(default_deltas[i]),
            "learned_delta_energy": _round6(learned_deltas[i]),
            "evidence_impact": _round6(impacts[i]),
            "default_rank": default_rank_map[i],
            "learned_rank": learned_rank_map[i],
        })

    comparison_json = {"per_action_comparison": per_action}

    artifacts = [
        _artifact(ctx, plugin_id, "learned_energy_weights.json", weights_json),
        _artifact(ctx, plugin_id, "weight_comparison.json", comparison_json),
    ]

    findings = [{
        "id": stable_id(f"{plugin_id}:weights"),
        "kind": "kona_weight_learner",
        "measurement_type": "modeled",
        "calibration_actions": len(calibration_actions),
        "default_rank_correlation": _round6(default_corr_val),
        "learned_rank_correlation": _round6(learned_corr_val),
        "improvement": improvement,
        "improved": improved,
        "top_weight_change": {
            "metric": max_change_metric,
            "default": default_norm.get(max_change_metric, 0.0),
            "learned": learned_weights.get(max_change_metric, 0.0),
        },
    }]

    return PluginResult(
        status="ok",
        summary=f"Weight learner: {len(calibration_actions)} actions, corr {default_corr_val:.3f}->{learned_corr_val:.3f} ({'improved' if improved else 'no improvement'})",
        metrics=_basic_metrics(df, sample_meta),
        findings=findings,
        artifacts=artifacts,
        error=None,
        references=default_references_for_plugin(plugin_id),
        debug={},
    )


KONA_VIS_HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_kona_3d_landscape_v1": _kona_3d_landscape_v1,
    "analysis_kohonen_energy_map_v1": _kohonen_energy_map_v1,
    "analysis_kona_weight_learner_v1": _kona_weight_learner_v1,
}
