"""Shared USD asset processing utilities.

Provides mathematically exact operations on USD part files, callable both
from the command-line tool (tools/center_usd_assets.py) and at runtime inside
Isaac Sim (e.g., after importing a new part).

Public API
----------
rigorously_center_prim(target_prim, stage)
    Modify target_prim's xformOp:translate so the AABB centroid of all
    descendant mesh geometry is at world origin.  Returns the centroid offset
    that was applied (Gf.Vec3d), or None if no geometry was found.

find_mesh_container(stage, root_prim=None)
    Auto-detect the Xform prim that directly parents the mesh geometry in a
    typical STL-imported USD file (World → node_ → mesh_).  Returns None when
    no suitable prim is found.

center_stage_file(input_path, output_path=None, target_prim_path=None)
    Open a USD file, center its geometry, and save.  Returns a CenterResult.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CenterResult:
    """Outcome of a single-file centering operation."""

    input_path: str
    output_path: str
    success: bool
    centroid_before: Optional[tuple]   # (x, y, z) in USD units
    target_prim_path: str
    message: str = ""


# ---------------------------------------------------------------------------
# Core math — all pxr imports are deferred so this module is importable even
# without the pxr library on the import path (useful for mocking in tests).
# ---------------------------------------------------------------------------

def rigorously_center_prim(
    target_prim,
    stage,
    logger: Callable[[str], None] = print,
) -> Optional["Gf.Vec3d"]:  # type: ignore[name-defined]
    """Translate *target_prim* so its descendant geometry AABB centroid is at
    world origin.

    Algorithm
    ---------
    1. Create a ``UsdGeom.BBoxCache`` with ``useExtentsHint=False`` (forces
       computation from actual vertex data, not cached extents).
    2. Call ``ComputeWorldBound(target_prim)`` to get the exact world-space
       AABB.
    3. Compute ``centroid = (aabb.min + aabb.max) / 2``.
    4. Fetch target_prim's existing ``xformOp:translate`` (zero if absent).
    5. Apply ``new_translate = existing_translate − centroid``.
       Proof: new_world_centroid
              = new_translate + local_vertex_centroid
              = (T − centroid) + (centroid − T)   [since centroid = T + C_local]
              = 0  ✓
    6. Preserve any existing ``xformOp:orient`` / ``xformOp:scale`` ops.

    Returns the world-space centroid that was subtracted, or None if no
    descendant geometry was found.
    """
    from pxr import Usd, UsdGeom, Gf  # type: ignore

    tc = Usd.TimeCode.Default()

    # Step 1-2: Compute world-space AABB via BBoxCache.
    # useExtentsHint=False forces pxr to compute from vertex positions rather
    # than the (possibly stale or absent) extentsHint attribute.
    cache = UsdGeom.BBoxCache(tc, [UsdGeom.Tokens.default_], useExtentsHint=False)
    bbox  = cache.ComputeWorldBound(target_prim)
    aabb  = bbox.ComputeAlignedRange()

    if aabb.IsEmpty():
        logger(f"[CENTER] No geometry found under {target_prim.GetPath()}")
        return None

    # Step 3: AABB centroid in world space.
    aabb_min = aabb.GetMin()
    aabb_max = aabb.GetMax()
    world_centroid = Gf.Vec3d(
        (aabb_min[0] + aabb_max[0]) * 0.5,
        (aabb_min[1] + aabb_max[1]) * 0.5,
        (aabb_min[2] + aabb_max[2]) * 0.5,
    )

    # Step 4: Existing translate on target_prim (Gf.Vec3d, default 0).
    xf = UsdGeom.Xformable(target_prim)
    if not xf:
        logger(f"[CENTER] {target_prim.GetPath()} is not Xformable")
        return None

    existing_translate = Gf.Vec3d(0.0, 0.0, 0.0)
    translate_op: Optional[UsdGeom.XformOp] = None  # type: ignore[assignment]

    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and not op.IsInverseOp():
            raw = op.Get(tc)
            if raw is not None:
                existing_translate = Gf.Vec3d(float(raw[0]), float(raw[1]), float(raw[2]))
            translate_op = op
            break

    # Step 5: Compute new translate.
    # new_translate = existing_translate − world_centroid
    new_translate = existing_translate - world_centroid

    # Step 6: Apply — modify the existing op or add a new one.
    if translate_op is not None:
        translate_op.Set(new_translate)
    else:
        # AddTranslateOp inserts at the front of the op stack by default.
        new_op = xf.AddXformOp(
            UsdGeom.XformOp.TypeTranslate,
            UsdGeom.XformOp.PrecisionDouble,
        )
        new_op.Set(new_translate)

    logger(
        f"[CENTER] {target_prim.GetPath()}"
        f"  centroid_before=({world_centroid[0]:.4f}, {world_centroid[1]:.4f}, {world_centroid[2]:.4f})"
        f"  new_translate=({new_translate[0]:.4f}, {new_translate[1]:.4f}, {new_translate[2]:.4f})"
    )
    return world_centroid


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def find_mesh_container(
    stage,
    root_prim=None,
    exclude_prim=None,
) -> Optional["Usd.Prim"]:  # type: ignore[name-defined]
    """Return the shallowest Xform that directly parents at least one Mesh prim.

    In the typical STL-imported layout (World → node_ → mesh_) this returns
    ``node_`` — the prim whose translate we should modify.

    Parameters
    ----------
    root_prim:
        Restrict search to descendants of this prim.  Defaults to the stage's
        defaultPrim (or pseudo-root).
    exclude_prim:
        Xform to exclude from consideration (usually the defaultPrim / file
        root, whose transform will be replaced by MagicAssembly).
    """
    from pxr import Usd, UsdGeom  # type: ignore

    if root_prim is None:
        root_prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()

    if exclude_prim is None:
        exclude_prim = stage.GetDefaultPrim()

    # BFS: find the shallowest Xform that has a Mesh child.
    # We walk depth-first but collect only the shallowest matches.
    candidates: List = []
    for prim in Usd.PrimRange(root_prim):
        if prim == root_prim:
            continue
        if prim.GetTypeName() != "Xform":
            continue
        if exclude_prim and str(prim.GetPath()) == str(exclude_prim.GetPath()):
            continue
        for child in prim.GetChildren():
            if child.GetTypeName() == "Mesh":
                candidates.append(prim)
                break

    if not candidates:
        # Fallback: return root_prim itself (flat file where mesh is directly
        # under defaultPrim).
        for child in root_prim.GetChildren():
            if child.GetTypeName() == "Mesh":
                return root_prim
        return None

    # Return shallowest candidate.
    candidates.sort(key=lambda p: len(p.GetPath().pathString.split("/")))
    return candidates[0]


# ---------------------------------------------------------------------------
# File-level entry point
# ---------------------------------------------------------------------------

def center_stage_file(
    input_path: str,
    output_path: Optional[str] = None,
    target_prim_path: Optional[str] = None,
    logger: Callable[[str], None] = print,
) -> CenterResult:
    """Open *input_path*, center its geometry, and save to *output_path*.

    Parameters
    ----------
    input_path:
        Path to a ``.usd`` / ``.usda`` / ``.usdc`` file.
    output_path:
        Where to write the result.  If None, overwrites *input_path*.
    target_prim_path:
        Explicit USD path of the Xform to center (e.g. ``/World/node_``).
        If None, uses :func:`find_mesh_container` to auto-detect.
    logger:
        Callable for progress / diagnostic messages.

    Returns
    -------
    CenterResult
    """
    from pxr import Usd  # type: ignore

    out = output_path or input_path

    try:
        stage = Usd.Stage.Open(input_path)
    except Exception as exc:
        return CenterResult(
            input_path=input_path,
            output_path=out,
            success=False,
            centroid_before=None,
            target_prim_path="",
            message=f"Failed to open stage: {exc}",
        )

    # Find target prim.
    if target_prim_path:
        target = stage.GetPrimAtPath(target_prim_path)
        if not (target and target.IsValid()):
            return CenterResult(
                input_path=input_path,
                output_path=out,
                success=False,
                centroid_before=None,
                target_prim_path=target_prim_path,
                message=f"Target prim '{target_prim_path}' not found in stage",
            )
    else:
        target = find_mesh_container(stage)
        if target is None:
            return CenterResult(
                input_path=input_path,
                output_path=out,
                success=False,
                centroid_before=None,
                target_prim_path="",
                message="No mesh-containing Xform found — is there geometry in the file?",
            )

    target_path_str = str(target.GetPath())

    # Center.
    centroid = rigorously_center_prim(target, stage, logger=logger)
    if centroid is None:
        return CenterResult(
            input_path=input_path,
            output_path=out,
            success=False,
            centroid_before=None,
            target_prim_path=target_path_str,
            message="BBoxCache returned empty bound — geometry may have no vertex data",
        )

    # Save.
    try:
        if out == input_path:
            stage.GetRootLayer().Save()
        else:
            stage.GetRootLayer().Export(out)
    except Exception as exc:
        return CenterResult(
            input_path=input_path,
            output_path=out,
            success=False,
            centroid_before=tuple(centroid),
            target_prim_path=target_path_str,
            message=f"Failed to save: {exc}",
        )

    return CenterResult(
        input_path=input_path,
        output_path=out,
        success=True,
        centroid_before=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        target_prim_path=target_path_str,
        message="OK",
    )
