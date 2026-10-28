"""Deterministic local-axis previews of public sampled CAD, never scene state."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw


def render_public_cad_preview(points, normals, opposite_side=False):
    """Return (PNG bytes, compact metadata) from finite metre-scale Nx3 samples.

    Normals must be nonzero and correspond to points. Front-facing point splats
    are painted far-to-near; no convex fill or inferred surface closes cavities.
    Sampling gaps and inconsistent mesh winding can still hide/distort details.
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 4 or
            normals.shape != points.shape or not np.isfinite(points).all() or
            not np.isfinite(normals).all()):
        raise ValueError("Finite compatible Nx3 points and normals, N >= 4, required")
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
        raise ValueError("Nonzero finite normals required")
    low, high = points.min(axis=0), points.max(axis=0)
    extent = high - low
    if not np.isfinite(extent).all() or np.any(extent <= 1e-9):
        raise ValueError("Nondegenerate bounds on all three local axes required")
    normals = normals / lengths[:, None]
    centered = points - (low / 2 + high / 2)
    views = [
        ("Along +X", [1, 0, 0], [0, 0, 1]),
        ("Along +Y", [0, 1, 0], [0, 0, 1]),
        ("Along +Z", [0, 0, 1], [0, 1, 0]),
        ("Oblique +X +Y +Z", [1, 1, 1], [0, 0, 1]),
    ]
    if opposite_side:
        views=[(label.replace('+','-'),(-np.asarray(direction)).tolist(),up) for label,direction,up in views]
    image = Image.new("RGB", (1000, 860), "#edf0ed")
    draw = ImageDraw.Draw(image)
    draw.text((18, 14), "SAMPLED PUBLIC CAD REFERENCE | local axes | metres", fill="#172c35")
    draw.text((18, 32), "Not a live camera image or scene pose. Sparse samples; cavities may be incomplete.", fill="#374b53")
    bounds_text = "  ".join(f"{axis}: [{a:.6g}, {b:.6g}]" for axis, a, b in zip("XYZ", low, high))
    draw.text((18, 51), "Sample bounds (m)  " + bounds_text, fill="#172c35")
    draw.text((18, 70), "Extents (m)  " + ", ".join(f"{a}={v:.6g}" for a, v in zip("XYZ", extent)), fill="#172c35")
    descriptions = []
    for index, (label, direction, up) in enumerate(views):
        x0, y0 = 12 + (index % 2) * 494, 98 + (index // 2) * 376
        draw.rectangle((x0, y0, x0 + 482, y0 + 364), fill="#fcfcf8", outline="#bdc8c7")
        draw.text((x0 + 12, y0 + 10), label + " (local viewer direction)", fill="#172c35")
        toward = np.asarray(direction, dtype=float)
        toward /= np.linalg.norm(toward)
        right = np.cross(np.asarray(up, dtype=float), toward)
        right /= np.linalg.norm(right)
        vertical = np.cross(toward, right)
        basis = np.column_stack((right, vertical, toward))
        projected = centered @ basis
        span = np.ptp(projected[:, :2], axis=0)
        scale = min(428 / span[0], 260 / span[1])
        mid = (projected[:, :2].min(axis=0) + projected[:, :2].max(axis=0)) / 2
        xy = (projected[:, :2] - mid) * [scale, -scale] + [x0 + 241, y0 + 175]
        facing = normals @ toward
        visible = np.flatnonzero(facing > 0)
        # Coordinate tie-breaks keep painter order independent of cache ordering.
        order = np.lexsort((points[visible, 2], points[visible, 1], points[visible, 0],
                            facing[visible], projected[visible, 2]))
        for i in visible[order]:
            x, y = np.rint(xy[i]).astype(int)
            shade = float(.3 + .7 * facing[i])
            color = tuple(int(v * shade) for v in (58, 145, 164))
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
        origin = np.array([x0 + 62, y0 + 323])
        for axis, vector, color in zip("XYZ", np.eye(3), ("#b43d32", "#257c45", "#325dab")):
            delta = np.array([vector @ right, -(vector @ vertical)]) * 32
            end = origin + delta
            draw.line((tuple(origin), tuple(end)), fill=color, width=2)
            draw.text(tuple(end + [3, -5]), "+" + axis, fill=color)
        draw.text((x0 + 125, y0 + 327), "Local axes; independently fitted view", fill="#374b53")
        descriptions.append({"label": label, "viewer_direction_local": toward.tolist(),
                             "front_facing_samples": len(visible)})
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    metadata = {
        "reference": "sampled public CAD reference, not live camera or scene pose",
        "frame": "CAD local axes, metres",
        "sample_count": len(points),
        "sample_bounds_m": [low.tolist(), high.tolist()],
        "sample_extents_m": extent.tolist(),
        "image_size_px": list(image.size),
        "views": descriptions,
        "rendering": "front-facing normal shading, depth-sorted point splats; no surface reconstruction",
        "limitations": "Sparse samples and normal winding may obscure cavities; sample bounds are not exact mesh bounds",
    }
    return output.getvalue(), metadata
