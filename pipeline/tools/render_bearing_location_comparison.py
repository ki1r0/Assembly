"""Render measured and source-defined bearing locations for human review only."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def rotation(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def main():
    run = ROOT / 'runs/ours-full-batch-20261023T095818Z/18-bearing-input-top-onto-shaft'
    m = json.loads((run / 'private/evaluator-trace.json').read_text())[-1]['measurement']
    rf = rotation(m['fixture_quaternion_world_wxyz'])
    a = rf.T @ (np.array(m['moving_position_world_m']) - m['fixture_position_world_m'])
    ra = rf.T @ rotation(m['moving_quaternion_world_wxyz'])
    b = np.array([0., .002732481, 0.])
    rb = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    shaft = np.load(ROOT / 'assets/public_cad_models/shaft_input.npz')['points']
    bearing = np.load(ROOT / 'assets/public_cad_models/bearing_32205_input.npz')['points']
    canvas = Image.new('RGB', (1600, 960), '#f5f5f2')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    draw.text((40, 20), 'Task 18: same shaft, same scale; two DIFFERENT installation locations', fill='#17252a', font=font)
    scale = 5600

    def project(p, cy):
        # Local -Y points toward the upper end in the live scene.
        return 800 + p[:, 1]*scale, cy - p[:, 0]*scale

    for cy, label, t, r, color in [
        (260, 'A: old rollout final pose', a, ra, '#c44836'),
        (590, 'B: original assembly definition', b, rb, '#2369b1'),
    ]:
        draw.text((40, cy-195 if cy == 260 else cy-105), label, fill=color, font=font)
        for pts, ink in [(shaft, '#6e797b'), (bearing @ r.T + t, color)]:
            px, py = project(pts, cy)
            order = np.argsort(pts[:, 2])
            for i in order:
                draw.ellipse((px[i]-1, py[i]-1, px[i]+1, py[i]+1), fill=ink)
        origin_x = 800 + t[1]*scale
        draw.line((origin_x, cy+150, origin_x, cy-115), fill=color, width=2)
        draw.text((40, cy+155), f'Bearing CAD root: shaft-local Y = {t[1]*1000:+.3f} mm', fill=color, font=small)
        for mm in range(-100, 101, 25):
            x = 800 + mm/1000*scale
            draw.line((x, cy+185, x, cy+193), fill='#333333', width=2)
            draw.text((x-20, cy+195), str(mm), fill='#333333', font=small)
    draw.text((40, 830), f'CAD-root displacement along shaft: {abs(a[1]-b[1])*1000:.1f} mm (not a seating tolerance).', fill='#17252a', font=font)
    draw.text((40, 870), 'A preserves the measured tilt and lateral offset. B uses the original -90 deg Z orientation.', fill='#17252a', font=small)
    draw.text((40, 905), 'CAD point-cloud side view, not a simulator screenshot. Neither panel establishes physical success.', fill='#17252a', font=small)
    out = ROOT / 'diagnostics/evaluator-calibration-20261024'
    out.mkdir(parents=True, exist_ok=True)
    canvas.save(out / 'bearing18-location-A-vs-B.png')
    print(out / 'bearing18-location-A-vs-B.png')


if __name__ == '__main__':
    main()
