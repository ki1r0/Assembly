# Recorded public RGB-D regression fixture

Source: `runs/cape-peg-mvp-20261026/ours-dev-attempt01/dev-u0`, observation 00003.
The actor selected pixel (469,392). The old adaptive ROI returned a different
opening centered near (505.2,398.5). Files are copied public camera observations,
not synthesized images, private object poses, or runtime action targets.

The regression checks selection identity and the measured surrounding rim plane.
Its expected plane is an offline calibration assertion only, never an online
input to the perception algorithm or policy. Original failed run remains intact.
