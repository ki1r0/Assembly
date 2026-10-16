# Known limitations

This extraction preserves the latest controller behavior rather than changing its task policy. It does not establish completed peg insertion. The previous final four episodes all returned native success false; visible insertion progress and tool-level seating candidates were insufficient.

The insertion skill assumes a visually grasped upright bolt. In the horizontal-start trial, later insertion could restore the originally stored grasp orientation and undo a manual rotation; in-hand shaft estimation could also fail. The default prompt documents the upright-skill scope. Flat-start execution remains experimental.

Submillimetre control tolerances are target settings, not calibrated perception accuracy guarantees. Aperture visibility, counterbores, occlusion, in-hand tilt and deep contact can affect insertion. Lowering image resolution can make this worse.

Existing native evaluator criteria, force/joint limits, speed/tracking emergency stops and actor file restrictions are preserved. A timeout can contain real partial motion or stagnation. Neither bilateral force nor gripper closure identifies the held object.

The bundled RPent is a selected local adaptation: API planner, session/tools/memory/dashboard support, and generic Env/VLA/RPC interfaces. It is not a complete installation of every upstream robot, CLI, backend or learning stack. Historical benchmark memories are intentionally absent, so model behavior can differ from earlier experiments.
