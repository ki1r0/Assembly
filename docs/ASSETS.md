# External assets

Set `ASSEMBLY_ASSET_ROOT` to an existing local directory with this layout. USD references and textures beneath the robot asset must also remain available.

```text
<ASSEMBLY_ASSET_ROOT>/
  parts/
    M6 Hub Bolt.usd
    Casing Top.usd
    Hub Cover Output.usd
  vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/
    franka.usd
    ... referenced files ...
```

Assets are not downloaded or redistributed by this repository. Obtain the correct assembly USDs and NVIDIA robot assets under their respective licenses. Geometry and fixture calibration correspond to these original assets; replacing them requires recalibration and new validation. The launcher mounts the asset directory read-only. It does not require the previous machine's project directory or asset symlink.

Scene configuration uses `config/scenarios.json`, `config/fixture_calibration.json` and `isaac_runtime/config/bolt_insertion.yaml`. Private reset and evaluator geometry must not be inserted into model prompts, memory, observations or tool results. The backend loads the root config; the simulator-side copy is retained for legacy runtime defaults.
