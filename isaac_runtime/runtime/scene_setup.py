from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Tuple

from .paths import ASSETS_DIR

REMOTE_ROOM_PREFIXES = (
    "Towel_Room01_",
    "Floor",
)

LOCAL_FALLBACK_ASSETS: Dict[str, str] = {
    "/Root/table_low_327": "fallbacks/table_low.usda",
    "/Root/Orange_01": "fallbacks/orange.usda",
    "/Root/plasticpail_a01": "fallbacks/plasticpail_a01.usda",
    "/Root/utilitybucket_a01": "fallbacks/utilitybucket_a01.usda",
}

FRANKA_FALLBACK_ASSET = "fallbacks/mock_franka.usda"
ROOM_SHELL_ASSET = "fallbacks/simple_room_shell.usda"
LOCAL_MIRROR_ROOT = ASSETS_DIR / "vendor" / "remote_mirror"
USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
KNOWN_NESTED_RIGID_BODIES: Dict[str, str] = {
    "/Root/Output_Shaft/Output_Gear": "/Root/Output_Shaft",
    "/Root/Transfer_Shaft/Transfer_Gear": "/Root/Transfer_Shaft",
}


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_remote_asset_path(asset_path: str) -> bool:
    text = str(asset_path or "").strip()
    return text.startswith("http://") or text.startswith("https://")


def _is_remote_reference_prim(prim) -> bool:
    for kind, _ in _iter_remote_composition_entries(prim):
        if kind in {"reference", "payload"}:
            return True
    return False


def _prim_is_effectively_unresolved(prim) -> bool:
    if not prim or not prim.IsValid() or not prim.IsActive():
        return True
    if not _is_remote_reference_prim(prim):
        return False
    children = list(prim.GetChildren())
    return len(children) <= 1 and not prim.GetTypeName()


def _iter_listop_items(list_op) -> Iterator[object]:
    if not list_op:
        return
    for attr in ("explicitItems", "prependedItems", "addedItems", "appendedItems", "orderedItems"):
        for item in getattr(list_op, attr, []) or []:
            yield item


def _iter_remote_composition_entries(prim) -> Iterator[Tuple[str, str]]:
    if not prim or not prim.IsValid():
        return
    refs = prim.GetMetadata("references")
    for item in _iter_listop_items(refs):
        asset_path = str(getattr(item, "assetPath", "") or "").strip()
        if _is_remote_asset_path(asset_path):
            yield ("reference", asset_path)
    payloads = prim.GetMetadata("payload")
    for item in _iter_listop_items(payloads):
        asset_path = str(getattr(item, "assetPath", "") or "").strip()
        if _is_remote_asset_path(asset_path):
            yield ("payload", asset_path)


def _clear_external_composition(prim) -> None:
    try:
        prim.GetReferences().ClearReferences()
    except Exception:
        pass
    try:
        prim.GetPayloads().ClearPayloads()
    except Exception:
        pass


def _apply_local_composition(stage, prim_path: str, asset_path: Path, kind: str, logger: Callable[[str], None]) -> bool:
    resolved = asset_path.resolve()
    if not resolved.is_file():
        logger(f"[ASSET][WARN] Local asset missing for {prim_path}: {resolved}")
        return False
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
    _clear_external_composition(prim)
    if kind == "payload":
        prim.GetPayloads().AddPayload(str(resolved))
    else:
        prim.GetReferences().AddReference(str(resolved))
    logger(f"[ASSET] Applied local {kind}: {prim_path} -> {resolved}")
    return True


def _apply_reference_fallback(stage, prim_path: str, rel_asset_path: str, logger: Callable[[str], None]) -> bool:
    asset_file = (ASSETS_DIR / rel_asset_path).resolve()
    return _apply_local_composition(stage, prim_path, asset_file, "reference", logger)


def _remote_url_to_local_path(url: str) -> Path:
    parsed = urllib.parse.urlparse(str(url).strip())
    host = parsed.netloc or "remote"
    remote_path = parsed.path.lstrip("/")
    return LOCAL_MIRROR_ROOT / host / remote_path


def _download_remote_file(url: str, target: Path, logger: Callable[[str], None]) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, tmp_target.open("wb") as handle:
            handle.write(response.read())
        tmp_target.replace(target)
        logger(f"[ASSET] Downloaded remote asset: {url} -> {target}")
        return True
    except Exception as exc:
        try:
            tmp_target.unlink(missing_ok=True)
        except Exception:
            pass
        logger(f"[ASSET][WARN] Failed to download {url}: {exc}")
        return False


def _iter_layer_dependencies(layer) -> Iterator[str]:
    try:
        for dep in layer.GetCompositionAssetDependencies() or []:
            text = str(dep or "").strip()
            if text:
                yield text
    except Exception:
        pass
    for dep in getattr(layer, "subLayerPaths", []) or []:
        text = str(dep or "").strip()
        if text:
            yield text


def _mirror_remote_asset(url: str, logger: Callable[[str], None], visited: set[str]) -> Path | None:
    remote_url = str(url or "").strip()
    if not _is_remote_asset_path(remote_url):
        return None
    if remote_url in visited:
        local_path = _remote_url_to_local_path(remote_url)
        return local_path if local_path.is_file() else None
    visited.add(remote_url)

    local_path = _remote_url_to_local_path(remote_url)
    if not local_path.is_file():
        if not _download_remote_file(remote_url, local_path, logger):
            return None

    if local_path.suffix.lower() not in USD_EXTENSIONS:
        return local_path

    try:
        from pxr import Sdf

        layer = Sdf.Layer.FindOrOpen(str(local_path))
    except Exception as exc:
        logger(f"[ASSET][WARN] Failed to inspect USD dependencies for {local_path}: {exc}")
        return local_path

    if not layer:
        return local_path

    parent_remote = remote_url.rsplit("/", 1)[0] + "/"
    for dep in _iter_layer_dependencies(layer):
        if dep.startswith(("anon:", "omni:", "file:")):
            continue
        if _is_remote_asset_path(dep):
            child_url = dep
        else:
            child_url = urllib.parse.urljoin(parent_remote, dep)
        _mirror_remote_asset(child_url, logger, visited)
    return local_path


def _collect_remote_targets(stage) -> List[Tuple[str, str, str]]:
    targets: List[Tuple[str, str, str]] = []
    for prim in stage.Traverse():
        for kind, asset_path in _iter_remote_composition_entries(prim):
            targets.append((prim.GetPath().pathString, kind, asset_path))
    return targets


def _iter_asset_values(value) -> Iterator[object]:
    try:
        from pxr import Sdf
    except Exception:
        return

    if isinstance(value, Sdf.AssetPath):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, Sdf.AssetPath):
                yield item


def _unresolved_asset_paths_under(stage, root_prim_path: str) -> List[Tuple[str, str, str]]:
    findings: List[Tuple[str, str, str]] = []
    prefix = root_prim_path.rstrip("/") + "/"
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if prim_path != root_prim_path and not prim_path.startswith(prefix):
            continue
        for attr in prim.GetAttributes():
            try:
                values = list(_iter_asset_values(attr.Get()))
            except Exception:
                continue
            for asset in values:
                raw_path = str(getattr(asset, "path", "") or "").strip()
                if not raw_path:
                    continue
                resolved = str(getattr(asset, "resolvedPath", "") or "").strip()
                if resolved:
                    continue
                findings.append((prim_path, attr.GetName(), raw_path))
    return findings


def _disable_nested_rigid_bodies(stage, logger: Callable[[str], None]) -> int:
    try:
        from pxr import UsdPhysics
    except Exception as exc:
        logger(f"[PHYSICS][WARN] Nested rigid-body cleanup skipped: {exc}")
        return 0

    def _disable_rigid_body(prim, parent_path: str) -> int:
        if prim is None or not prim.IsValid():
            return 0
        changed = False
        for api_schema in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
            try:
                if prim.HasAPI(api_schema) and prim.RemoveAPI(api_schema):
                    changed = True
            except Exception:
                continue

        if not changed:
            rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            if not rigid_api:
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            enabled_attr = rigid_api.GetRigidBodyEnabledAttr()
            if not enabled_attr or not enabled_attr.IsValid():
                enabled_attr = rigid_api.CreateRigidBodyEnabledAttr()
            if enabled_attr.Get() is not False:
                enabled_attr.Set(False)
                changed = True

        if changed:
            logger(f"[PHYSICS] Removed nested rigid body from {prim.GetPath()} under {parent_path}")
            return 1
        return 0

    def _rigid_body_enabled(prim) -> bool:
        if not prim or not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return False
        rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
        enabled_attr = rigid_api.GetRigidBodyEnabledAttr()
        if enabled_attr and enabled_attr.IsValid():
            value = enabled_attr.Get()
            if value is False:
                return False
        return True

    disabled = 0
    for child_path, parent_path in KNOWN_NESTED_RIGID_BODIES.items():
        child_prim = stage.GetPrimAtPath(child_path)
        disabled += _disable_rigid_body(child_prim, parent_path)

    for prim in stage.Traverse():
        if not _rigid_body_enabled(prim):
            continue
        ancestor = prim.GetParent()
        while ancestor and ancestor.IsValid():
            if _rigid_body_enabled(ancestor):
                disabled += _disable_rigid_body(prim, ancestor.GetPath())
                break
            ancestor = ancestor.GetParent()
    return disabled


def apply_local_scene_fallbacks(stage, *, logger: Callable[[str], None] = print) -> Dict[str, int]:
    prefer_local = _bool_env("HAC_PREFER_LOCAL_SCENE_ASSETS", True)
    auto_download = _bool_env("HAC_AUTO_DOWNLOAD_SCENE_ASSETS", True)
    enable_room_shell = _bool_env("HAC_ENABLE_ROOM_SHELL_FALLBACK", False)
    changed = 0
    skipped = 0
    mirrored = 0
    unresolved_replaced = 0

    visited: set[str] = set()
    targets = _collect_remote_targets(stage)
    room_shell_needed = False

    for prim_path, kind, remote_url in targets:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        if not (prefer_local or _prim_is_effectively_unresolved(prim)):
            skipped += 1
            continue

        local_path = _remote_url_to_local_path(remote_url)
        if not local_path.is_file() and auto_download:
            downloaded = _mirror_remote_asset(remote_url, logger, visited)
            if downloaded and downloaded.is_file():
                mirrored += 1
                local_path = downloaded

        if local_path.is_file():
            if _apply_local_composition(stage, prim_path, local_path, kind, logger):
                changed += 1
                continue

        fallback_rel = LOCAL_FALLBACK_ASSETS.get(prim_path)
        if fallback_rel and _apply_reference_fallback(stage, prim_path, fallback_rel, logger):
            changed += 1
            continue
        if prim_path == "/Franka" and _apply_reference_fallback(stage, prim_path, FRANKA_FALLBACK_ASSET, logger):
            changed += 1
            continue
        if prim.GetName().startswith(REMOTE_ROOM_PREFIXES):
            room_shell_needed = True
        skipped += 1

    for prim_path, fallback_rel in LOCAL_FALLBACK_ASSETS.items():
        prim = stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            continue
        unresolved = _unresolved_asset_paths_under(stage, prim_path)
        if not unresolved:
            continue
        logger(
            f"[ASSET] Replacing {prim_path} with local fallback because {len(unresolved)} asset path(s) under that prim did not resolve."
        )
        if _apply_reference_fallback(stage, prim_path, fallback_rel, logger):
            changed += 1
            unresolved_replaced += 1

    if room_shell_needed and enable_room_shell:
        room_path = "/Root/LocalRoomShell"
        room_prim = stage.GetPrimAtPath(room_path)
        if not room_prim or not room_prim.IsValid():
            stage.DefinePrim(room_path, "Xform")
        if _apply_reference_fallback(stage, room_path, ROOM_SHELL_ASSET, logger):
            changed += 1
            logger("[ASSET] Enabled simple room shell fallback because some room assets could not be localized.")

    nested_rigid_bodies_disabled = _disable_nested_rigid_bodies(stage, logger)
    changed += nested_rigid_bodies_disabled

    return {
        "changed": changed,
        "skipped": skipped,
        "mirrored": mirrored,
        "unresolved_replaced": unresolved_replaced,
        "nested_rigid_bodies_disabled": nested_rigid_bodies_disabled,
    }
