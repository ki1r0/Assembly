"""Decode exact Bolt-cap finger contacts from PhysX contact-report buffers.

PhysX 5.1 documents the contact normal as pointing from shape 1 to shape 0
and says impulse is a world-space vector whose division by dt gives force.
The SDK contact extractor constructs that vector as ``normal * impulse``
(PhysX 5.4.1 ``PxContactPair::extractContacts``). Thus the vector is oriented
for shape/actor 0; negate it only when Bolt is collider 1. Raw SDK values are
preserved, including their signs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import hypot, isfinite
from operator import index
from typing import Literal


BOLT_ACTOR_PATH = "/World/envs/env_0/Bolt"
CAP_COLLIDER_PATH = "/World/envs/env_0/Bolt/collision_cap"
FINGER_ACTOR_PATHS = {
    "left": "/World/envs/env_0/Robot/panda_leftfinger",
    "right": "/World/envs/env_0/Robot/panda_rightfinger",
}

Vector3 = tuple[float, float, float]
ContactEventName = Literal["CONTACT_FOUND", "CONTACT_PERSIST"]


@dataclass(frozen=True)
class BoltFingerContact:
    """One exact cap/finger contact point with raw and Bolt-oriented values."""

    finger: Literal["left", "right"]
    event_type: ContactEventName
    header_index: int
    contact_data_index: int
    contact_data_offset: int
    contact_data_count: int
    actor0_path: str
    actor1_path: str
    collider0_path: str
    collider1_path: str
    raw_position_w_m: Vector3
    raw_normal_w: Vector3
    raw_impulse_w_ns: Vector3
    raw_separation_m: float
    normal_on_bolt_w: Vector3
    impulse_on_bolt_w_ns: Vector3
    force_on_bolt_w_n: Vector3
    normal_compression_n: float


def decode_bolt_finger_contacts(
    headers: Sequence[object],
    contact_data: Sequence[object],
    *,
    dt_s: float,
    path_decoder: Callable[[object], object],
) -> dict[str, tuple[BoltFingerContact, ...]]:
    """Decode FOUND/PERSIST reports for the exact cap and Panda finger actors.

    ``path_decoder`` should be ``PhysicsSchemaTools.intToSdfPath`` in Isaac
    runtime. It is injected so this module remains importable and testable
    without Kit or pxr.
    """
    dt = _finite_number(dt_s, "dt_s")
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")

    records: dict[str, list[BoltFingerContact]] = {"left": [], "right": []}
    finger_by_actor = {path: side for side, path in FINGER_ACTOR_PATHS.items()}

    for header_index, header in enumerate(headers):
        event_name = _event_name(_field(header, "type"))
        if event_name == "CONTACT_LOST":
            continue
        if event_name == "CONTACT_FOUND":
            accepted_event: ContactEventName = "CONTACT_FOUND"
        elif event_name == "CONTACT_PERSIST":
            accepted_event = "CONTACT_PERSIST"
        else:
            raise ValueError(f"unsupported PhysX contact event type: {event_name!r}")

        collider0_path = _decode_path(path_decoder, _field(header, "collider0"), "collider0")
        collider1_path = _decode_path(path_decoder, _field(header, "collider1"), "collider1")
        if collider0_path == CAP_COLLIDER_PATH:
            bolt_collider_index = 0
        elif collider1_path == CAP_COLLIDER_PATH:
            bolt_collider_index = 1
        else:
            continue

        actor0_path = _decode_path(path_decoder, _field(header, "actor0"), "actor0")
        actor1_path = _decode_path(path_decoder, _field(header, "actor1"), "actor1")
        if bolt_collider_index == 0:
            if actor0_path != BOLT_ACTOR_PATH:
                continue
            finger_actor_path = actor1_path
            finger_collider_path = collider1_path
        else:
            if actor1_path != BOLT_ACTOR_PATH:
                continue
            finger_actor_path = actor0_path
            finger_collider_path = collider0_path
        finger = finger_by_actor.get(finger_actor_path)
        if finger is None:
            continue

        offset = _nonnegative_index(_field(header, "contact_data_offset"), "contact_data_offset")
        count = _nonnegative_index(_field(header, "num_contact_data"), "num_contact_data")
        data_length = len(contact_data)
        if offset > data_length or count > data_length - offset:
            raise ValueError(
                "PhysX contact-data slice is out of bounds: "
                f"offset={offset}, count={count}, length={data_length}"
            )

        orientation = 1.0 if bolt_collider_index == 0 else -1.0
        for local_index in range(count):
            contact_index = offset + local_index
            point = contact_data[contact_index]
            position = _finite_vector3(_field(point, "position"), "position")
            raw_normal = _finite_vector3(_field(point, "normal"), "normal")
            raw_impulse = _finite_vector3(_field(point, "impulse"), "impulse")
            separation = _finite_number(_field(point, "separation"), "separation")
            normal_length = hypot(*raw_normal)
            if not isfinite(normal_length) or normal_length == 0.0:
                raise ValueError("contact normal must be finite and non-zero")

            normal_on_bolt = tuple(orientation * value / normal_length for value in raw_normal)
            impulse_on_bolt = tuple(orientation * value for value in raw_impulse)
            force_on_bolt = tuple(value / dt for value in impulse_on_bolt)
            compression = sum(force_on_bolt[i] * normal_on_bolt[i] for i in range(3))
            if any(not isfinite(value) for value in (*normal_on_bolt, *impulse_on_bolt, *force_on_bolt)):
                raise ValueError("derived Bolt contact force is not finite")
            if not isfinite(compression):
                raise ValueError("derived normal compression is not finite")

            records[finger].append(
                BoltFingerContact(
                    finger=finger,
                    event_type=accepted_event,
                    header_index=header_index,
                    contact_data_index=contact_index,
                    contact_data_offset=offset,
                    contact_data_count=count,
                    actor0_path=actor0_path,
                    actor1_path=actor1_path,
                    collider0_path=collider0_path,
                    collider1_path=collider1_path,
                    raw_position_w_m=position,
                    raw_normal_w=raw_normal,
                    raw_impulse_w_ns=raw_impulse,
                    raw_separation_m=separation,
                    normal_on_bolt_w=normal_on_bolt,
                    impulse_on_bolt_w_ns=impulse_on_bolt,
                    force_on_bolt_w_n=force_on_bolt,
                    normal_compression_n=compression,
                )
            )

    return {finger: tuple(items) for finger, items in records.items()}


def _field(value: object, name: str) -> object:
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise ValueError(f"PhysX report is missing required field {name!r}") from exc


def _event_name(value: object) -> str:
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if not isinstance(name, str):
        raise ValueError(f"PhysX event type has no enum name: {value!r}")
    return name


def _decode_path(path_decoder: Callable[[object], object], value: object, name: str) -> str:
    decoded = path_decoder(value)
    path_string = getattr(decoded, "pathString", None)
    path = path_string if isinstance(path_string, str) else str(decoded)
    if not path.startswith("/"):
        raise ValueError(f"decoded {name} is not an absolute USD path: {path!r}")
    return path


def _nonnegative_index(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric data") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite numeric data")
    return result


def _finite_vector3(value: object, name: str) -> Vector3:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain three finite values")
    try:
        components = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if len(components) != 3:
        raise ValueError(f"{name} must contain three finite values")
    result = tuple(_finite_number(component, f"{name}[{i}]") for i, component in enumerate(components))
    return result  # type: ignore[return-value]


__all__ = [
    "BOLT_ACTOR_PATH",
    "CAP_COLLIDER_PATH",
    "FINGER_ACTOR_PATHS",
    "BoltFingerContact",
    "decode_bolt_finger_contacts",
]
