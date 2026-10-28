import numpy as np

from hub_evaluator import axisymmetric_seat_stable, settle_window_contact


def test_seat_accepts_submillimetre_contact_jitter_without_requiring_zero_spin():
    p = np.zeros((60, 3))
    p[:, 0] = .0001 * np.sin(np.arange(60))
    # Rotation about this installation axis leaves the seating relation unchanged.
    a = np.tile([0., 0., 1.], (60, 1))
    assert axisymmetric_seat_stable(p, a)


def test_seat_rejects_translation_and_axis_wobble():
    p = np.zeros((60, 3));a = np.tile([0., 0., 1.], (60, 1))
    p[-1, 0] = .002
    assert not axisymmetric_seat_stable(p, a)
    p[:] = 0
    a[-1] = [np.sin(np.deg2rad(1)), 0., np.cos(np.deg2rad(1))]
    assert not axisymmetric_seat_stable(p, a)


def test_seat_rejects_missing_nonfinite_or_degenerate_evidence():
    assert not axisymmetric_seat_stable([], [])
    assert not axisymmetric_seat_stable(np.zeros((60, 3)), np.zeros((60, 3)))
    p = np.zeros((60, 3));p[0, 0] = np.nan
    assert not axisymmetric_seat_stable(p, np.tile([0, 0, 1], (60, 1)))


def test_intermittent_support_uses_real_window_and_preserves_penetration_peak():
    contacts = [True, False] * 30
    penetration = [0.] * 60
    assert settle_window_contact(contacts, penetration) == (True, 0.)
    penetration[0] = .002
    assert settle_window_contact(contacts, penetration) == (True, .002)
    assert settle_window_contact([False]*60, penetration) == (False, .002)
    assert settle_window_contact([], []) == (False, None)
