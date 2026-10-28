import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaac_runtime'))
from runtime.bolt_harness import measurement


def test_fixture_reports_include_each_collider_not_only_first(monkeypatch):
    calls = []
    def read(view, *, dt_s, sensor_index, filter_index):
        calls.append((dt_s, sensor_index, filter_index))
        return () if filter_index == 0 else ('actual_support',)
    monkeypatch.setattr(measurement, 'read_filtered_contact_reports', read)
    assert measurement.read_all_filtered_contact_reports(
        SimpleNamespace(filter_count=2), dt_s=1/120, sensor_index=0) == ('actual_support',)
    assert calls == [(1/120, 0, 0), (1/120, 0, 1)]


def test_empty_fixture_filters_are_not_treated_as_no_contact():
    with pytest.raises(ValueError, match='no resolved filters'):
        measurement.read_all_filtered_contact_reports(
            SimpleNamespace(filter_count=0), dt_s=1/120, sensor_index=0)
