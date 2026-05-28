from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from sequential_submission_infer import materialize_context_from_history, materialize_contexts_from_history_one_pass  # noqa: E402


class FakeRawContext:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.event_count = len(kwargs["events"])

    def to_dict(self):
        return dict(self.__dict__)


class FakeReaderMod:
    SCHEMA_VERSION = "schema"
    BENCHMARK_ID = "benchmark"
    T1 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    T1_PLUS_W = dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)
    T2 = dt.datetime(2026, 1, 10, tzinfo=dt.timezone.utc)
    RawContext = FakeRawContext

    @staticmethod
    def _parse_datetime(value):
        return value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(value)

    @staticmethod
    def _iso(value):
        return value.isoformat()

    @staticmethod
    def _stable_json_sha256(payload):
        def default(obj):
            if isinstance(obj, dt.datetime):
                return obj.isoformat()
            raise TypeError(type(obj).__name__)
        encoded = json.dumps(payload, sort_keys=True, default=default).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


class FakeReader:
    dataset_manifest_id = "manifest"
    canonical_root = "/canonical"
    split = "eval"


class FakeEvent:
    def __init__(self, ts, encoded_id, valid=True):
        self.event_time = ts
        self.fields = {"timestamp_quality_status": "valid" if valid else "invalid"}
        self.encoded_id = encoded_id

    def model_visible_dict(self):
        return {"encoded_id": self.encoded_id, "event_time_unix_s": int(self.event_time.timestamp())}


class FakeHistory:
    user_id = "u"
    dataset_version = "v"

    def __init__(self, events):
        self.events = events


def test_one_pass_context_materialization_matches_single_target_path() -> None:
    base = FakeReaderMod.T1
    history = FakeHistory([
        FakeEvent(base + dt.timedelta(hours=1), 1),
        FakeEvent(base + dt.timedelta(days=2, hours=1), 2),
        FakeEvent(base + dt.timedelta(days=2, hours=2), 3, valid=False),
        FakeEvent(base + dt.timedelta(days=3), 4),
    ])
    targets = [
        {"target_id": "t1", "user_id": "u", "target_ts": base + dt.timedelta(days=2, hours=12), "raw_context_event_count": 2},
        {"target_id": "t2", "user_id": "u", "target_ts": base + dt.timedelta(days=4), "raw_context_event_count": 3},
    ]

    expected = [(target, materialize_context_from_history(FakeReaderMod, FakeReader, history, target)) for target in targets]
    actual = materialize_contexts_from_history_one_pass(FakeReaderMod, FakeReader, history, targets)

    assert [ctx.event_count for _, ctx in actual] == [ctx.event_count for _, ctx in expected]
    assert [ctx.events for _, ctx in actual] == [ctx.events for _, ctx in expected]
    assert [ctx.checksum for _, ctx in actual] == [ctx.checksum for _, ctx in expected]


if __name__ == "__main__":
    test_one_pass_context_materialization_matches_single_target_path()
    print("context materialization tests passed")
