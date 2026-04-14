import pytest
import types
import logging
from next_event_prediction import TrainIterableDataset

class DummyFS:
    def glob(self, pattern):
        return []

def make_dataset(max_sequence_length=5):
    return TrainIterableDataset(
        filesystem=DummyFS(),  # Not used in test
        parquet_dir='',
        max_sequence_length=max_sequence_length,
        rank=0,
        world_size=1,
    )

def make_row(ad_ids, timestamps=None, user_id=123):
    if timestamps is None:
        timestamps = list(range(len(ad_ids)))
    row = types.SimpleNamespace(
        user_id=user_id,
        ad_ids=ad_ids,
        timestamps=timestamps,
    )
    return row

def test_process_row():
    # Test case 1:
    # ad_ids:        [ 10, 20, 30, 40 ]
    # input:         [ 10, 20, 30 ] 
    # labels:        [ 20  30, 40 ]
    dataset = make_dataset(max_sequence_length=6)
    ad_ids = [10, 20, 30, 40]
    timestamps = [100, 200, 300, 400]
    row = make_row(ad_ids, timestamps)
    out = dataset._process_row(row)
    assert out["input"].tolist() == [10, 20, 30] + [0, 0, 0]
    assert out["label"].tolist() == [20, 30, 40] + [0, 0, 0]
    assert out["timestamps"].tolist() == [100, 200, 300] + [0, 0, 0]
    assert out["user_id"] == 123
    assert out["length"] == 3
    assert out["ratings"] == -1  # Placeholder

    # Test case 2:
    # ad_ids:        [ 4, 7, 3, 4, 5, 7, 7 ]
    # input:         [ 4, 7, 3, 4, 5, 7 ] 
    # labels:        [ 7, 3, 4, 5, 7, 7 ]
    dataset = make_dataset(max_sequence_length=8)
    ad_ids = [ 4, 7, 3, 4, 5, 7, 7 ]
    timestamps = [100, 200, 300, 400, 401, 402, 403]
    row = make_row(ad_ids, timestamps)
    out = dataset._process_row(row)
    assert out["input"].tolist() == [ 4, 7, 3, 4, 5, 7 ] + [0, 0]
    assert out["label"].tolist() == [ 7, 3, 4, 5, 7, 7 ] + [0, 0]
    assert out["timestamps"].tolist() == [100, 200, 300, 400, 401, 402] + [ 0, 0]
    assert out["user_id"] == 123
    assert out["length"] == 6
    assert out["ratings"] == -1  # Placeholder

    # Test case 3:
    # ad_ids:        [ 4, 7, 3, 4, 5, 7, 7 ]
    # input:         [ 4, 7, 3, 4, 5, 7 ] 
    # labels:        [ 7, 3, 4, 5, 7, 7 ]
    dataset = make_dataset(max_sequence_length=5)
    ad_ids = [ 4, 7, 3, 4, 5, 7, 7 ]
    timestamps = [100, 200, 300, 400, 401, 402, 403]
    row = make_row(ad_ids, timestamps)
    out = dataset._process_row(row)
    assert out["input"].tolist() == [ 7, 3, 4, 5, 7 ] 
    assert out["label"].tolist() == [ 3, 4, 5, 7, 7 ] 
    assert out["timestamps"].tolist() == [200, 300, 400, 401, 402]
    assert out["length"] == 5
    assert out["ratings"] == -1  # Placeholder
    logging.info("Test passed for process_row.")
