import pytest
import logging
import types
from conditional_next_event_prediction import TrainIterableDataset

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

def make_row(ad_ids, action_ids, timestamps=None, user_id=123):
    if timestamps is None:
        timestamps = list(range(len(ad_ids)))
    row = types.SimpleNamespace(
        user_id=user_id,
        ad_ids=ad_ids,
        action_ids=action_ids,
        timestamps=timestamps,
    )
    return row

def test_build_conditional_next_event_labels():
    dataset = make_dataset()

    # Test case 1:
    # input:        [ (10, 0), (20, 1), (30, 0), (40, 1) ]
    # labels:       [   20   ,    0,     40,      0   ]    
    ad_ids = [10, 20, 30, 40]
    action_ids = [0, 1, 0, 1]
    labels = dataset._build_conditional_next_event_labels(ad_ids, action_ids)
    assert labels == [20, 0, 40, 0] 

    # Test case 2:
    # input:        [ (4, 0), (7, 0), (3, 0), (4, 1), (5, 0), (7, 0), (7, 1) ]
    # labels:       [   0   ,    0,     4,      0,     0,       7,      0   ]    
    ad_ids = [4, 7, 3, 4, 5, 7, 7]
    action_ids = [0, 0, 0, 1, 0, 0, 1]
    labels = dataset._build_conditional_next_event_labels(ad_ids, action_ids)
    assert labels == [0, 0, 4, 0, 0, 7, 0] 

    logging.info("Test passed for build_next_positive_event_labels.")


def test_process_row():
    # Test case 1:
    # input:        [ (10, 0), (20, 1), (30, 0), (40, 1) ]
    # labels:       [   20   ,    0,     40,      0   ]
    dataset = make_dataset(max_sequence_length=6)
    ad_ids = [10, 20, 30, 40]
    action_ids = [0, 1, 0, 1]
    row = make_row(ad_ids, action_ids)
    out = dataset._process_row(row)
    assert out["input"].tolist() == ad_ids + [0, 0]
    assert out["ratings"].tolist() == action_ids + [0, 0]
    assert out["label"].tolist() == [20, 0, 40, 0] + [0, 0]
    assert out["user_id"] == 123
    assert out["length"] == 4

    # Test case 2:
    # input:        [ (4, 0), (7, 0), (3, 0), (4, 1), (5, 0), (7, 0), (7, 1) ]
    # labels:       [   0   ,    0,      4,     0 ,    0,       7,     0     ]  
    dataset = make_dataset(max_sequence_length=8)
    ad_ids = [4, 7, 3, 4, 5, 7, 7]
    action_ids = [0, 0, 0, 1, 0, 0, 1]
    row = make_row(ad_ids, action_ids)
    out = dataset._process_row(row)
    assert out["input"].tolist() == ad_ids + [0]
    assert out["ratings"].tolist() == action_ids + [0]
    assert out["label"].tolist() == [0, 0, 4, 0, 0, 7, 0] + [0]
    assert out["user_id"] == 123
    assert out["length"] == 7

    # Test case 3:
    # input:        [ (4, 0), (7, 0), (3, 0), (4, 1), (5, 0), (7, 0), (7, 1) ]
    # labels:       [   0   ,    0,      4,     0 ,    0,       7,     0     ]  
    dataset = make_dataset(max_sequence_length=5)
    ad_ids = [4, 7, 3, 4, 5, 7, 7]
    action_ids = [0, 0, 0, 1, 0, 0, 1]
    row = make_row(ad_ids, action_ids)
    out = dataset._process_row(row)
    assert out["input"].tolist() == [3, 4, 5, 7, 7]
    assert out["ratings"].tolist() == [0, 1, 0, 0, 1]
    assert out["label"].tolist() == [4, 0, 0, 7, 0]
    assert out["user_id"] == 123
    assert out["length"] == 5

    logging.info("Test passed for process_row.")