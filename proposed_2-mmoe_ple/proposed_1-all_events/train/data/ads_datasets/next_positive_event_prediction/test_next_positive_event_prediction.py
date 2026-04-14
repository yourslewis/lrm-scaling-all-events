import pytest
import logging
from .next_positive_event_prediction import TrainIterableDataset
from ..special_tokens import MASK_TOKEN
import types

class DummyFS:
    def glob(self, pattern):
        return []

def make_dataset(mask_prob=0.0, max_sequence_length=5, mask_token=-1):
    return TrainIterableDataset(
        filesystem=DummyFS(),
        parquet_dir='',
        max_sequence_length=max_sequence_length,
        rank=0,
        world_size=1,
        mask_prob=mask_prob,
        mask_token=mask_token,
    )

def make_row(ad_ids, action_ids, timestamps=None, user_id=123):
    if timestamps is None:
        timestamps = list(range(len(ad_ids)))
    # Simulate a namedtuple or pandas row
    row = types.SimpleNamespace(
        user_id=user_id,
        ad_ids=ad_ids,
        action_ids=action_ids,
        timestamps=timestamps,
    )
    return row

def test_process_row_no_mask():
    # Test case 1:
    # input:        [ (10, 0), (20, 1), (30, 0), (40, 1) ]
    # labels:       [   20   ,    40,     40,      0   ]
    dataset = make_dataset(mask_prob=0.0, max_sequence_length=6)
    ad_ids = [10, 20, 30, 40]
    action_ids = [0, 1, 0, 1]
    row = make_row(ad_ids, action_ids)
    out = dataset._process_row(row)
    # Should use original ad_ids as input
    assert out["input"].tolist() == ad_ids + [0, 0]
    assert out["ratings"].tolist() == action_ids + [0, 0]
    assert out["label"].tolist() == [20, 40, 40, 0] + [0, 0]
    assert out["user_id"] == 123
    assert out["length"] == 4

    # Test case 2:
    # input:        [ (4, 0), (7, 0), (3, 0), (4, 1), (5, 0), (7, 0), (7, 1) ]
    # labels:       [   4   ,    4,       4,     7 ,    7,       7,     0     ]  
    dataset = make_dataset(mask_prob=0.0, max_sequence_length=8)
    ad_ids = [4, 7, 3, 4, 5, 7, 7]
    action_ids = [0, 0, 0, 1, 0, 0, 1]
    row = make_row(ad_ids, action_ids)
    out = dataset._process_row(row)
    # Should use original ad_ids as input
    assert out["input"].tolist() == ad_ids + [0]
    assert out["ratings"].tolist() == action_ids + [0]
    assert out["label"].tolist() == [4, 4, 4, 7, 7, 7, 0] + [0]
    assert out["user_id"] == 123
    assert out["length"] == 7

    # Test case 3:
    # input:        [ (4, 0), (7, 0), (3, 0), (4, 1), (5, 0), (7, 0), (7, 1) ]
    # labels:       [   4   ,    4,       4,     7 ,    7,       7,     0     ]  
    dataset = make_dataset(mask_prob=0.0, max_sequence_length=5)
    ad_ids = [4, 7, 3, 4, 5, 7, 7]
    action_ids = [0, 0, 0, 1, 0, 0, 1]
    row = make_row(ad_ids, action_ids)
    out = dataset._process_row(row)
    assert out["input"].tolist() == [3, 4, 5, 7, 7]
    assert out["ratings"].tolist() == [0, 1, 0, 0, 1]
    assert out["label"].tolist() == [4, 7, 7, 7, 0]
    assert out["user_id"] == 123
    assert out["length"] == 5

    logging.info("Test passed for test_process_row_no_mask.")

def test_process_row_with_mask():
    # Test case 1:
    # input:        [ (10, 0), (20, 1), (30, 0), (40, 1) ]
    # masked_input: [   10,      20,      30,     40 ]
    # labels:       [   20   ,    40,     40,      0   ]
    dataset = make_dataset(mask_prob=1.0, max_sequence_length=6)
    ad_ids = [10, 20, 30, 40]
    action_ids = [0, 1, 0, 1]
    row = make_row(ad_ids, action_ids)
    out = dataset._process_row(row)
    assert out["input"].tolist() == ad_ids + [0, 0]
    assert out["ratings"].tolist() == action_ids + [0, 0]
    assert out["label"].tolist() == [20, 40, 40, 0] + [0, 0]
    assert out["user_id"] == 123
    assert out["length"] == 4

    # Test case 2:
    # input:        [ (4, 0), (7, 0), (3, 0), (4, 1), (5, 0), (7, 0), (7, 1) ]
    # masked_input: [   mask,   mask,     3,     4,     5,      mask,   7   ] ]
    # labels:       [   4   ,    4,       4,     7 ,    7,       7,     0     ] 
    dataset = make_dataset(mask_prob=1.0, max_sequence_length=8)
    ad_ids = [4, 7, 3, 4, 5, 7, 7]
    action_ids = [0, 0, 0, 1, 0, 0, 1]
    row = make_row(ad_ids, action_ids)
    out = dataset._process_row(row)
    assert out["input"].tolist() == [-1, -1, 3, 4, 5, -1, 7] + [0]
    assert out["ratings"].tolist() == action_ids + [0]
    assert out["label"].tolist() == [4, 4, 4, 7, 7, 7, 0] + [0]
    assert out["user_id"] == 123
    assert out["length"] == 7

    # Test case 3:
    # input:        [ (4, 0), (7, 0), (3, 0), (4, 1), (5, 0), (7, 0), (7, 1) ]
    # masked_input: [   mask,   mask,     3,     4,     5,      mask,   7   ] ]
    # labels:       [   4   ,    4,       4,     7 ,    7,       7,     0     ] 
    dataset = make_dataset(mask_prob=1.0, max_sequence_length=5)
    ad_ids = [4, 7, 3, 4, 5, 7, 7]
    action_ids = [0, 0, 0, 1, 0, 0, 1]
    row = make_row(ad_ids, action_ids)
    out = dataset._process_row(row)
    assert out["input"].tolist() == [3, 4, 5, -1, 7]
    assert out["ratings"].tolist() == [0,1, 0, 0, 1]
    assert out["label"].tolist() == [4, 7, 7, 7, 0]
    assert out["user_id"] == 123
    assert out["length"] == 5

    logging.info("Test passed for test_process_row_with_mask.")

def test_build_next_positive_event_labels():
    dataset = make_dataset()
    MASK_TOKEN = -1  # Assuming MASK_TOKEN is defined as -1

    # Test case 1:
    # input:        [ (10, 0), (20, 1), (30, 0), (40, 1) ]
    # masked_input: [   10,      20,      30,     40 ]
    # labels:       [   20   ,    40,     40,      0   ]
    ad_ids = [10, 20, 30, 40]
    action_ids = [0, 1, 0, 1]
    masked_ad_ids, out_action_ids, labels = dataset._build_next_positive_event_labels(ad_ids, action_ids, MASK_TOKEN)
    assert labels == [20, 40, 40, 0]
    expected_masked = [10, 20, 30, 40]
    assert masked_ad_ids == expected_masked
    assert out_action_ids == action_ids
    
    # Test case 2:
    # input:        [ (4, 0), (7, 0), (3, 0), (4, 1), (5, 0), (7, 0), (7, 1) ]
    # masked_input: [   mask,   mask,     3,     4,     5,      mask,   7   ] ]
    # labels:       [   4   ,    4,       4,     7 ,    7,       7,     0     ]   
    ad_ids = [4, 7, 3, 4, 5, 7, 7]
    action_ids = [0, 0, 0, 1, 0, 0, 1]
    masked_ad_ids, out_action_ids, labels = dataset._build_next_positive_event_labels(ad_ids, action_ids, MASK_TOKEN)
    assert labels == [4, 4, 4, 7, 7, 7, 0]
    expected_masked = [-1, -1, 3, 4, 5, -1, 7]
    assert masked_ad_ids == expected_masked
    assert out_action_ids == action_ids

    logging.info("Test passed for build_next_positive_event_labels.")
