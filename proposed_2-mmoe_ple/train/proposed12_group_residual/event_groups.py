"""Event-group mapping for P12 group residual heads."""

import torch

# 0 is reserved for padding/unknown group.
GROUP_ID_TO_NAME = {
    0: "UNK",
    1: "Ad",
    2: "Browsing",
    3: "Search",
    4: "Purchase",
    5: "Others",
}

# Event type IDs from semantic_next_event_prediction.EVENT_TYPE_DICT.
EVENT_TYPE_TO_GROUP_ID = {
    1: 1,   # NativeClick -> Ad
    2: 1,   # SearchClick -> Ad
    3: 2,   # EdgePageTitle -> Browsing
    6: 2,   # UET -> Browsing
    9: 2,   # UETShoppingView -> Browsing
    4: 3,   # EdgeSearchQuery -> Search
    5: 3,   # OrganicSearchQuery -> Search
    8: 4,   # UETShoppingCart -> Purchase
    10: 4,  # AbandonCart -> Purchase
    11: 4,  # EdgeShoppingCart -> Purchase
    12: 4,  # EdgeShoppingPurchase -> Purchase
    7: 5,   # OutlookSenderDomain -> Others
}


def build_event_type_to_group_tensor(num_event_types: int) -> torch.Tensor:
    """Return tensor mapping event_type_id -> group_id, including id 0."""
    mapping = torch.zeros(num_event_types + 1, dtype=torch.long)
    for event_type_id, group_id in EVENT_TYPE_TO_GROUP_ID.items():
        if event_type_id <= num_event_types:
            mapping[event_type_id] = group_id
    return mapping
