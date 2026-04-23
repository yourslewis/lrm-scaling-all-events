"""Canonical event type taxonomy shared by training and evaluation code."""

EVENT_TYPE_DICT = {
    "UNK": 0,
    "NativeClick": 1,
    "SearchClick": 2,
    "EdgePageTitle": 3,
    "EdgeSearchQuery": 4,
    "OrganicSearchQuery": 5,
    "UET": 6,
    "OutlookSenderDomain": 7,
    "UETShoppingCart": 8,
    "UETShoppingView": 9,
    "AbandonCart": 10,
    "EdgeShoppingCart": 11,
    "EdgeShoppingPurchase": 12,
    "ChromePageTitle": 13,
    "MSN": 14,
}

EVENT_TYPE_NAMES = {event_type_id: name for name, event_type_id in EVENT_TYPE_DICT.items()}

GROUP_MAP = {
    1: "Ad",
    2: "Ad",
    3: "Browsing",
    6: "Browsing",
    9: "Browsing",
    13: "Browsing",
    14: "Browsing",
    4: "Search",
    5: "Search",
    8: "Purchase",
    10: "Purchase",
    11: "Purchase",
    12: "Purchase",
    7: "Others",
}

DOMAIN_MAP = {
    1: 0,
    2: 0,
    3: 1,
    6: 1,
    9: 1,
    13: 1,
    14: 1,
    4: 2,
    5: 2,
    8: 3,
    10: 3,
    11: 3,
    12: 3,
    7: 4,
}
