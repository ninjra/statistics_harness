from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules  # type: ignore


@dataclass(frozen=True)
class MiningCaps:
    max_rows: int = 50_000
    max_items: int = 200
    max_itemset_len: int = 6
    max_rules: int = 200


def mine_frequent_itemsets_fpgrowth(one_hot_df: Any, *, min_support: float, caps: MiningCaps) -> Any:
    return fpgrowth(one_hot_df, min_support=float(min_support), use_colnames=True, max_len=int(caps.max_itemset_len))


def mine_frequent_itemsets_apriori(one_hot_df: Any, *, min_support: float, caps: MiningCaps) -> Any:
    return apriori(one_hot_df, min_support=float(min_support), use_colnames=True, max_len=int(caps.max_itemset_len))


def mine_association_rules(freq_itemsets: Any, *, min_confidence: float, min_lift: float, caps: MiningCaps) -> Any:
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=float(min_confidence))
    if hasattr(rules, "__getitem__"):
        try:
            rules = rules[rules["lift"] >= float(min_lift)]
        except Exception:
            pass
    try:
        return rules.head(int(caps.max_rules))
    except Exception:
        return rules

