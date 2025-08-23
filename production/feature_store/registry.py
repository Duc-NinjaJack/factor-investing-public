#!/usr/bin/env python3
"""
Schema Registry and Backfill Management
======================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional


@dataclass
class SchemaRegistry:
    schemas: Dict[str, Dict] = field(default_factory=dict)

    def register(self, stage: str, schema: Dict) -> None:
        self.schemas[stage] = dict(schema)

    def get(self, stage: str) -> Dict:
        return self.schemas.get(stage, {})


@dataclass
class BackfillManager:
    available_dates: List[str]
    list_existing_fn: Callable[[str], List[str]]  # stage -> list of date strings present
    run_stage_fn: Callable[[str, str], None]      # (stage, date_str) -> materialize

    def backfill_missing(self, stage: str, dates: Optional[List[str]] = None) -> List[str]:
        target_dates = dates or self.available_dates
        existing = set(self.list_existing_fn(stage))
        missing = [d for d in target_dates if d not in existing]
        for d in missing:
            self.run_stage_fn(stage, d)
        return missing


