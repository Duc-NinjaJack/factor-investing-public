#!/usr/bin/env python3
"""
Calendar Service (Skeleton)
===========================

Provides unified access to price and holdings anchor dates with simple policies:
- price anchors: first trading day of month from price index
- holdings anchors: fundamentals-aware (placeholder) monthly anchors
- intersection anchors: common dates between the two
- nearest-with-tolerance: nearest available date not exceeding tolerance

This is a minimal skeleton to satisfy instrumentation and policy wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import pandas as pd


@dataclass
class CalendarService:
    price_calendar: pd.DatetimeIndex
    holdings_calendar: pd.DatetimeIndex
    logger: Optional[logging.Logger] = None

    @staticmethod
    def _first_of_month(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if dates.empty:
            return dates
        s = pd.Series(dates)
        return s.groupby(s.dt.to_period('M')).head(1).sort_values().to_numpy()

    @classmethod
    def from_price_series(
        cls,
        price_index: pd.DatetimeIndex,
        holdings_index: Optional[pd.DatetimeIndex] = None,
        logger: Optional[logging.Logger] = None,
    ) -> 'CalendarService':
        price_anchors = pd.DatetimeIndex(cls._first_of_month(price_index))
        holdings_anchors = pd.DatetimeIndex(cls._first_of_month(holdings_index)) if holdings_index is not None else price_anchors
        return cls(price_anchors, holdings_anchors, logger)

    def get_price_anchors(self) -> pd.DatetimeIndex:
        return self.price_calendar

    def get_holdings_anchors(self) -> pd.DatetimeIndex:
        return self.holdings_calendar

    def get_intersection_anchors(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(set(self.price_calendar).intersection(set(self.holdings_calendar))))

    def choose_anchor(self, target_date: pd.Timestamp, policy: str) -> Tuple[str, pd.Timestamp, int]:
        """
        Returns: (anchor_type, anchor_date, delta_days)
        policy examples: 'holdings', 'price', 'intersection', 'nearest:3d'
        """
        policy = (policy or '').strip() or 'nearest:3d'
        if ':' in policy:
            kind, tol = policy.split(':', 1)
            try:
                tolerance_days = int(tol.strip(' dD'))
            except Exception:
                tolerance_days = 3
        else:
            kind = policy
            tolerance_days = 3

        if kind == 'holdings':
            anchors = self.holdings_calendar
        elif kind == 'price':
            anchors = self.price_calendar
        elif kind == 'intersection':
            anchors = self.get_intersection_anchors()
        else:  # nearest
            anchors = pd.DatetimeIndex(sorted(set(self.price_calendar).union(set(self.holdings_calendar))))

        if anchors.empty:
            return kind if kind in {'holdings', 'price', 'intersection'} else 'nearest', target_date, 0

        # Find nearest with bias to past (no look-ahead)
        idx = anchors.searchsorted(target_date, side='right') - 1
        if idx < 0:
            idx = 0
        anchor_date = anchors[idx]
        delta = abs((pd.to_datetime(target_date) - pd.to_datetime(anchor_date)).days)

        anchor_type = kind if kind in {'holdings', 'price', 'intersection'} else 'nearest'
        if anchor_type == 'nearest' and delta > tolerance_days:
            # Respect tolerance; caller may override policy if needed
            if self.logger:
                self.logger.warning("Anchor nearest exceeded tolerance: delta_days=%d > %d", delta, tolerance_days)

        if self.logger:
            self.logger.info("anchor=%s | date=%s | delta_days=%d", anchor_type, pd.to_datetime(anchor_date).date(), delta)
        return anchor_type, pd.to_datetime(anchor_date), int(delta)


