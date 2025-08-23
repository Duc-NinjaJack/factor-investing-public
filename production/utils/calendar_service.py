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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd


@dataclass
class CalendarService:
    price_calendar: pd.DatetimeIndex
    holdings_calendar: pd.DatetimeIndex
    logger: Optional[logging.Logger] = None
    default_policy: str = 'nearest:3d'
    allow_nearest_override: bool = False
    _anchor_cache: Dict[Tuple[pd.Timestamp, str], Tuple[str, pd.Timestamp, int]] = field(default_factory=dict, init=False, repr=False)
    _policy_logged: bool = field(default=False, init=False, repr=False)

    @staticmethod
    def _first_of_month(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if dates.empty:
            return dates
        s = pd.Series(dates)
        return s.groupby(s.dt.to_period('M')).head(1).sort_values().to_numpy()

    @staticmethod
    def _quarter_end_for_month(date_like: pd.Timestamp) -> pd.Timestamp:
        # Map month to its quarter end month
        m = int(pd.to_datetime(date_like).month)
        q_end_month = ((m - 1) // 3 + 1) * 3
        # Use month end for the quarter end month
        q_end = pd.Timestamp(pd.to_datetime(date_like).year, q_end_month, 1) + pd.offsets.MonthEnd(0)
        return q_end.normalize()

    @classmethod
    def _fundamentals_aware_anchors(
        cls,
        price_index: pd.DatetimeIndex,
        reporting_lag_days: int,
    ) -> pd.DatetimeIndex:
        """
        Build holdings anchors using fundamentals reporting lag policy:
        - For each calendar month in the price series span, take the corresponding quarter end
        - Add reporting_lag_days to obtain the publish date
        - Anchor to the first trading day on or after the publish date
        """
        if price_index.empty:
            return price_index
        px = pd.DatetimeIndex(pd.to_datetime(price_index)).sort_values()
        # Monthly grid across available range
        months = pd.period_range(start=px.min().to_period('M'), end=px.max().to_period('M'), freq='M')
        anchors: List[pd.Timestamp] = []
        for p in months:
            month_start = p.to_timestamp(how='start')
            q_end = cls._quarter_end_for_month(month_start)
            publish = q_end + pd.Timedelta(days=int(reporting_lag_days))
            # First trading day on or after publish
            pos = px.searchsorted(publish, side='left')
            if pos >= len(px):
                # If beyond range, use last available trading day
                pos = len(px) - 1
            anchors.append(px[pos].normalize())
        return pd.DatetimeIndex(sorted(set(anchors)))

    @classmethod
    def from_price_series(
        cls,
        price_index: pd.DatetimeIndex,
        holdings_index: Optional[pd.DatetimeIndex] = None,
        logger: Optional[logging.Logger] = None,
        default_policy: str = 'nearest:3d',
        allow_nearest_override: bool = False,
        reporting_lag_days: Optional[int] = None,
    ) -> 'CalendarService':
        price_anchors = pd.DatetimeIndex(cls._first_of_month(price_index))
        if holdings_index is not None:
            holdings_anchors = pd.DatetimeIndex(cls._first_of_month(holdings_index))
        elif isinstance(reporting_lag_days, (int, float)):
            # Fundamentals-aware holdings anchors using reporting lag
            try:
                holdings_anchors = cls._fundamentals_aware_anchors(price_index, int(reporting_lag_days))
            except Exception:
                holdings_anchors = price_anchors
        else:
            holdings_anchors = price_anchors
        svc = cls(price_anchors, holdings_anchors, logger, default_policy, allow_nearest_override)
        if logger:
            logger.info("calendar_policy=%s | tolerance_enforced=%s", default_policy, not allow_nearest_override)
        return svc

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
        policy = (policy or '').strip() or (self.default_policy or 'nearest:3d')
        # Log policy once per service lifetime
        if self.logger and not self._policy_logged:
            self.logger.info("calendar_policy=%s | tolerance_enforced=%s", policy, not self.allow_nearest_override)
            self._policy_logged = True

        # Normalize and cache key
        tgt = pd.to_datetime(target_date).normalize()
        cache_key = (tgt, policy)
        if cache_key in self._anchor_cache:
            return self._anchor_cache[cache_key]
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
            result = (kind if kind in {'holdings', 'price', 'intersection'} else 'nearest', tgt, 0)
            self._anchor_cache[cache_key] = result
            return result

        # Find nearest with bias to past (no look-ahead)
        idx = anchors.searchsorted(target_date, side='right') - 1
        if idx < 0:
            idx = 0
        anchor_date = anchors[idx]
        delta = abs((pd.to_datetime(target_date) - pd.to_datetime(anchor_date)).days)

        anchor_type = kind if kind in {'holdings', 'price', 'intersection'} else 'nearest'
        if anchor_type == 'nearest' and delta > tolerance_days:
            # Enforce tolerance unless explicitly allowed to override
            if not self.allow_nearest_override:
                if self.logger:
                    self.logger.error("Nearest anchor exceeded tolerance and enforcement is active: delta_days=%d > %d", delta, tolerance_days)
                raise ValueError(f"Nearest anchor exceeded tolerance: delta_days={delta} > {tolerance_days}")
            else:
                if self.logger:
                    self.logger.warning("Nearest anchor exceeded tolerance but override allowed: delta_days=%d > %d", delta, tolerance_days)

        if self.logger:
            self.logger.info("anchor=%s | date=%s | delta_days=%d", anchor_type, pd.to_datetime(anchor_date).date(), delta)
        result = (anchor_type, pd.to_datetime(anchor_date), int(delta))
        self._anchor_cache[cache_key] = result
        return result


