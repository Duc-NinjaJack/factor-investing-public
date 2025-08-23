#!/usr/bin/env python3
"""
Feature Store I/O
=================

- Partitioned Parquet storage with immutable metadata
- Stage layout: artifacts/feature_store/{stage}/date=YYYY-MM-DD/part.parquet
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from production.utils.parallel import materialize_dataframe


@dataclass
class FeatureStoreIO:
    base_dir: Path = Path('artifacts/feature_store')

    def _stage_dir(self, stage: str) -> Path:
        return Path(self.base_dir) / stage

    def _partition_dir(self, stage: str, date_str: str) -> Path:
        return self._stage_dir(stage) / f"date={date_str}"

    def _metadata_path(self, stage: str, date_str: str) -> Path:
        return self._partition_dir(stage, date_str) / 'metadata.json'

    def _data_path(self, stage: str, date_str: str) -> Path:
        return self._partition_dir(stage, date_str) / 'part.parquet'

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        # Deterministic hash across runs for identical content
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        return hashlib.sha256(csv_bytes).hexdigest()

    def write_partition(
        self,
        stage: str,
        date_str: str,
        df: pd.DataFrame,
        universe_size: Optional[int] = None,
        extra_metadata: Optional[Dict] = None,
        logger: Optional[object] = None,
    ) -> str:
        part_dir = self._partition_dir(stage, date_str)
        part_dir.mkdir(parents=True, exist_ok=True)

        data_path = self._data_path(stage, date_str)
        meta_path = self._metadata_path(stage, date_str)

        content_hash = self._hash_dataframe(df)
        metadata = {
            'stage': stage,
            'date': date_str,
            'num_rows': int(len(df)),
            'num_cols': int(len(df.columns)),
            'universe_size': int(universe_size) if universe_size is not None else None,
            'content_hash': content_hash,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        # Immutability: if exists, verify identical hash
        if meta_path.exists() and data_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
                if existing.get('content_hash') != content_hash:
                    raise FileExistsError(f"Partition already exists with different content: {stage} {date_str}")
                # Return existing path
                return str(data_path.resolve())
            except Exception:
                raise

        # Materialize dataframe to Parquet
        written_path = materialize_dataframe(df, str(data_path))

        # Write metadata
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        if logger:
            try:
                logger.info(
                    "feature_store_write stage=%s date=%s rows=%d cols=%d",
                    stage, date_str, len(df), len(df.columns)
                )
            except Exception:
                pass
        return written_path

    def read_partition(self, stage: str, date_str: str) -> pd.DataFrame:
        data_path = self._data_path(stage, date_str)
        if not data_path.exists():
            raise FileNotFoundError(f"Partition not found: {stage} {date_str}")
        return pd.read_parquet(data_path)


