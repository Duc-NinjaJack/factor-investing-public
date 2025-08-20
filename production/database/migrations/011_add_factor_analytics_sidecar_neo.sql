-- Agent Neo's Exact Sidecar Schema for Factor-Level Analytics
-- DO NOT MODIFY EXISTING factor_scores_qvm TABLE
-- This creates SEPARATE analytics tables for individual factor storage

-- 1. Factor dimension table (metadata for 19 individual factors)
CREATE TABLE IF NOT EXISTS dim_factor (
  factor_id SMALLINT PRIMARY KEY,
  factor_code VARCHAR(32) UNIQUE NOT NULL,          -- e.g., 'roae', 'f_score', 'mom_12m'
  pillar ENUM('Quality','Value','Momentum','Defensive') NOT NULL,
  description VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 2. Raw factor values (tall, normalized - stores RAW values, NOT z-scores)
CREATE TABLE IF NOT EXISTS factor_signals_raw (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  ticker VARCHAR(10) NOT NULL,
  date DATE NOT NULL,
  strategy_version VARCHAR(50) NOT NULL,
  factor_id SMALLINT NOT NULL,              -- FK to dim_factor
  raw_value DECIMAL(20,10) NULL,            -- Raw factor value BEFORE z-score normalization
  calculation_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  
  -- Agent Neo's exact requirement: idempotent UPSERT key
  UNIQUE KEY uq_tdsv (ticker, date, strategy_version, factor_id),
  
  -- Performance indexes for analytics queries
  INDEX idx_date_factor (date, factor_id),
  INDEX idx_ticker_date (ticker, date),
  INDEX idx_version_date (strategy_version, date),
  INDEX idx_pillar_date (factor_id, date),
  
  -- Agent Neo Fix #6: Add composite index for version-factor-date lookups
  INDEX idx_ver_factor_date (strategy_version, factor_id, date)
  
  -- Note: Foreign key constraint removed due to MySQL partitioning limitation
  -- FOREIGN KEY (factor_id) REFERENCES dim_factor(factor_id)
) ENGINE=InnoDB
PARTITION BY RANGE (TO_DAYS(date)) (
  PARTITION p2016 VALUES LESS THAN (TO_DAYS('2017-01-01')),
  PARTITION p2017 VALUES LESS THAN (TO_DAYS('2018-01-01')),
  PARTITION p2018 VALUES LESS THAN (TO_DAYS('2019-01-01')),
  PARTITION p2019 VALUES LESS THAN (TO_DAYS('2020-01-01')),
  PARTITION p2020 VALUES LESS THAN (TO_DAYS('2021-01-01')),
  PARTITION p2021 VALUES LESS THAN (TO_DAYS('2022-01-01')),
  PARTITION p2022 VALUES LESS THAN (TO_DAYS('2023-01-01')),
  PARTITION p2023 VALUES LESS THAN (TO_DAYS('2024-01-01')),
  PARTITION p2024 VALUES LESS THAN (TO_DAYS('2025-01-01')),
  PARTITION pmax  VALUES LESS THAN MAXVALUE
);

-- 3. Normalization statistics for exact z-score reconstruction
-- Agent Neo's requirement: per-date, per-sector, per-factor normalization stats
CREATE TABLE IF NOT EXISTS factor_norm_stats (
  date DATE NOT NULL,
  sector VARCHAR(64) NOT NULL,
  strategy_version VARCHAR(50) NOT NULL,
  factor_id SMALLINT NOT NULL,
  mean_value DECIMAL(20,10) NOT NULL,        -- For z-score: (raw - mean) / std
  std_value  DECIMAL(20,10) NOT NULL,        -- For z-score: (raw - mean) / std
  universe_size INT NOT NULL,                -- Number of stocks in calculation
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- Agent Neo's exact requirement: composite primary key
  PRIMARY KEY (date, sector, strategy_version, factor_id),
  INDEX idx_date_version (date, strategy_version)
  -- Note: Foreign key constraint removed due to MySQL partitioning limitation  
  -- FOREIGN KEY (factor_id) REFERENCES dim_factor(factor_id)
) ENGINE=InnoDB;

-- 4. Populate factor dimension with QVM v2.1.1 flat methodology factors
-- Agent Neo's requirement: 19-factor complete mapping
INSERT IGNORE INTO dim_factor (factor_id, factor_code, pillar, description) VALUES
-- Quality Pillar (Q35) - Agent Neo's exact factor list
(1, 'roae', 'Quality', 'Return on Average Equity (TTM)'),
(2, 'gross_margin', 'Quality', 'Gross Margin (TTM)'),
(3, 'f_score', 'Quality', 'Piotroski F-Score (vectorized, sector-specific)'),
(4, 'debt_equity', 'Quality', 'Debt to Equity Ratio'),

-- Value Pillar (V30)
(5, 'earnings_yield', 'Value', 'Earnings Yield (E/P ratio)'),
(6, 'book_value', 'Value', 'Price to Book Value'),
(7, 'fcf_yield', 'Value', 'Free Cash Flow Yield (TTM)'),
(8, 'sales_multiple', 'Value', 'Price to Sales Ratio'),

-- Momentum Pillar (M20)
(9, 'mom_12m', 'Momentum', '12-month momentum (skip 1 month)'),
(10, 'mom_6m', 'Momentum', '6-month momentum'),
(11, 'mom_3m', 'Momentum', '3-month momentum'),

-- Defensive Pillar (D15)
(12, 'inv_volatility_63d', 'Defensive', 'Inverse Volatility (63-day)'),
(13, 'beta', 'Defensive', 'Market Beta'),
(14, 'low_vol_score', 'Defensive', 'Low Volatility Score'),

-- Sector-specific factors (Banking)
(15, 'nim', 'Quality', 'Net Interest Margin (Banking-specific)'),
(16, 'cost_income', 'Quality', 'Cost-to-Income Ratio (Banking-specific)'),

-- Sector-specific factors (Securities)
(17, 'operating_margin_sec', 'Quality', 'Operating Margin (Securities-specific)'),
(18, 'cost_ratio_sec', 'Quality', 'Cost Ratio (Securities-specific)'),

-- Insurance factors (if implemented)
(19, 'insurance_quality', 'Quality', 'Insurance-specific Quality Score');

-- 5. Agent Neo's requirement: Create view for on-demand z-scores (no z persisted)
CREATE OR REPLACE VIEW v_factor_z AS
SELECT r.ticker, r.date, d.factor_code, n.sector,
       (r.raw_value - n.mean_value) / NULLIF(n.std_value,0) AS z_score,
       r.strategy_version, d.pillar
FROM factor_signals_raw r
JOIN dim_factor d ON d.factor_id = r.factor_id
JOIN factor_norm_stats n
  ON n.date = r.date AND n.factor_id = r.factor_id
 AND n.strategy_version = r.strategy_version
JOIN master_info m ON m.ticker = r.ticker AND n.sector = m.sector;

-- Verification queries
SELECT 'Sidecar schema deployed successfully' AS status;
SELECT pillar, COUNT(*) as factor_count FROM dim_factor GROUP BY pillar;
SELECT 'Ready for wrapper implementation' AS next_step;