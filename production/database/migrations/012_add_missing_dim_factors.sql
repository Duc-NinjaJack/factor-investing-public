-- Add missing factor codes required by analytics sidecar persistence
-- Safe to run multiple times due to INSERT IGNORE

INSERT IGNORE INTO dim_factor (factor_id, factor_code, pillar, description) VALUES
  (20, 'roaa', 'Quality', 'Return on Average Assets (TTM) - Banks'),
  (21, 'net_profit_margin', 'Quality', 'Net Profit Margin (TTM) - Non-financials'),
  (22, 'operating_margin', 'Quality', 'Operating Margin (TTM) - Non-financials'),
  (23, 'ebitda_margin', 'Quality', 'EBITDA Margin (TTM) - Non-financials'),
  (24, 'ev_ebitda', 'Value', 'Enterprise Value to EBITDA (inverted to EBITDA/EV for raw)');

-- Optional verification
SELECT factor_id, factor_code FROM dim_factor WHERE factor_id BETWEEN 20 AND 24 ORDER BY factor_id;

