-- Canonical research view: stable factor codes and orientations
-- Maps ev_ebitda → ebitda_to_ev and low_vol_score → low_volatility, book_value → book_to_price, sales_multiple → sales_to_price

CREATE OR REPLACE VIEW v_factor_signals_raw_canonical AS
SELECT
  r.ticker,
  r.date,
  r.strategy_version,
  CASE
    WHEN d.factor_code = 'ev_ebitda' THEN 'ebitda_to_ev'
    WHEN d.factor_code = 'low_vol_score' THEN 'low_volatility'
    WHEN d.factor_code = 'book_value' THEN 'book_to_price'
    WHEN d.factor_code = 'sales_multiple' THEN 'sales_to_price'
    ELSE d.factor_code
  END AS factor_code,
  CASE
    WHEN d.factor_code = 'ev_ebitda' THEN NULLIF(1.0 / NULLIF(r.raw_value,0), 0)
    ELSE r.raw_value
  END AS raw_value
FROM factor_signals_raw r
JOIN dim_factor d USING(factor_id);


