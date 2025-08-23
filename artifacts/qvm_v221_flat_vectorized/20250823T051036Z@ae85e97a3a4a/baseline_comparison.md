# Baseline Comparison (Sequential vs Parallel)

- Sequential run: `20250823T050030Z@ae85e97a3a4a`  (jobs=1, wall_clock_ms=540771.11)
- Parallel run: `20250823T051036Z@ae85e97a3a4a`    (jobs=4, wall_clock_ms=163257.4)

## Equivalence Check: FAILED

Mismatched holdings on 30 date(s). Showing first 5:

- 2018-03-16:
  - seq: ['ASM', 'BWE', 'CII', 'DHG', 'DPM', 'DRC', 'HAR', 'HSG', 'IDI', 'KSB', 'MSN', 'NKG', 'NLG', 'PDR', 'PHR', 'PVD', 'PVS', 'SBT', 'SHN', 'TCH']
  - par: ['CVT', 'DCM', 'DXG', 'FPT', 'GMD', 'HAG', 'ITA', 'KBC', 'KSB', 'MSN', 'MWG', 'NKG', 'NT2', 'NVL', 'PAN', 'QCG', 'SAB', 'VCG', 'VIC', 'VJC']
- 2018-06-14:
  - seq: ['ASM', 'BMP', 'CII', 'DHG', 'DPM', 'DS3', 'HSG', 'IDI', 'KSB', 'MSN', 'NLG', 'PDR', 'PHR', 'PNJ', 'PVD', 'PVS', 'SBT', 'TCH', 'VNM', 'VRE']
  - par: ['CEO', 'CVT', 'DGW', 'DPM', 'DST', 'DXG', 'FPT', 'GMD', 'HAG', 'KBC', 'KSB', 'MSN', 'MWG', 'NT2', 'NVL', 'QCG', 'SAB', 'VCG', 'VIC', 'VJC']
- 2018-09-13:
  - seq: ['AAA', 'ASM', 'CII', 'CVT', 'DGW', 'DHG', 'HSG', 'IDI', 'MSN', 'MWG', 'NLG', 'PDR', 'PNJ', 'PVD', 'PVS', 'SAB', 'SBT', 'TCH', 'VNM', 'VRE']
  - par: ['CEO', 'CTD', 'CVT', 'DGW', 'DIG', 'DXG', 'FPT', 'GMD', 'HAG', 'KBC', 'MSN', 'MWG', 'NVL', 'POW', 'QCG', 'SAB', 'VCG', 'VIC', 'VJC', 'VRC']
- 2018-12-14:
  - seq: ['ASM', 'CII', 'CTI', 'DPM', 'HSG', 'IDI', 'KSB', 'MSN', 'MWG', 'NLG', 'PDR', 'PHR', 'PNJ', 'PVD', 'PVS', 'SAB', 'SBT', 'TCH', 'VNM', 'VRE']
  - par: ['CEO', 'DGW', 'DPM', 'DXG', 'FPT', 'GMD', 'HAG', 'KBC', 'KSB', 'MSN', 'MWG', 'NVL', 'QCG', 'SAB', 'TNG', 'VCG', 'VFG', 'VIC', 'VJC', 'VNG']
- 2019-03-18:
  - seq: ['ASM', 'BWE', 'CII', 'DHG', 'DPM', 'HDG', 'HSG', 'KSB', 'LCG', 'MSN', 'NLG', 'PDR', 'PHR', 'PNJ', 'PPC', 'PVD', 'PVS', 'SBT', 'TCH', 'VTP']
  - par: ['CEO', 'DPM', 'DXG', 'FPT', 'GMD', 'HAG', 'HDG', 'ITA', 'KBC', 'KSB', 'MSN', 'MWG', 'NVL', 'PPC', 'SAB', 'TNG', 'VCG', 'VIC', 'VJC', 'VRC']
## Timings (sum of per-date elapsed ms)

- elapsed_ms_universe: seq=37676.3, par=53855.7, delta=16179.4 (-42.9% vs seq)
- elapsed_ms_factors: seq=496185.8, par=522681.3, delta=26495.5 (-5.3% vs seq)
- elapsed_ms_quality: seq=1555.3, par=1396.7, delta=-158.7 (+10.2% vs seq)
- elapsed_ms_value: seq=1910.0, par=1841.1, delta=-68.9 (+3.6% vs seq)
- elapsed_ms_momentum: seq=16728.8, par=16871.2, delta=142.5 (-0.9% vs seq)
- elapsed_ms_lowvol: seq=4559.5, par=6996.4, delta=2436.8 (-53.4% vs seq)
- elapsed_ms_fscore: seq=460944.2, par=485681.8, delta=24737.6 (-5.4% vs seq)
- elapsed_ms_fcf: seq=7001.3, par=6698.1, delta=-303.2 (+4.3% vs seq)

## SQL Counters (sum of per-date deltas)

- sql_queries: seq=4236, par=4239
- sql_rows: seq=1652523, par=1653165

## Anomalies

- Holdings mismatch; parallel equivalence not established
