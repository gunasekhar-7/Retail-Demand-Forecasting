-- Final grain: one row per (item, store, day). Scope is injected by the CLI from config.
CREATE OR REPLACE TABLE panel AS
SELECT
    s.item_id || '_' || s.store_id AS series_id,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    c.date,
    CAST(s.sales AS INTEGER) AS sales,
    CAST(p.sell_price AS DOUBLE) AS sell_price,
    c.event_name_1,
    c.event_type_1,
    c.event_name_2,
    c.event_type_2,
    CASE s.state_id
        WHEN 'CA' THEN c.snap_CA
        WHEN 'TX' THEN c.snap_TX
        WHEN 'WI' THEN c.snap_WI
        ELSE 0
    END AS snap,
    c.snap_CA,
    c.snap_TX,
    c.snap_WI,
    c.wday,
    c.weekday,
    c.month,
    c.year,
    CAST(substr(s.d, 3) AS INTEGER) AS d_index
FROM long_sales s
JOIN stg_calendar c ON c.d = s.d
LEFT JOIN stg_prices p
    ON p.store_id = s.store_id
   AND p.item_id = s.item_id
   AND p.wm_yr_wk = c.wm_yr_wk
WHERE s.store_id IN ({stores})
  AND s.dept_id IN ({depts})
ORDER BY series_id, date;
