-- Type the raw competition files. `read_csv_auto` infers d_* columns as BIGINT.
CREATE OR REPLACE TABLE stg_calendar AS
SELECT
    CAST(date AS DATE) AS date,
    CAST(wm_yr_wk AS INTEGER) AS wm_yr_wk,
    weekday,
    CAST(wday AS INTEGER) AS wday,
    CAST(month AS INTEGER) AS month,
    CAST(year AS INTEGER) AS year,
    d,
    event_name_1,
    event_type_1,
    event_name_2,
    event_type_2,
    CAST(snap_CA AS INTEGER) AS snap_CA,
    CAST(snap_TX AS INTEGER) AS snap_TX,
    CAST(snap_WI AS INTEGER) AS snap_WI
FROM read_csv_auto('{calendar_path}', header = true);

CREATE OR REPLACE TABLE stg_sales AS
SELECT * FROM read_csv_auto('{sales_path}', header = true);

CREATE OR REPLACE TABLE stg_prices AS
SELECT
    store_id,
    item_id,
    CAST(wm_yr_wk AS INTEGER) AS wm_yr_wk,
    CAST(sell_price AS DOUBLE) AS sell_price
FROM read_csv_auto('{prices_path}', header = true);
