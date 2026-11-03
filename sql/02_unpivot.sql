-- 30,490 rows x 1,941 columns -> 59.2M long rows, entirely on the SQL engine.
-- COLUMNS(* EXCLUDE ...) avoids listing 1,941 column names.
CREATE OR REPLACE TABLE long_sales AS
UNPIVOT stg_sales
ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
INTO NAME d VALUE sales;
