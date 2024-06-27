-- 101-index_name_score.sql
-- 101-drop_existing_index.sql
DROP INDEX idx_name_first_score ON names;
CREATE INDEX idx_name_first_score 
ON names (LEFT(name, 1), score);
