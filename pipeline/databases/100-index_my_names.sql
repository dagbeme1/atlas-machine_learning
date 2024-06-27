-- Import the table dump
SOURCE names.sql;

-- Add a computed column for the first letter of the name
ALTER TABLE names ADD COLUMN first_letter 
CHAR(1) GENERATED ALWAYS 
AS (LEFT(name, 1)) STORED;

-- Create an index on the computed column
CREATE INDEX idx_name_first 
ON names(first_letter)
;
