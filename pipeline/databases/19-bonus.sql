-- creates a stored procedure AddBonus that adds a new correction for a student.
DELIMITER $$

CREATE PROCEDURE AddBonus (
    IN user_id INT,
    IN project_name VARCHAR(255),
    IN score INT
)
BEGIN
    DECLARE project_id INT;

    -- Attempt to get the project_id of the project_name
    SELECT id INTO project_id
    FROM projects
    WHERE name = project_name;

    -- If the project_id is NULL, the project does not exist, so insert it
    IF project_id IS NULL THEN
        INSERT INTO projects(name)
        VALUES (project_name);
        -- Get the new project_id
        SELECT LAST_INSERT_ID() INTO project_id;
    END IF;

    -- Insert the correction
    INSERT INTO corrections (
        user_id, project_id, score
    )
    VALUES (
        user_id, project_id, score
    );
END;

DELIMITER ;
