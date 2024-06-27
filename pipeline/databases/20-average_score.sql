-- creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser (
    IN new_id INT
)
BEGIN
    DECLARE avg_score DECIMAL(10,2);

    -- Calculate the average score for the user
    SELECT AVG(score) INTO avg_score
    FROM corrections
    WHERE user_id = new_id;

    -- Update the user's average score
    UPDATE users
    SET average_score = avg_score
    WHERE id = new_id;
END;

DELIMITER ;
