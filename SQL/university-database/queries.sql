-- queries.sql (University)

-- 1. Average instructor salary by department
SELECT dept_name, ROUND(AVG(salary), 2) AS avg_salary
FROM instructor
GROUP BY dept_name
ORDER BY avg_salary DESC;

-- 2. Courses offered per department in current year
SELECT dept_name, COUNT(DISTINCT course_id) AS course_count
FROM section
WHERE year = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY dept_name
ORDER BY course_count DESC;

-- 3. Students with GPA >= 3.5 (assuming letter grade conversion)
SELECT s.name, AVG(
    CASE grade
        WHEN 'A' THEN 4.0
        WHEN 'A-' THEN 3.7
        WHEN 'B+' THEN 3.3
        WHEN 'B' THEN 3.0
        WHEN 'B-' THEN 2.7
        WHEN 'C+' THEN 2.3
        WHEN 'C' THEN 2.0
        WHEN 'D' THEN 1.0
        ELSE 0
    END
) AS gpa
FROM student s
JOIN takes t ON s.ID = t.ID
GROUP BY s.name
HAVING AVG(
    CASE grade
        WHEN 'A' THEN 4.0
        WHEN 'A-' THEN 3.7
        WHEN 'B+' THEN 3.3
        WHEN 'B' THEN 3.0
        WHEN 'B-' THEN 2.7
        WHEN 'C+' THEN 2.3
        WHEN 'C' THEN 2.0
        WHEN 'D' THEN 1.0
        ELSE 0
    END
) >= 3.5
ORDER BY gpa DESC;

-- 4. Department budgets above the overall average
SELECT dept_name, budget
FROM department
WHERE budget > (SELECT AVG(budget) FROM department);

-- 5. Instructor-student ratio per department
SELECT d.dept_name,
       COUNT(DISTINCT i.ID) AS instructors,
       COUNT(DISTINCT s.ID) AS students,
       ROUND(COUNT(DISTINCT s.ID)::NUMERIC / NULLIF(COUNT(DISTINCT i.ID),0), 2) AS ratio
FROM department d
LEFT JOIN instructor i ON d.dept_name = i.dept_name
LEFT JOIN student s ON d.dept_name = s.dept_name
GROUP BY d.dept_name;

-- Top student per department by GPA using window function
WITH grades AS (
  SELECT t.ID,
         CASE t.grade
           WHEN 'A' THEN 4.0 WHEN 'A-' THEN 3.7 WHEN 'B+' THEN 3.3
           WHEN 'B' THEN 3.0 WHEN 'B-' THEN 2.7 WHEN 'C+' THEN 2.3
           WHEN 'C' THEN 2.0 WHEN 'D' THEN 1.0 ELSE 0 END AS gp
  FROM takes t
), student_gpa AS (
  SELECT s.ID, s.name, s.dept_name, AVG(g.gp) AS gpa
  FROM student s
  JOIN grades g ON s.ID = g.ID
  GROUP BY s.ID, s.name, s.dept_name
)
SELECT name, dept_name, gpa
FROM (
  SELECT *, RANK() OVER (PARTITION BY dept_name ORDER BY gpa DESC) AS rnk
  FROM student_gpa
) ranked
WHERE rnk = 1
ORDER BY dept_name;

-- Course schedule matrix via CROSS JOIN (all courses x distinct semester/year)
SELECT c.course_id, sy.semester, sy.year
FROM course c
CROSS JOIN (
  SELECT DISTINCT semester, year FROM section
) sy
ORDER BY c.course_id, sy.year, sy.semester;
