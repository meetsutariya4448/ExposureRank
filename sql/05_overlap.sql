SELECT (SELECT count(*) FROM (SELECT user_id,video_id FROM small INTERSECT SELECT user_id,video_id FROM big)) pair_overlap,
(SELECT count(*) FROM (SELECT * FROM small INTERSECT SELECT * FROM big)) exact_row_overlap,
(SELECT count(DISTINCT (user_id,video_id)) FROM big) big_unique_pairs,
(SELECT count(*) FROM big) big_rows;
