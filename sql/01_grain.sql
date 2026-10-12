SELECT count(*) AS "rows", count(DISTINCT (user_id,video_id)) unique_pairs,
count(DISTINCT user_id) users, count(DISTINCT video_id) items FROM small;
