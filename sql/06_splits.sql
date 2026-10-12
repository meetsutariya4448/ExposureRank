SELECT split,count(*) pairs,count(DISTINCT (user_id,video_id)) unique_pairs FROM pairs GROUP BY split ORDER BY split;
