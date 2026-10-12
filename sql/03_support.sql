SELECT (SELECT min(n) FROM (SELECT user_id,count(*) n FROM small GROUP BY user_id)) min_user_support,
(SELECT min(n) FROM (SELECT video_id,count(*) n FROM small GROUP BY video_id)) min_item_support,
count(DISTINCT user_id)*count(DISTINCT video_id)-count(*) missing_pairs FROM small;
