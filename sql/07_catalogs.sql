SELECT split,min(n) min_candidates,max(n) max_candidates,count(*) users FROM (SELECT split,user_id,count(*) n FROM pairs WHERE split>0 GROUP BY split,user_id) GROUP BY split ORDER BY split;
