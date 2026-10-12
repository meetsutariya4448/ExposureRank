SELECT sum(n) reconciled_training_rows,sum(pos) reconciled_training_positives,count(*) items FROM (SELECT video_id,count(*) n,sum(y) pos FROM pairs WHERE split=0 GROUP BY video_id);
