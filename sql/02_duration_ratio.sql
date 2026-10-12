SELECT count(*) FILTER(WHERE user_id IS NULL OR video_id IS NULL OR NOT isfinite(watch_ratio) OR watch_ratio<0 OR NOT isfinite(video_duration) OR video_duration<=0 OR NOT isfinite(play_duration) OR play_duration<0) invalid,
count(*) FILTER(WHERE abs(watch_ratio-play_duration/video_duration)>.0001+1e-6*abs(watch_ratio)) discrepancies,
count(*) FILTER(WHERE (watch_ratio>=1)!=(play_duration/video_duration>=1) OR (watch_ratio>=.5)!=(play_duration/video_duration>=.5) OR (watch_ratio>=2)!=(play_duration/video_duration>=2)) binary_changes,
max(watch_ratio) maximum_ratio, count(*) FILTER(WHERE watch_ratio>100) ratio_gt100 FROM small;
