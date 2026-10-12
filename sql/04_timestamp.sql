SELECT count(*) FILTER(WHERE timestamp IS NULL) missing_timestamp,min(date) first_date,max(date) last_date FROM small;
