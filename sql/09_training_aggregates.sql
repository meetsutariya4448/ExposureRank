SELECT count(*) training_rows,sum(y) positive_rows,(sum(y)+1)/(count(*)+2) beta_prior FROM pairs WHERE split=0;
