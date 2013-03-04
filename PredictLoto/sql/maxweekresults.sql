select * from lotoresults where weekid=
(select max(weekid) from lotoresults) 