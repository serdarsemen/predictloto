select lotoresults.weekid from lotoresults 
,
(select weekid from lotoresults where 
input1=1 
) as tbl

where lotoresults.weekid= tbl.weekid-1
 and 
lotoresults.input1=1