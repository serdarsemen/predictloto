DROP TABLE IF EXISTS SAYISALPREDICT;
CREATE TABLE `SAYISALPREDICT` 
(
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `algo` varchar(30) NOT NULL,
  `targeterr` double NOT NULL,
  `populationsize` double NOT NULL,
  `populationdensity` double NOT NULL,
  `weekid` int(10) NOT NULL,
  `realoutput` varchar(400) NOT NULL,
  `successfulpredictcount` int(10) NOT NULL,
  `successfulpredict` varchar(250) NOT NULL,
  `predict` varchar(250) NOT NULL,
 PRIMARY KEY (`id`)
) ;

--INSERT INTO SAYISALPREDICT (`algo`,
 -- `targeterr` ,
 -- `populationsize` ,
 -- `populationdensity` ,
 -- `weekid` ,
 -- `realoutput` ,
 -- `successfulpredictcount`,
 -- `successfulpredict` ,
 -- `predict`) VALUES ("NEAT",0.024,1000,0.2,851,"12; 23; 34; 36; 37; 46",2,
--"{23=22.56, 34=34.47}","{6=5.89, 14=14.18, 23=22.56, 25=25.0, 34=34.47, 41=41.42}")

