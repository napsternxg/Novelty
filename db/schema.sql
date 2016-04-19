DROP TABLE IF EXISTS `mesh_scores`;

CREATE TABLE `mesh_scores` (
  `MeshTerm` varchar(200) NOT NULL,
  `Year` year(4) NOT NULL,
  `AbsVal` int(11) NOT NULL,
  `TFirstP` int(11) NOT NULL,
  `VolFirstP` float NOT NULL,
  `PredVal` float,
  `Velocity` float,
  `Acceleration` float
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `meshpair_scores`;

CREATE TABLE `meshpair_scores` (
  `Mesh1` varchar(200) NOT NULL,
  `Mesh2` varchar(200) NOT NULL,
  `Year` year(4) NOT NULL,
  `AbsVal` int(11) NOT NULL,
  `TFirstP` int(11) NOT NULL,
  `VolFirstP` float NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `novelty_scores`;

CREATE TABLE `novelty_scores` (
  `PMID` int(10) NOT NULL,
  `Year` year(4) NOT NULL,
  `AbsVal` int(11),
  `TFirstP` int(11),
  `VolFirstP` int(11),
  `acc_pos_vel_min` float,
  `acc_pos_vel_max` float,
  `acc_neg_vel_max` float,
  `acc_neg_vel_min` float,
  `Pair_AbsVal` int(11),
  `Pair_TFirstP` int(11),
  `Pair_VolFirstP` int(11),
  `Mesh_counts` int(11),
  `Exploded_Mesh_counts` int(11)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
