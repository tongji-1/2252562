-- 选择使用的数据库
USE flood;

-- 检查表是否已存在，若存在则删除
DROP TABLE IF EXISTS position;

-- 创建新表position
CREATE TABLE `position` (
    `lat` DECIMAL(10, 6),      -- 经度
    `lng` DECIMAL(10, 6),      -- 纬度
    `attribute` VARCHAR(255)   -- 属性描述
);

-- 创建新表research
DROP TABLE IF EXISTS research;
CREATE TABLE `research` (
    `lat` DECIMAL(10, 6),      -- 经度
    `lng` DECIMAL(10, 6),      -- 纬度
    `attribute` VARCHAR(255)   -- 属性描述
);

-- 创建新表meteorology
DROP TABLE IF EXISTS meteorology;
CREATE TABLE `meteorology` (
    `lat` DECIMAL(10, 6),      -- 经度
    `lng` DECIMAL(10, 6),      -- 纬度
    `attribute` VARCHAR(255)   -- 属性描述
);

-- 创建新表runoff
DROP TABLE IF EXISTS runoff;
CREATE TABLE `runoff` (
    `lat` DECIMAL(10, 6),      -- 经度
    `lng` DECIMAL(10, 6),      -- 纬度
    `attribute` VARCHAR(255)   -- 属性描述
);


-- position表中插入数据
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283028, 121.496090, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283103, 121.496283, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283528, 121.496691, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283760, 121.496521, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284160, 121.496649, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284212, 121.496850, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284030, 121.497183, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283822, 121.497524, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283383, 121.497395, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283210, 121.497882, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283565, 121.498100, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283490, 121.498376, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283208, 121.498275, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.282607, 121.498536, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.282036, 121.497736, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.281826, 121.498480, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283169, 121.498751, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.282937, 121.499190, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283138, 121.499257, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283070, 121.499549, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.282930, 121.500048, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283345, 121.500317, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283226, 121.500757, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283642, 121.500976, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283464, 121.500725, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283608, 121.500227, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283786, 121.499689, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284066, 121.499141, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283825, 121.498760, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283992, 121.498531, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284320, 121.498290, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284384, 121.497623, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284513, 121.497298, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285068, 121.497621, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284867, 121.498232, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284714, 121.498699, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284409, 121.499564, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284166, 121.500181, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284051, 121.500595, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284300, 121.500990, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284774, 121.501055, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284678, 121.501475, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284887, 121.501194, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285025, 121.500757, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285289, 121.500155, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285137, 121.500025, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284957, 121.500565, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284560, 121.500418, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284719, 121.499814, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285005, 121.499183, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285187, 121.498717, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285557, 121.498843, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285656, 121.498700, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285862, 121.498218, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285961, 121.498766, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286167, 121.499226, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285986, 121.499752, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285737, 121.500347, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285354, 121.501321, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285813, 121.501599, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285852, 121.502005, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285982, 121.502327, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286202, 121.501795, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286202, 121.501795, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286676, 121.502162, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286777, 121.501731, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286932, 121.501263, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287175, 121.500632, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286755, 121.500467, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286508, 121.500034, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286528, 121.499497, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286889, 121.499234, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287300, 121.499440, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287915, 121.498943, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.288177, 121.498358, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287695, 121.497896, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287543, 121.498106, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287689, 121.498290, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287552, 121.498411, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287434, 121.498652, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287311, 121.498855, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287096, 121.498665, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286406, 121.497966, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286593, 121.497320, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286396, 121.497040, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286955, 121.497336, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287322, 121.497569, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287819, 121.497292, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287562, 121.497066, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.287075, 121.496780, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286962, 121.496324, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286649, 121.496039, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286879, 121.495293, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286902, 121.494706, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286603, 121.494800, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286340, 121.495549, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285759, 121.495986, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285486, 121.495497, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285394, 121.495684, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284953, 121.495544, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284638, 121.495649, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284357, 121.495507, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283842, 121.495351, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285531, 121.494899, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283366, 121.495322, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283268, 121.495505, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283202, 121.495735, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.283445, 121.496205, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284276, 121.496622, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.284466, 121.496723, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285250, 121.497175, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.285570, 121.497371, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.286147, 121.497477, NULL);
INSERT INTO `position` (`lat`, `lng`, `attribute`) VALUES (31.282709, 121.495501, NULL);

-- research表中插入数据
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283445, 121.496205, '气象站点7，降水量710毫升，风速0米每秒，管网负载30%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284409, 121.499564, '气象站点5，降水量373毫升，风速1米每秒，管网负载19%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284051, 121.500595, '气象站点7，降水量547毫升，风速2米每秒，管网负载13%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.282937, 121.499190, '气象站点3，降水量734毫升，风速2米每秒，管网负载4%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283565, 121.498100, '气象站点8，降水量109毫升，风速1米每秒，管网负载21%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284513, 121.497298, '气象站点5，降水量286毫升，风速2米每秒，管网负载11%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283490, 121.498376, '气象站点4，降水量651毫升，风速1米每秒，管网负载19%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285852, 121.502005, '气象站点6，降水量257毫升，风速0米每秒，管网负载17%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287096, 121.498665, '气象站点7，降水量33毫升，风速1米每秒，管网负载4%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284774, 121.501055, '气象站点6，降水量663毫升，风速1米每秒，管网负载17%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286167, 121.499226, '气象站点8，降水量139毫升，风速1米每秒，管网负载10%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.281826, 121.498480, '气象站点5，降水量368毫升，风速1米每秒，管网负载8%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286879, 121.495293, '气象站点1，降水量210毫升，风速1米每秒，管网负载1%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285656, 121.498700, '气象站点4，降水量297毫升，风速0米每秒，管网负载13%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.282036, 121.497736, '气象站点6，降水量167毫升，风速0米每秒，管网负载26%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284678, 121.501475, '气象站点3，降水量389毫升，风速1米每秒，管网负载8%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283028, 121.496090, '气象站点7，降水量147毫升，风速0米每秒，管网负载23%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284957, 121.500565, '气象站点4，降水量774毫升，风速1米每秒，管网负载3%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287434, 121.498652, '气象站点5，降水量139毫升，风速1米每秒，管网负载25%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285557, 121.498843, '气象站点5，降水量564毫升，风速1米每秒，管网负载16%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285737, 121.500347, '气象站点8，降水量571毫升，风速1米每秒，管网负载7%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284719, 121.499814, '气象站点2，降水量144毫升，风速1米每秒，管网负载23%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287819, 121.497292, '气象站点6，降水量138毫升，风速1米每秒，管网负载29%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284357, 121.495507, '气象站点4，降水量413毫升，风速1米每秒，管网负载25%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286962, 121.496324, '气象站点6，降水量360毫升，风速1米每秒，管网负载1%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287562, 121.497066, '气象站点4，降水量758毫升，风速1米每秒，管网负载9%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285354, 121.501321, '气象站点4，降水量45毫升，风速2米每秒，管网负载21%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286147, 121.497477, '气象站点8，降水量684毫升，风速1米每秒，管网负载13%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285486, 121.495497, '气象站点2，降水量390毫升，风速1米每秒，管网负载30%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284714, 121.498699, '气象站点7，降水量51毫升，风速2米每秒，管网负载14%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283786, 121.499689, '气象站点2，降水量358毫升，风速0米每秒，管网负载28%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283528, 121.496691, '气象站点5，降水量638毫升，风速1米每秒，管网负载25%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283608, 121.500227, '气象站点4，降水量264毫升，风速1米每秒，管网负载25%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284212, 121.496850, '气象站点6，降水量174毫升，风速2米每秒，管网负载8%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283383, 121.497395, '气象站点7，降水量784毫升，风速1米每秒，管网负载0%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285025, 121.500757, '气象站点8，降水量224毫升，风速1米每秒，管网负载20%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286603, 121.494800, '气象站点6，降水量50毫升，风速0米每秒，管网负载18%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283103, 121.496283, '气象站点2，降水量2毫升，风速2米每秒，管网负载16%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284638, 121.495649, '气象站点4，降水量754毫升，风速1米每秒，管网负载26%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285531, 121.494899, '气象站点6，降水量680毫升，风速1米每秒，管网负载2%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287543, 121.498106, '气象站点3，降水量8毫升，风速1米每秒，管网负载12%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285862, 121.498218, '气象站点6，降水量194毫升，风速2米每秒，管网负载12%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284066, 121.499141, '气象站点1，降水量722毫升，风速2米每秒，管网负载1%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283822, 121.497524, '气象站点5，降水量388毫升，风速2米每秒，管网负载6%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285250, 121.497175, '气象站点8，降水量686毫升，风速2米每秒，管网负载9%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286202, 121.501795, '气象站点5，降水量324毫升，风速0米每秒，管网负载11%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285137, 121.500025, '气象站点2，降水量596毫升，风速2米每秒，管网负载7%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286202, 121.501795, '气象站点4，降水量573毫升，风速2米每秒，管网负载7%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287915, 121.498943, '气象站点3，降水量428毫升，风速1米每秒，管网负载15%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287552, 121.498411, '气象站点1，降水量667毫升，风速2米每秒，管网负载4%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283210, 121.497882, '气象站点7，降水量190毫升，风速1米每秒，管网负载6%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283268, 121.495505, '气象站点6，降水量771毫升，风速1米每秒，管网负载18%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285759, 121.495986, '气象站点3，降水量580毫升，风速2米每秒，管网负载26%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283345, 121.500317, '气象站点4，降水量498毫升，风速2米每秒，管网负载19%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.288177, 121.498358, '气象站点4，降水量650毫升，风速2米每秒，管网负载9%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283992, 121.498531, '气象站点1，降水量447毫升，风速1米每秒，管网负载14%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283226, 121.500757, '气象站点4，降水量112毫升，风速0米每秒，管网负载4%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285068, 121.497621, '气象站点8，降水量94毫升，风速0米每秒，管网负载4%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286340, 121.495549, '气象站点3，降水量588毫升，风速0米每秒，管网负载27%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284466, 121.496723, '气象站点4，降水量530毫升，风速2米每秒，管网负载2%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286955, 121.497336, '气象站点4，降水量624毫升，风速0米每秒，管网负载22%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286932, 121.501263, '气象站点8，降水量96毫升，风速0米每秒，管网负载6%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285982, 121.502327, '气象站点3，降水量578毫升，风速1米每秒，管网负载0%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287311, 121.498855, '气象站点2，降水量796毫升，风速1米每秒，管网负载3%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286889, 121.499234, '气象站点2，降水量80毫升，风速2米每秒，管网负载1%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286508, 121.500034, '气象站点2，降水量282毫升，风速1米每秒，管网负载28%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284320, 121.498290, '气象站点2，降水量246毫升，风速2米每秒，管网负载10%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286528, 121.499497, '气象站点1，降水量93毫升，风速2米每秒，管网负载4%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283070, 121.499549, '气象站点3，降水量273毫升，风速0米每秒，管网负载18%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284166, 121.500181, '气象站点4，降水量22毫升，风速1米每秒，管网负载29%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.282607, 121.498536, '气象站点8，降水量475毫升，风速0米每秒，管网负载17%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283202, 121.495735, '气象站点8，降水量521毫升，风速0米每秒，管网负载27%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283169, 121.498751, '气象站点4，降水量359毫升，风速1米每秒，管网负载2%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285961, 121.498766, '气象站点8，降水量41毫升，风速2米每秒，管网负载29%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287689, 121.498290, '气象站点6，降水量490毫升，风速1米每秒，管网负载12%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285005, 121.499183, '气象站点8，降水量786毫升，风速0米每秒，管网负载4%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286649, 121.496039, '气象站点7，降水量540毫升，风速2米每秒，管网负载9%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285394, 121.495684, '气象站点8，降水量307毫升，风速2米每秒，管网负载28%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284953, 121.495544, '气象站点4，降水量790毫升，风速1米每秒，管网负载20%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283642, 121.500976, '气象站点8，降水量480毫升，风速1米每秒，管网负载14%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283760, 121.496521, '气象站点7，降水量767毫升，风速0米每秒，管网负载21%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286777, 121.501731, '气象站点4，降水量434毫升，风速1米每秒，管网负载22%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286755, 121.500467, '气象站点6，降水量397毫升，风速0米每秒，管网负载15%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284276, 121.496622, '气象站点1，降水量681毫升，风速1米每秒，管网负载18%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284030, 121.497183, '气象站点6，降水量20毫升，风速1米每秒，管网负载7%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285570, 121.497371, '气象站点3，降水量697毫升，风速0米每秒，管网负载21%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284867, 121.498232, '气象站点3，降水量457毫升，风速1米每秒，管网负载9%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287300, 121.499440, '气象站点2，降水量36毫升，风速1米每秒，管网负载10%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.282930, 121.500048, '气象站点8，降水量547毫升，风速1米每秒，管网负载26%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285289, 121.500155, '气象站点8，降水量82毫升，风速0米每秒，管网负载1%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287075, 121.496780, '气象站点8，降水量13毫升，风速2米每秒，管网负载9%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287695, 121.497896, '气象站点7，降水量213毫升，风速0米每秒，管网负载30%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286396, 121.497040, '气象站点1，降水量623毫升，风速1米每秒，管网负载24%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284887, 121.501194, '气象站点4，降水量640毫升，风速1米每秒，管网负载19%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286593, 121.497320, '气象站点5，降水量20毫升，风速2米每秒，管网负载5%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283825, 121.498760, '气象站点4，降水量201毫升，风速1米每秒，管网负载27%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283366, 121.495322, '气象站点7，降水量534毫升，风速1米每秒，管网负载26%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285813, 121.501599, '气象站点3，降水量674毫升，风速1米每秒，管网负载2%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286406, 121.497966, '气象站点7，降水量703毫升，风速0米每秒，管网负载7%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287175, 121.500632, '气象站点6，降水量619毫升，风速0米每秒，管网负载14%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283138, 121.499257, '气象站点2，降水量540毫升，风速2米每秒，管网负载5%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284160, 121.496649, '气象站点3，降水量322毫升，风速1米每秒，管网负载16%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.287322, 121.497569, '气象站点7，降水量363毫升，风速1米每秒，管网负载15%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283208, 121.498275, '气象站点2，降水量51毫升，风速1米每秒，管网负载18%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284560, 121.500418, '气象站点2，降水量402毫升，风速1米每秒，管网负载15%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284384, 121.497623, '气象站点5，降水量707毫升，风速0米每秒，管网负载21%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285986, 121.499752, '气象站点1，降水量82毫升，风速2米每秒，管网负载6%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.285187, 121.498717, '气象站点8，降水量762毫升，风速2米每秒，管网负载2%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283464, 121.500725, '气象站点5，降水量126毫升，风速1米每秒，管网负载0%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286676, 121.502162, '气象站点2，降水量123毫升，风速1米每秒，管网负载4%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.284300, 121.500990, '气象站点5，降水量516毫升，风速1米每秒，管网负载5%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.286902, 121.494706, '气象站点7，降水量361毫升，风速2米每秒，管网负载18%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.283842, 121.495351, '气象站点3，降水量171毫升，风速2米每秒，管网负载19%');
INSERT INTO `research` (`lat`, `lng`, `attribute`) VALUES (31.282709, 121.495501, '气象站点4，降水量460毫升，风速1米每秒，管网负载10%');

-- meteorology表中插入数据
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283445, 121.496205, '气象站点1，降水量45毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284409, 121.499564, '气象站点2，降水量26毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284051, 121.500595, '气象站点4，降水量7毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.282937, 121.499190, '气象站点6，降水量53毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283565, 121.498100, '气象站点4，降水量6毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284513, 121.497298, '气象站点4，降水量3毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283490, 121.498376, '气象站点4，降水量41毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285852, 121.502005, '气象站点7，降水量52毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.287096, 121.498665, '气象站点2，降水量31毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284774, 121.501055, '气象站点4，降水量54毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.286167, 121.499226, '气象站点5，降水量53毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.281826, 121.498480, '气象站点1，降水量27毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.286879, 121.495293, '气象站点2，降水量58毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285656, 121.498700, '气象站点8，降水量1毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.282036, 121.497736, '气象站点4，降水量52毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284678, 121.501475, '气象站点2，降水量75毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283028, 121.496090, '气象站点2，降水量34毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284957, 121.500565, '气象站点5，降水量9毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.287434, 121.498652, '气象站点7，降水量22毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285557, 121.498843, '气象站点8，降水量77毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285737, 121.500347, '气象站点2，降水量34毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284719, 121.499814, '气象站点3，降水量51毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.287819, 121.497292, '气象站点6，降水量70毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284357, 121.495507, '气象站点6，降水量49毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.286962, 121.496324, '气象站点4，降水量35毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.287562, 121.497066, '气象站点1，降水量27毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285354, 121.501321, '气象站点3，降水量75毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.286147, 121.497477, '气象站点2，降水量16毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285486, 121.495497, '气象站点1，降水量39毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284714, 121.498699, '气象站点8，降水量28毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283786, 121.499689, '气象站点5，降水量50毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283528, 121.496691, '气象站点7，降水量7毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283608, 121.500227, '气象站点6，降水量28毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284212, 121.496850, '气象站点5，降水量47毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283383, 121.497395, '气象站点4，降水量37毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285025, 121.500757, '气象站点8，降水量42毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.286603, 121.494800, '气象站点8，降水量4毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283103, 121.496283, '气象站点1，降水量25毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284638, 121.495649, '气象站点7，降水量55毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285531, 121.494899, '气象站点5，降水量20毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.287543, 121.498106, '气象站点6，降水量37毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285862, 121.498218, '气象站点2，降水量0毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.284066, 121.499141, '气象站点5，降水量73毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283822, 121.497524, '气象站点1，降水量22毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285250, 121.497175, '气象站点2，降水量25毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.286202, 121.501795, '气象站点5，降水量14毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285137, 121.500025, '气象站点5，降水量69毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.286202, 121.501795, '气象站点4，降水量45毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.287915, 121.498943, '气象站点1，降水量8毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.287552, 121.498411, '气象站点7，降水量9毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283210, 121.497882, '气象站点4，降水量12毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283268, 121.495505, '气象站点5，降水量41毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.285759, 121.495986, '气象站点6，降水量57毫升，风速0米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283345, 121.500317, '气象站点5，降水量52毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.288177, 121.498358, '气象站点4，降水量33毫升，风速1米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283992, 121.498531, '气象站点5，降水量50毫升，风速2米每秒');
INSERT INTO `meteorology` (`lat`, `lng`, `attribute`) VALUES (31.283226, 121.500757, '气象站点7，降水量33毫升，风速1米每秒');

-- runoff表中插入数据
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.285068, 121.497621, '排水设施1，管网负载22%，降水量524毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286340, 121.495549, '排水设施3，管网负载13%，降水量50毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284466, 121.496723, '排水设施1，管网负载12%，降水量273毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286955, 121.497336, '排水设施6，管网负载2%，降水量122毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286932, 121.501263, '排水设施1，管网负载14%，降水量701毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.285982, 121.502327, '排水设施6，管网负载12%，降水量151毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.287311, 121.498855, '排水设施4，管网负载5%，降水量371毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286889, 121.499234, '排水设施3，管网负载9%，降水量249毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286508, 121.500034, '排水设施2，管网负载19%，降水量512毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284320, 121.498290, '排水设施4，管网负载5%，降水量104毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286528, 121.499497, '排水设施2，管网负载6%，降水量692毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283070, 121.499549, '排水设施5，管网负载8%，降水量567毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284166, 121.500181, '排水设施5，管网负载21%，降水量780毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.282607, 121.498536, '排水设施3，管网负载14%，降水量579毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283202, 121.495735, '排水设施5，管网负载29%，降水量576毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283169, 121.498751, '排水设施3，管网负载8%，降水量133毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.285961, 121.498766, '排水设施5，管网负载2%，降水量653毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.287689, 121.498290, '排水设施1，管网负载1%，降水量448毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.285005, 121.499183, '排水设施6，管网负载28%，降水量557毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286649, 121.496039, '排水设施7，管网负载4%，降水量16毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.285394, 121.495684, '排水设施3，管网负载14%，降水量356毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284953, 121.495544, '排水设施4，管网负载6%，降水量696毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283642, 121.500976, '排水设施5，管网负载4%，降水量435毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283760, 121.496521, '排水设施6，管网负载18%，降水量347毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286777, 121.501731, '排水设施3，管网负载13%，降水量714毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286755, 121.500467, '排水设施8，管网负载1%，降水量617毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284276, 121.496622, '排水设施8，管网负载8%，降水量434毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284030, 121.497183, '排水设施6，管网负载12%，降水量77毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.285570, 121.497371, '排水设施4，管网负载4%，降水量280毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284867, 121.498232, '排水设施1，管网负载9%，降水量750毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.287300, 121.499440, '排水设施6，管网负载12%，降水量763毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.282930, 121.500048, '排水设施2，管网负载5%，降水量601毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.285289, 121.500155, '排水设施4，管网负载11%，降水量78毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.287075, 121.496780, '排水设施1，管网负载6%，降水量183毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.287695, 121.497896, '排水设施1，管网负载21%，降水量746毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286396, 121.497040, '排水设施3，管网负载9%，降水量52毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284887, 121.501194, '排水设施1，管网负载25%，降水量117毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286593, 121.497320, '排水设施2，管网负载16%，降水量237毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283825, 121.498760, '排水设施2，管网负载25%，降水量378毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283366, 121.495322, '排水设施4，管网负载19%，降水量324毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.285813, 121.501599, '排水设施3，管网负载23%，降水量698毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286406, 121.497966, '排水设施7，管网负载9%，降水量33毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.287175, 121.500632, '排水设施8，管网负载20%，降水量36毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283138, 121.499257, '排水设施4，管网负载5%，降水量531毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284160, 121.496649, '排水设施8，管网负载22%，降水量627毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.287322, 121.497569, '排水设施8，管网负载26%，降水量97毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283208, 121.498275, '排水设施1，管网负载9%，降水量330毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284560, 121.500418, '排水设施7，管网负载19%，降水量197毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284384, 121.497623, '排水设施6，管网负载30%，降水量690毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.285986, 121.499752, '排水设施2，管网负载11%，降水量422毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.285187, 121.498717, '排水设施1，管网负载27%，降水量298毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283464, 121.500725, '排水设施2，管网负载7%，降水量485毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286676, 121.502162, '排水设施5，管网负载7%，降水量447毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.284300, 121.500990, '排水设施5，管网负载26%，降水量407毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.286902, 121.494706, '排水设施6，管网负载0%，降水量500毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.283842, 121.495351, '排水设施1，管网负载27%，降水量497毫升');
INSERT INTO `runoff` (`lat`, `lng`, `attribute`) VALUES (31.282709, 121.495501, '排水设施3，管网负载14%，降水量258毫升');