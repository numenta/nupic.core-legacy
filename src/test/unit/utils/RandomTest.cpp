/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/**
 * @file
 */


#include "RandomTest.hpp"
#include <nupic/os/Env.hpp>
#include <nupic/ntypes/MemStream.hpp>
#include <nupic/utils/LoggingException.hpp>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

using namespace nupic;

RandomTest::RandomTest() {};

RandomTest::~RandomTest() {};

#include "RandomPrivateOrig.c"

// Expected values with seed of 148
// Comparing against expected values ensures the same result
// on all platforms.
UInt32 expected[1000] =
{
33267067, 1308471064, 567525506, 744151466, 1514731226,
320263095, 850223971, 272111712, 1447892216, 1051935310,
767613522, 1966421570, 740590150, 238534644, 439141684,
1006390399, 683226375, 105163871, 1148420320, 1897583466,
1364189210, 2056512848, 2012249758, 724656770, 862951273,
739791114, 2018040533, 733864322, 181139176, 1764207967,
1203036433, 214406243, 925195384, 1770561940, 958557709,
292442962, 2090825035, 1808781680, 564554675, 1391233604,
713233342, 1332168197, 1210171526, 1453823492, 1570702841,
1649313210, 312730244, 106445569, 1754477081, 1461150564,
2004029035, 971182643, 1370179764, 1868795145, 1695839414,
85647389, 461102611, 1566396299, 819511712, 642241788,
1183120618, 2022548145, 856648031, 2108316002, 1645626437,
1815205741, 253275317, 1588967825, 1476503773, 817829992,
832717781, 42253468, 2514541, 2042889307, 1496076960,
1573217382, 1544718869, 1808807204, 1679662951, 1151712303,
1122474120, 1536208338, 2122894946, 345170236, 1257519835,
1671250712, 430817626, 1718622447, 1090163363, 1250329338,
213380587, 125800334, 1125393835, 1070028618, 86632688,
623536625, 737750711, 339908005, 65020802, 66770837,
1157737997, 897738583, 109024305, 1160252538, 793144242,
1605101265, 585986273, 190379463, 1266424822, 118165576,
1342091766, 241415294, 1654373915, 1317503065, 586585531,
764410102, 841270129, 1017403157, 335548901, 1931433493,
120248847, 548929488, 2057233827, 1245642682, 1618958107,
2143866515, 1869179307, 209225170, 336290873, 1934200109,
275996007, 1494028870, 684455044, 385020312, 506797761,
1477599286, 1990121578, 1092784034, 1667978750, 1109062752,
1210949610, 862586868, 1350478046, 717839877, 32606285,
1937063577, 1482249980, 873876415, 806983086, 1817798881,
657826260, 927231933, 219244722, 567576439, 25390968,
1838202829, 563959306, 1894570275, 2047427999, 900250179,
1681286737, 175940359, 246795402, 218258133, 560960671,
753593163, 1695857420, 403598601, 1846377197, 1216352522,
1512661353, 909843159, 2078939390, 715655752, 1627683037,
2111545676, 505235681, 962449369, 837938443, 1312218768,
632764602, 1495764703, 91967053, 852009324, 2063341142,
117358021, 542728505, 479816800, 2011928297, 442672857,
1380066980, 1545731386, 618613216, 1626862382, 1763989519,
1179573887, 232971897, 1312363291, 1583172489, 2079349094,
381232165, 948350194, 841708605, 312687908, 1664005946,
321907994, 276749936, 21757980, 1284357363, 1114688379,
1333976748, 1917121966, 462969434, 1425943801, 621647642,
378826928, 1543301823, 1164376148, 858643728, 1407746472,
1607049005, 91227060, 805994210, 78178573, 1718089442,
422500081, 1257752460, 1951061339, 1734863373, 693441301,
1882926785, 2116095538, 1641791496, 577151743, 281299798,
1158313794, 899059737, 558049734, 1180071774, 35933453,
1672738113, 366564874, 1953055419, 2135707547, 1792508676,
427219413, 367050827, 1188326851, 1591595561, 1225694556,
448589675, 1051160918, 1316921616, 1254583885, 1129339491,
887527411, 1677083966, 239608304, 691105102, 1264463691,
933049605, 426548240, 1233075582, 427357453, 1003699983,
1514375380, 1585671248, 1902759720, 2072425115, 618259374,
1938693173, 1597679580, 984824249, 1744264944, 1585903480,
629849277, 24000710, 1952954307, 1818176128, 1615596271,
1031165215, 119282155, 519273542, 200603184, 1373866040,
1648613033, 1088130595, 903466358, 1888221337, 1779235697,
20446402, 673787295, 58300289, 1253521984, 1101144748,
1062000272, 620413716, 539332348, 817276345, 545355183,
1157591723, 608485870, 2143034764, 2142415972, 205267167,
1581454596, 624781601, 229267877, 1386925255, 295474081,
1844864148, 270606823, 414756236, 216654042, 471210007,
1788622276, 1865267076, 1559340602, 544604986, 1606004765,
1191092651, 565051388, 132308412, 1249392941, 1818573372,
1233453161, 163909565, 291503441, 1772785509, 981185910,
836858624, 782893584, 1589671781, 832409740, 777825908,
1794938948, 266380688, 1402607509, 2024206825, 1653305944,
1698081590, 1721587325, 1923912767, 2112837826, 1938241368,
247639126, 1753976454, 1656024796, 1806979728, 151097793,
1114545913, 850588731, 716149181, 1246854326, 2099981672,
387238906, 332823839, 116407590, 678742347, 2105609348,
1097593500, 1515600971, 741019285, 539781633, 200527064,
1518845193, 187236933, 466907752, 773969055, 63960110,
2120213696, 324566997, 1785547436, 1896642815, 289921176,
1576305156, 2144281941, 2043897630, 1084846304, 1803778021,
47511775, 51908569, 506883105, 763660957, 1298762895,
459381129, 1150899863, 1631586734, 575788719, 1829642210,
1589712435, 1673382220, 1197759533, 183248072, 65680205,
1398286597, 1702093265, 252917139, 1865194350, 328578672,
316877249, 1837924398, 653145670, 2102424685, 1587083566,
943066846, 1531246193, 1583881859, 839480828, 468608849,
1240176233, 886992604, 520517419, 1747059338, 1650653561,
1819280314, 58956819, 654069776, 1303383401, 634745539,
336228338, 745612188, 160644111, 1533987871, 928860260,
226324316, 784790821, 483469877, 479241455, 502501523,
812048550, 796118705, 192942273, 1465194220, 751059742,
1780025839, 260777418, 134822288, 1216424051, 1100258246,
603431137, 309116636, 1987250850, 1123948556, 2056175974,
1490420763, 795745223, 2115132793, 2144490539, 2099128624,
602394684, 333235229, 697257164, 763038795, 1867223101,
1626117424, 989363112, 504530274, 2109587301, 1468604567,
1007031797, 774152203, 117239624, 1199974070, 91862775,
868299367, 832516262, 352640193, 1003121655, 2048940313,
1452898440, 1606552792, 210573301, 1292665642, 583017701,
119265627, 635602758, 1378762924, 86914772, 632609649,
1330407900, 689309457, 965844879, 2027665064, 1452348252,
685584332, 1506298840, 294227716, 1190114606, 1468402493,
1762832284, 49662755, 95071049, 1880071908, 1249636825,
186933824, 600887627, 2082153087, 539574018, 1604009282,
1983609752, 1992472458, 1063078427, 46699405, 1137654452,
1646096128, 165965032, 1773257210, 877375404, 252879805,
258383212, 60299656, 942189262, 1224228091, 2087964720,
247053866, 1909812423, 1446779912, 541281583, 952443381,
767698757, 156630219, 1002106136, 862769806, 2036702127,
104259313, 1049703631, 490106107, 38928753, 1589277649,
2094115389, 2022538505, 1434266459, 1009710168, 2069237911,
424437263, 508322648, 87719295, 50210826, 1385698052,
340599100, 308594038, 1445997708, 1282788362, 1532822129,
1386478780, 1529842229, 1295150904, 685775044, 2071123812,
100110637, 1453473802, 80270383, 1102216773, 168759960,
2116972510, 1206476086, 1218463591, 459594969, 1245404839,
660257592, 406226711, 1120459697, 2094524051, 1415936879,
1042213960, 371477667, 1924259528, 1129933255, 421688493,
1162473932, 1470532356, 730282531, 460987993, 605837070,
115621012, 1847466773, 2135679299, 1410771916, 385758170,
2059319463, 1510882553, 1839231972, 2139589846, 465615678,
2007991932, 2109078709, 1672091764, 1078971876, 421190030,
770012956, 1739229468, 827416741, 1890472653, 1686269872,
95869973, 785202965, 2057747539, 2020129501, 1915136220,
331952384, 1035119785, 1238184928, 1062234915, 1496107778,
1844021999, 1177855927, 1196090904, 1832217650, 441144195,
1581849074, 1744053466, 1952026748, 1273597398, 1736159664,
270158778, 1134105682, 1697754725, 1942250542, 65593910,
2118944756, 564779850, 1804823379, 798877849, 307768855,
1343609603, 894747822, 1092971820, 1253873494, 767393675,
860624393, 1585825878, 1802513461, 2098809321, 500577145,
1151137591, 1795347672, 1678433072, 199744847, 1480081675,
2119577267, 1781593921, 1076651493, 1924120367, 907707671,
665327509, 46795497, 2041813354, 215598587, 1989046039,
2107407264, 187059695, 406342242, 1764746995, 985937544,
714111097, 960872950, 1880685367, 1807082918, 67262796,
500595394, 520223663, 1653088674, 155625207, 471549336,
6182171, 1306762799, 119413361, 1684615243, 1506507646,
1599495036, 1656708862, 1140617920, 528662881, 1433345581,
2048325591, 1193990390, 1480141078, 1942655297, 1409588977,
1321703470, 1902578914, 1596648672, 1728045712, 1519842261,
435102569, 294673161, 333231564, 168304288, 2101756079,
400494360, 668899682, 474496094, 2053583035, 824524890,
946045431, 2059765206, 2131287689, 1065458792, 1596896802,
1490311687, 517470180, 1106122016, 483445959, 1046133061,
391983950, 384287903, 92639803, 1872125028, 179459552,
1502228781, 1046344850, 2082038466, 951393805, 626906914,
1454397080, 1386496374, 921580076, 1787628644, 1554800662,
875852507, 40639356, 76216697, 1350348602, 2094222391,
900741587, 148910385, 2006503950, 884545628, 1214369177,
1455917104, 227373667, 1731839357, 414555472, 710819627,
630488770, 806539422, 1095107530, 723128573, 531180803,
1274567082, 77873706, 1577525653, 1209121901, 1029267512,
56948920, 516035333, 268280238, 978528996, 156180329,
1823080901, 1854381503, 196819685, 1899297598, 1057246457,
143558429, 652555537, 1206156842, 2578731, 1537101165,
273042371, 1458495835, 1764474832, 2004881728, 1873051307,
327810811, 487886850, 532107082, 1422918341, 1211015424,
1063287885, 550001776, 1288889130, 493329890, 1759123677,
170672994, 550278810, 127675362, 438953233, 1528807806,
283855691, 114550486, 1235705662, 480675376, 2013848084,
145468471, 624233805, 518919973, 1351625314, 626812536,
2056021138, 1624667685, 2085308371, 1673012322, 1482065766,
1810876031, 2000823134, 1969952616, 195499465, 1276257827,
1033484392, 1258787350, 1826259603, 174889875, 1752117240,
1437899632, 345562869, 154912403, 1565574994, 784516102,
1683720209, 1849430685, 899066588, 771942223, 182622414,
765431024, 917410695, 806856219, 1284350997, 121552361,
1433668756, 1192888487, 1746220046, 1371493479, 718417162,
1080802164, 1034885862, 571756648, 903271133, 1230385327,
1848014475, 1936755525, 341689029, 1526790431, 2111645400,
2093806270, 817206415, 309724622, 101235025, 235297762,
1094240724, 1784955234, 2084728447, 1993307313, 409413810,
119867213, 611254689, 1326824505, 926723433, 1895605687,
1448376866, 212908541, 941010526, 1047113264, 1584402020,
1659427688, 2127915429, 471804235, 83700688, 883702914,
1702189562, 1931715164, 672974791, 2043878592, 1311021947,
637136544, 1990201214, 2128228362, 946861166, 2091436239,
216042476, 2041101890, 1728907825, 153287276, 1886925555,
2138321635, 273154489, 350696597, 1317662492, 1199877922,
98818636, 618555710, 1412786463, 1039829162, 1665668975,
849704836, 551773203, 1646100756, 1321509071, 635473891,
382320022, 876214985, 419705407, 1055294813, 772609929,
1730727354, 1692431357, 615327495, 1711472069, 491808875,
559280086, 1927514545, 385427118, 140704264, 2080801821,
124869025, 131542251, 206472663, 475565622, 1449204744,
1406350585, 574384258, 2067760454, 671653401, 1614213421,
1585945781, 1521358237, 18502976, 1084562889, 695383660,
653976867, 1466882911, 1571598645, 1073682275, 374694077,
196724927, 656925981, 2067125434, 812052422, 220914402,
411450662, 1371332509, 945300, 796877780, 1512036773,
2081747121, 921746805, 1643579024, 140736136, 1397312428,
945300120, 1547086722, 1971696686, 865576927, 71256475,
1438426459, 304039060, 1592614712, 1456929435, 1388601950,
140514724, 2110906303, 708001213, 1712113369, 1037104930,
1082695290, 1908838296, 1694030911, 1002337077, 573407071,
1914945314, 1413787739, 1944739580, 1915890614, 63181871,
1309292705, 1850154087, 984928676, 805388081, 1990890224,
234757456, 1750688202, 1390493298, 58970495, 468781481,
1461749773, 1497396954, 772820541, 906880837, 806842742,
13938843, 1047395561, 770265397, 721940057, 612025282,
1807370327, 1804635347, 373379931, 1353917590, 659488776,
946787002, 1121379256, 2073276515, 744042934, 889786222,
2136458386, 2053335639, 592456662, 973903415, 711240072
};

void RandomTest::RunTests()
{
  UInt32 r1, r2, r3;
  // make sure the global instance is seeded from time()
  // in the test situation, we can be sure we were seeded less than 100000 seconds ago
  // make sure random number system is initialized by creating a random
  // object. Use the object to make sure the compiler doesn't complain about
  // an unused variable
  Random r;
  UInt64 x = r.getUInt64();
  TEST(x != 0);


  mysrandom(148);
  Random rWithSeed(148);
  for (auto & elem : expected)
  {
    r2 = rWithSeed.getUInt32();
    r3 = myrandom();
    // Uncomment to generate expected values
    // NTA_DEBUG << "expected[" << i << "] = " << r1 << ";";
    if ((r2 != r3) || (r3 != elem))
    {
      // only create a test result if we get a failure. Otherwise we
      // end up creating and saving a lot of unnecessary test results.
      TESTEQUAL(r2, r3);
      break;
    }

  }
  if (r2 == r3)
  {
    // create one positive test result if everything is good
    TEST(true);
  }
  TESTEQUAL(148U, rWithSeed.getSeed());

  {
    // same test, different seed
    mysrandom(98765);
    Random r(98765);
    for (int i = 0; i < 1000; i++)
    {
      r1 = r.getUInt32();
      r2 = myrandom();
      if (r1 != r2)
      {
        TESTEQUAL(r1, r2);
        break;
      }
    }
    if (r1 == r2)
    {
      // create one positive test result if everything is good
      TEST(true);
    }
    TESTEQUAL(98765U, r.getSeed());
  }

  {
    // test copy constructor.
    Random r1(289436);
    int i;
    for (i = 0; i < 100; i++)
      r1.getUInt32();
    Random r2(r1);

    UInt32 v1, v2;
    for (i = 0; i < 100; i++)
    {
      v1 = r1.getUInt32();
      v2 = r2.getUInt32();
      if (v1 != v2)
        break;
    }
    TEST2("copy constructor", v1 == v2);
  }

  {
    // test operator=
    Random r1(289436);
    int i;
    for (i = 0; i < 100; i++)
      r1.getUInt32();
    Random r2(86726008);
    for (i = 0; i < 100; i++)
      r2.getUInt32();

    r2 = r1;
    UInt32 v1, v2;
    for (i = 0; i < 100; i++)
    {
      v1 = r1.getUInt32();
      v2 = r2.getUInt32();
      if (v1 != v2)
        break;
    }
    TEST2("operator=", v1 == v2);
  }

  {
    // test serialization/deserialization
    Random r1(862973);
    int i;
    for (i = 0; i < 100; i++)
      r1.getUInt32();

    // serialize
    OMemStream ostream;
    ostream << r1;

    // print out serialization for debugging
    std::string x(ostream.str(), ostream.pcount());
    NTA_INFO << "random serialize string: '" << x << "'";
    // Serialization should be deterministic and platform independent
    std::string expectedString = "random-v1 862973 randomimpl-v1 31 1624753037 2009419031 633377166 -1574892086 1144529851 1406716263 1553314465 1423305391 27869257 -777179591 -476654188 -1158534456 1569565720 -1530107687 -689150702 2101555921 1535380948 1896399958 -452440042 1943129361 -1905643199 -1174375166 2019813130 -1490833398 1624596014 -694914945 517970867 1155092552 -244858627 -823365838 -938489650 7 10 endrandom-v1";
    TESTEQUAL(expectedString, x);


    // deserialize into r2
    std::string s(ostream.str(), ostream.pcount());
    std::stringstream ss(s);
    Random r2;
    ss >> r2;

    // r1 and r2 should be identical
    TESTEQUAL(r1.getSeed(), r2.getSeed());

    UInt32 v1, v2;
    for (i = 0; i < 100; i++)
    {
      v1 = r1.getUInt32();
      v2 = r2.getUInt32();
      NTA_CHECK(v1 == v2);
    }
    TESTEQUAL2("serialization", v1, v2);
  }

  {
    // make sure that we are returning values in the correct range
    // @todo perform statistical tests
    Random r;
    UInt32 seed = r.getSeed();
    TEST2("seed not zero", seed != 0);
    int i;
    UInt32 max32 = 10000000;
    UInt64 max64 = (UInt64)max32 * (UInt64)max32;
    for (i = 0; i < 200; i++)
    {
      UInt32 i32 = r.getUInt32(max32);
      TEST2("UInt32",  i32 < max32);
      UInt64 i64 = r.getUInt64(max64);
      TEST2("UInt64",  i64 < max64);
      Real64 r64 = r.getReal64();
      TEST2("Real64", r64 >= 0.0 && r64 < 1.0);
    }
  }

  {
    // tests for sampling

    UInt32 population[] = {1, 2, 3, 4};
    Random r(42);

    {
      // choose some elements
      UInt32 choices[2];
      r.sample(population, 4, choices, 2);
      TESTEQUAL2("check element 0", 2, choices[0]);
      TESTEQUAL2("check element 1", 4, choices[1]);
    }

    {
      // choose all elements
      UInt32 choices[4];
      r.sample(population, 4, choices, 4);
      TESTEQUAL2("check element 0", 1, choices[0]);
      TESTEQUAL2("check element 1", 2, choices[1]);
      TESTEQUAL2("check element 2", 3, choices[2]);
      TESTEQUAL2("check element 3", 4, choices[3]);
    }

    {
      // nChoices > nPopulation
      UInt32 choices[5];
      bool caught = false;
      try
      {
        r.sample(population, 4, choices, 5);
      }
      catch (LoggingException& exc)
      {
        caught = true;
      }
      TEST2("checking for exception from population too small", caught);
    }
  }

  {
    // tests for shuffling
    Random r(42);
    UInt32 arr[] = {1, 2, 3, 4};

    UInt32* start = arr;
    UInt32* end = start + 4;
    r.shuffle(start, end);

    TESTEQUAL2("check element 0", 3, arr[0]);
    TESTEQUAL2("check element 1", 2, arr[1]);
    TESTEQUAL2("check element 2", 4, arr[2]);
    TESTEQUAL2("check element 3", 1, arr[3]);
  }

  {
    // tests for Cap'n Proto serialization
    Random r1, r2;
    UInt32 v1, v2;

    const char* outputPath = "RandomTest1.temp";

    {
      std::ofstream out(outputPath, std::ios::binary);
      r1.write(out);
      out.close();
    }
    {
      std::ifstream in(outputPath, std::ios::binary);
      r2.read(in);
      in.close();
    }
    v1 = r1.getUInt32();
    v2 = r2.getUInt32();
    TESTEQUAL2("check serialization for unused Random object", v1, v2);

    {
      std::ofstream out(outputPath, std::ios::binary);
      r1.write(out);
      out.close();
    }
    {
      std::ifstream in(outputPath, std::ios::binary);
      r2.read(in);
      in.close();
    }
    v1 = r1.getUInt32();
    v2 = r2.getUInt32();
    TESTEQUAL2("check serialization for used Random object", v1, v2);

    // clean up
    remove(outputPath);
  }

}
