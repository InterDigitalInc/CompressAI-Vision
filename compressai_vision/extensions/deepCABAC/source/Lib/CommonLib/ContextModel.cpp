/* -----------------------------------------------------------------------------
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or 
other Intellectual Property Rights other than the copyrights concerning 
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2022, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


------------------------------------------------------------------------------------------- */
#include "ContextModel.h"

const uint8_t rps_table[256] = {
  128, 112, 97,  84,  74,  65,  57, 50, 45,  39,  34,  30,  27,  23,  20,  18, 15,  14,  12,  11,  10,  9,   7,  7,
  5,   5,   4,   4,   3,   3,   2,  2,  142, 125, 108, 93,  82,  72,  63,  56, 50,  43,  38,  33,  30,  26,  22, 20,
  17,  16,  13,  12,  11,  10,  8,  8,  6,   6,   5,   5,   3,   3,   2,   2,  156, 137, 119, 103, 90,  79,  70, 61,
  55,  48,  42,  37,  33,  28,  24, 22, 19,  17,  15,  13,  12,  11,  9,   9,  6,   6,   5,   5,   4,   4,   2,  2,
  171, 150, 130, 112, 99,  87,  76, 67, 60,  52,  46,  40,  36,  31,  27,  24, 21,  19,  16,  15,  13,  12,  10, 10,
  7,   7,   6,   6,   4,   4,   3,  3,  185, 162, 141, 121, 107, 94,  82,  73, 65,  56,  50,  43,  39,  34,  29, 26,
  22,  21,  17,  16,  14,  13,  11, 11, 8,   8,   6,   6,   4,   4,   3,   3,  199, 175, 152, 131, 115, 101, 89, 78,
  70,  61,  54,  47,  42,  36,  31, 28, 24,  22,  19,  17,  15,  14,  12,  12, 8,   8,   7,   7,   5,   5,   3,  3,
  213, 187, 163, 140, 123, 108, 95, 84, 75,  65,  58,  50,  45,  39,  33,  30, 26,  24,  20,  18,  16,  15,  13, 13,
  9,   9,   7,   7,   5,   5,   3,  3,  228, 200, 174, 150, 132, 116, 102, 90, 80,  70,  62,  54,  48,  42,  36, 32,
  28,  26,  22,  20,  18,  16,  14, 14, 10,  10,  8,   8,   6,   6,   4,   4,
};

const int16_t curveTabArrayS4[32] = {
  2512, 2288, 2064, 1840, 1616, 1392, 1168, 944, 720, 560, 464, 368, 272, 208, 144, 80,
  64,   64,   64,   64,   64,   64,   64,   64,  64,  64,  64,  64,  64,  64,  64,  0,
};

const int16_t idxToEcoSet[9][4] = {
    {1, 4, 0, 0},
    {1, 4, -1310, -41},
    {1, 4, 3039, 95},
    {0, 5, 0, 0},
    {2, 6, 962, 30},
    {2, 6, 3039, 95},
    {2, 6, -673, -21},
    {3, 5, 0, 0},
    {3, 5, 962, 30},
};

void SBMPCtx::initState(uint8_t ecoInit)
{
  StoreInitIdx = ecoInit;

  int SH0 = idxToEcoSet[ecoInit][0];
  int SH1 = idxToEcoSet[ecoInit][1];

  S0plusS1 = idxToEcoSet[ecoInit][2];
  S0       = idxToEcoSet[ecoInit][3];

  r = SH0 + 4 + (SH1 << 4);
}

void SBMPCtx::resetState()
{
  S0plusS1 = idxToEcoSet[StoreInitIdx][2];
  S0       = idxToEcoSet[StoreInitIdx][3];
}

void SBMPCtxOptimizer::initStates()
{
  for (int i = 0; i < 9; i++)
  {
    int SH0 = idxToEcoSet[i][0];
    int SH1 = idxToEcoSet[i][1];

    S0plusS1[i] = idxToEcoSet[i][2]; //TO DO -> set Value
    S0[i]       = idxToEcoSet[i][3]; //TO DO -> set Value

    r[i] = SH0 + 4 + (SH1 << 4);
    accBits[i] = 0;    
  }   
}

void SBMPCtxOptimizer::resetStates()
{
  for (int i = 0; i < 9; i++)
  {
    S0plusS1[i] = idxToEcoSet[i][2];
    S0[i]       = idxToEcoSet[i][3];
  }
}

void SBMPCtxOptimizer::updateStates(int32_t minusBin)
{
  int S1 = 0;
  int sign = 0;
  int shiftS0 = 0;
  int shiftS1 = 0;

  for (int i = 0; i < 9; i++)
  {
    S1 = S0plusS1[i] - S0[i] * 16;
    sign = -2 * minusBin - 1;
    shiftS0 = r[i] & 15;
    shiftS1 = r[i] >> 4;
    S0[i] += sign * (curveTabArrayS4[16 + (sign * S0[i] >> 3)] >> shiftS0);
    S1 += sign * (curveTabArrayS4[16 + (sign * S1 >> 7)] >> shiftS1);
    S0plusS1[i] = S1 + 16 * S0[i];
  }
}

uint8_t SBMPCtxOptimizer::getBestIdx( )
{
  uint8_t bestIdx = 0;
  uint64_t minBits = accBits[0];

  uint64_t nzOffBits = 0;

  for (uint8_t i = 0; i < 9; i++)
  {
    nzOffBits = i > 0 ? 32768 * 3 : 0; // add 3 bits for i > 0

    if (accBits[i] + nzOffBits < minBits)
    {
      bestIdx = i;
      minBits = accBits[i] + nzOffBits;
    }
  }

  return bestIdx;
}

void SBMPCtx::updateState( int32_t minusBin )
{
    int S1 = S0plusS1 - S0 * 16;
    int sign = -2 * minusBin - 1;
    int shiftS0 = r & 15;
    int shiftS1 = r >> 4;
    S0 += sign * (curveTabArrayS4[16 + (sign * S0 >> 3)] >> shiftS0);
    S1 += sign * (curveTabArrayS4[16 + (sign * S1 >> 7)] >> shiftS1);
    S0plusS1 = S1 + 16 * S0;
}


uint32_t SBMPCtx::getRLPS( uint32_t range ) const
{
    return rps_table[(abs(S0plusS1 >> 7)) + (range & 0xe0)];
}


int32_t SBMPCtx::getMinusMPS() const
{
    return ~((int32_t)S0plusS1 >> 31);
}

const BinScaledEstBits SBMPScaledEstBits[1024] =
{
  {{   310, 237777}},{{   310, 237777}},{{   313, 237376}},{{   316, 236975}},{{   318, 236574}},{{   321, 236173}},{{   324, 235773}},{{   326, 235372}},
  {{   329, 234971}},{{   332, 234570}},{{   335, 234169}},{{   338, 233769}},{{   341, 233368}},{{   344, 232967}},{{   346, 232566}},{{   349, 232165}},
  {{   352, 231765}},{{   355, 231364}},{{   358, 230963}},{{   362, 230562}},{{   365, 230161}},{{   368, 229761}},{{   371, 229360}},{{   374, 228959}},
  {{   377, 228558}},{{   381, 228157}},{{   384, 227757}},{{   387, 227356}},{{   390, 226955}},{{   394, 226554}},{{   397, 226153}},{{   400, 225753}},
  {{   404, 225352}},{{   407, 224951}},{{   411, 224550}},{{   414, 224149}},{{   418, 223749}},{{   421, 223348}},{{   425, 222947}},{{   429, 222546}},
  {{   432, 222146}},{{   436, 221745}},{{   440, 221344}},{{   444, 220943}},{{   447, 220542}},{{   451, 220142}},{{   455, 219741}},{{   459, 219340}},
  {{   463, 218939}},{{   467, 218538}},{{   471, 218138}},{{   475, 217737}},{{   479, 217336}},{{   483, 216935}},{{   487, 216534}},{{   491, 216134}},
  {{   495, 215733}},{{   500, 215332}},{{   504, 214931}},{{   508, 214530}},{{   513, 214130}},{{   517, 213729}},{{   521, 213328}},{{   526, 212927}},
  {{   530, 212526}},{{   535, 212126}},{{   540, 211725}},{{   544, 211324}},{{   549, 210923}},{{   554, 210522}},{{   558, 210122}},{{   563, 209721}},
  {{   568, 209320}},{{   573, 208919}},{{   578, 208518}},{{   583, 208118}},{{   588, 207717}},{{   593, 207316}},{{   598, 206915}},{{   603, 206514}},
  {{   608, 206114}},{{   613, 205713}},{{   618, 205312}},{{   624, 204911}},{{   629, 204510}},{{   635, 204110}},{{   640, 203709}},{{   645, 203308}},
  {{   651, 202907}},{{   657, 202506}},{{   662, 202106}},{{   668, 201705}},{{   674, 201304}},{{   679, 200903}},{{   685, 200502}},{{   691, 200102}},
  {{   697, 199701}},{{   703, 199300}},{{   709, 198899}},{{   715, 198498}},{{   721, 198098}},{{   727, 197697}},{{   734, 197296}},{{   740, 196895}},
  {{   746, 196494}},{{   753, 196094}},{{   759, 195693}},{{   766, 195292}},{{   772, 194891}},{{   779, 194490}},{{   786, 194090}},{{   792, 193689}},
  {{   799, 193288}},{{   806, 192887}},{{   813, 192486}},{{   820, 192086}},{{   827, 191685}},{{   834, 191284}},{{   841, 190883}},{{   848, 190482}},
  {{   856, 190082}},{{   863, 189681}},{{   870, 189280}},{{   878, 188879}},{{   886, 188478}},{{   893, 188078}},{{   901, 187677}},{{   909, 187276}},
  {{   916, 186875}},{{   924, 186474}},{{   932, 186074}},{{   940, 185673}},{{   948, 185272}},{{   956, 184871}},{{   965, 184470}},{{   973, 184070}},
  {{   981, 183669}},{{   990, 183268}},{{   998, 182867}},{{  1007, 182466}},{{  1016, 182066}},{{  1024, 181665}},{{  1033, 181264}},{{  1042, 180863}},
  {{  1051, 180462}},{{  1060, 180062}},{{  1069, 179661}},{{  1078, 179260}},{{  1088, 178859}},{{  1097, 178458}},{{  1106, 178058}},{{  1116, 177657}},
  {{  1126, 177256}},{{  1135, 176855}},{{  1145, 176454}},{{  1155, 176054}},{{  1165, 175653}},{{  1175, 175252}},{{  1185, 174851}},{{  1195, 174450}},
  {{  1206, 174050}},{{  1216, 173649}},{{  1226, 173248}},{{  1237, 172847}},{{  1248, 172446}},{{  1259, 172046}},{{  1269, 171645}},{{  1280, 171244}},
  {{  1291, 170843}},{{  1303, 170442}},{{  1314, 170042}},{{  1325, 169641}},{{  1337, 169240}},{{  1348, 168839}},{{  1360, 168438}},{{  1372, 168038}},
  {{  1383, 167637}},{{  1395, 167236}},{{  1407, 166835}},{{  1420, 166434}},{{  1432, 166034}},{{  1444, 165633}},{{  1457, 165232}},{{  1469, 164831}},
  {{  1482, 164430}},{{  1495, 164030}},{{  1508, 163629}},{{  1521, 163228}},{{  1534, 162827}},{{  1547, 162426}},{{  1561, 162026}},{{  1574, 161625}},
  {{  1588, 161224}},{{  1601, 160823}},{{  1615, 160422}},{{  1629, 160022}},{{  1643, 159621}},{{  1658, 159220}},{{  1672, 158819}},{{  1687, 158418}},
  {{  1701, 158018}},{{  1716, 157617}},{{  1731, 157216}},{{  1746, 156815}},{{  1761, 156414}},{{  1776, 156014}},{{  1792, 155613}},{{  1807, 155212}},
  {{  1823, 154811}},{{  1839, 154410}},{{  1855, 154010}},{{  1871, 153609}},{{  1887, 153208}},{{  1903, 152807}},{{  1920, 152406}},{{  1937, 152006}},
  {{  1954, 151605}},{{  1971, 151204}},{{  1988, 150803}},{{  2005, 150402}},{{  2022, 150002}},{{  2040, 149601}},{{  2058, 149200}},{{  2076, 148799}},
  {{  2094, 148398}},{{  2112, 147998}},{{  2130, 147597}},{{  2149, 147196}},{{  2168, 146795}},{{  2186, 146395}},{{  2206, 145994}},{{  2225, 145593}},
  {{  2244, 145192}},{{  2264, 144791}},{{  2283, 144391}},{{  2303, 143990}},{{  2324, 143589}},{{  2344, 143188}},{{  2364, 142787}},{{  2385, 142387}},
  {{  2406, 141986}},{{  2427, 141585}},{{  2448, 141184}},{{  2469, 140783}},{{  2491, 140383}},{{  2513, 139982}},{{  2535, 139581}},{{  2557, 139180}},
  {{  2579, 138779}},{{  2602, 138379}},{{  2625, 137978}},{{  2648, 137577}},{{  2671, 137176}},{{  2694, 136775}},{{  2718, 136375}},{{  2742, 135974}},
  {{  2766, 135573}},{{  2790, 135172}},{{  2814, 134771}},{{  2839, 134371}},{{  2864, 133970}},{{  2889, 133569}},{{  2915, 133168}},{{  2940, 132767}},
  {{  2966, 132367}},{{  2992, 131966}},{{  3018, 131565}},{{  3045, 131164}},{{  3072, 130763}},{{  3099, 130363}},{{  3126, 129962}},{{  3153, 129561}},
  {{  3181, 129160}},{{  3209, 128759}},{{  3238, 128359}},{{  3266, 127958}},{{  3295, 127557}},{{  3324, 127156}},{{  3353, 126755}},{{  3383, 126355}},
  {{  3413, 125954}},{{  3443, 125553}},{{  3473, 125152}},{{  3504, 124751}},{{  3535, 124351}},{{  3566, 123950}},{{  3598, 123549}},{{  3630, 123148}},
  {{  3662, 122747}},{{  3694, 122347}},{{  3727, 121946}},{{  3760, 121545}},{{  3793, 121144}},{{  3827, 120743}},{{  3861, 120343}},{{  3895, 119942}},
  {{  3930, 119541}},{{  3965, 119140}},{{  4000, 118739}},{{  4035, 118339}},{{  4071, 117938}},{{  4107, 117537}},{{  4144, 117136}},{{  4181, 116735}},
  {{  4218, 116335}},{{  4256, 115934}},{{  4294, 115533}},{{  4332, 115132}},{{  4371, 114731}},{{  4410, 114331}},{{  4449, 113930}},{{  4489, 113529}},
  {{  4529, 113128}},{{  4569, 112727}},{{  4610, 112327}},{{  4651, 111926}},{{  4693, 111525}},{{  4735, 111124}},{{  4778, 110723}},{{  4820, 110323}},
  {{  4864, 109922}},{{  4907, 109521}},{{  4951, 109120}},{{  4996, 108719}},{{  5041, 108319}},{{  5086, 107918}},{{  5132, 107517}},{{  5178, 107116}},
  {{  5224, 106715}},{{  5271, 106315}},{{  5319, 105914}},{{  5367, 105513}},{{  5415, 105112}},{{  5464, 104711}},{{  5514, 104311}},{{  5563, 103910}},
  {{  5614, 103509}},{{  5664, 103108}},{{  5716, 102707}},{{  5767, 102307}},{{  5820, 101906}},{{  5872, 101505}},{{  5926, 101104}},{{  5980, 100703}},
  {{  6034, 100303}},{{  6089,  99902}},{{  6144,  99501}},{{  6200,  99100}},{{  6256,  98699}},{{  6313,  98299}},{{  6371,  97898}},{{  6429,  97497}},
  {{  6488,  97096}},{{  6547,  96695}},{{  6607,  96295}},{{  6667,  95894}},{{  6728,  95493}},{{  6790,  95092}},{{  6852,  94691}},{{  6915,  94291}},
  {{  6978,  93890}},{{  7042,  93489}},{{  7107,  93088}},{{  7172,  92687}},{{  7238,  92287}},{{  7305,  91886}},{{  7372,  91485}},{{  7440,  91084}},
  {{  7509,  90683}},{{  7578,  90283}},{{  7648,  89882}},{{  7719,  89481}},{{  7790,  89080}},{{  7863,  88679}},{{  7936,  88279}},{{  8009,  87878}},
  {{  8084,  87477}},{{  8159,  87076}},{{  8235,  86675}},{{  8311,  86275}},{{  8389,  85874}},{{  8467,  85473}},{{  8546,  85072}},{{  8626,  84671}},
  {{  8706,  84271}},{{  8788,  83870}},{{  8870,  83469}},{{  8953,  83068}},{{  9037,  82667}},{{  9122,  82267}},{{  9208,  81866}},{{  9294,  81465}},
  {{  9382,  81064}},{{  9470,  80663}},{{  9560,  80263}},{{  9650,  79862}},{{  9741,  79461}},{{  9833,  79060}},{{  9927,  78659}},{{ 10021,  78259}},
  {{ 10116,  77858}},{{ 10212,  77457}},{{ 10309,  77056}},{{ 10407,  76655}},{{ 10507,  76255}},{{ 10607,  75854}},{{ 10708,  75453}},{{ 10811,  75052}},
  {{ 10914,  74651}},{{ 11019,  74251}},{{ 11125,  73850}},{{ 11232,  73449}},{{ 11340,  73048}},{{ 11449,  72647}},{{ 11559,  72247}},{{ 11671,  71846}},
  {{ 11784,  71445}},{{ 11898,  71044}},{{ 12013,  70644}},{{ 12130,  70243}},{{ 12248,  69842}},{{ 12367,  69441}},{{ 12487,  69040}},{{ 12609,  68640}},
  {{ 12732,  68239}},{{ 12857,  67838}},{{ 12983,  67437}},{{ 13110,  67036}},{{ 13239,  66636}},{{ 13369,  66235}},{{ 13501,  65834}},{{ 13634,  65433}},
  {{ 13769,  65032}},{{ 13905,  64632}},{{ 14043,  64231}},{{ 14183,  63830}},{{ 14324,  63429}},{{ 14466,  63028}},{{ 14611,  62628}},{{ 14757,  62227}},
  {{ 14904,  61826}},{{ 15054,  61425}},{{ 15205,  61024}},{{ 15358,  60624}},{{ 15513,  60223}},{{ 15669,  59822}},{{ 15828,  59421}},{{ 15988,  59020}},
  {{ 16150,  58620}},{{ 16314,  58219}},{{ 16481,  57818}},{{ 16649,  57417}},{{ 16819,  57016}},{{ 16991,  56616}},{{ 17166,  56215}},{{ 17342,  55814}},
  {{ 17521,  55413}},{{ 17702,  55012}},{{ 17885,  54612}},{{ 18070,  54211}},{{ 18258,  53810}},{{ 18448,  53409}},{{ 18641,  53008}},{{ 18836,  52608}},
  {{ 19033,  52207}},{{ 19233,  51806}},{{ 19436,  51405}},{{ 19641,  51004}},{{ 19849,  50604}},{{ 20059,  50203}},{{ 20272,  49802}},{{ 20488,  49401}},
  {{ 20707,  49000}},{{ 20929,  48600}},{{ 21154,  48199}},{{ 21381,  47798}},{{ 21612,  47397}},{{ 21846,  46996}},{{ 22083,  46596}},{{ 22323,  46195}},
  {{ 22567,  45794}},{{ 22814,  45393}},{{ 23064,  44992}},{{ 23318,  44592}},{{ 23575,  44191}},{{ 23836,  43790}},{{ 24101,  43389}},{{ 24369,  42988}},
  {{ 24641,  42588}},{{ 24918,  42187}},{{ 25198,  41786}},{{ 25482,  41385}},{{ 25770,  40984}},{{ 26063,  40584}},{{ 26360,  40183}},{{ 26661,  39782}},
  {{ 26967,  39381}},{{ 27278,  38980}},{{ 27593,  38580}},{{ 27913,  38179}},{{ 28238,  37778}},{{ 28569,  37377}},{{ 28904,  36976}},{{ 29244,  36576}},
  {{ 29590,  36175}},{{ 29942,  35774}},{{ 30299,  35373}},{{ 30662,  34972}},{{ 31031,  34572}},{{ 31406,  34171}},{{ 31787,  33770}},{{ 32174,  33369}},
  {{ 32968,  32568}},{{ 33369,  32174}},{{ 33770,  31787}},{{ 34171,  31406}},{{ 34572,  31031}},{{ 34972,  30662}},{{ 35373,  30299}},{{ 35774,  29942}},
  {{ 36175,  29590}},{{ 36576,  29244}},{{ 36976,  28904}},{{ 37377,  28569}},{{ 37778,  28238}},{{ 38179,  27913}},{{ 38580,  27593}},{{ 38980,  27278}},
  {{ 39381,  26967}},{{ 39782,  26661}},{{ 40183,  26360}},{{ 40584,  26063}},{{ 40984,  25770}},{{ 41385,  25482}},{{ 41786,  25198}},{{ 42187,  24918}},
  {{ 42588,  24641}},{{ 42988,  24369}},{{ 43389,  24101}},{{ 43790,  23836}},{{ 44191,  23575}},{{ 44592,  23318}},{{ 44992,  23064}},{{ 45393,  22814}},
  {{ 45794,  22567}},{{ 46195,  22323}},{{ 46596,  22083}},{{ 46996,  21846}},{{ 47397,  21612}},{{ 47798,  21381}},{{ 48199,  21154}},{{ 48600,  20929}},
  {{ 49000,  20707}},{{ 49401,  20488}},{{ 49802,  20272}},{{ 50203,  20059}},{{ 50604,  19849}},{{ 51004,  19641}},{{ 51405,  19436}},{{ 51806,  19233}},
  {{ 52207,  19033}},{{ 52608,  18836}},{{ 53008,  18641}},{{ 53409,  18448}},{{ 53810,  18258}},{{ 54211,  18070}},{{ 54612,  17885}},{{ 55012,  17702}},
  {{ 55413,  17521}},{{ 55814,  17342}},{{ 56215,  17166}},{{ 56616,  16991}},{{ 57016,  16819}},{{ 57417,  16649}},{{ 57818,  16481}},{{ 58219,  16314}},
  {{ 58620,  16150}},{{ 59020,  15988}},{{ 59421,  15828}},{{ 59822,  15669}},{{ 60223,  15513}},{{ 60624,  15358}},{{ 61024,  15205}},{{ 61425,  15054}},
  {{ 61826,  14904}},{{ 62227,  14757}},{{ 62628,  14611}},{{ 63028,  14466}},{{ 63429,  14324}},{{ 63830,  14183}},{{ 64231,  14043}},{{ 64632,  13905}},
  {{ 65032,  13769}},{{ 65433,  13634}},{{ 65834,  13501}},{{ 66235,  13369}},{{ 66636,  13239}},{{ 67036,  13110}},{{ 67437,  12983}},{{ 67838,  12857}},
  {{ 68239,  12732}},{{ 68640,  12609}},{{ 69040,  12487}},{{ 69441,  12367}},{{ 69842,  12248}},{{ 70243,  12130}},{{ 70644,  12013}},{{ 71044,  11898}},
  {{ 71445,  11784}},{{ 71846,  11671}},{{ 72247,  11559}},{{ 72647,  11449}},{{ 73048,  11340}},{{ 73449,  11232}},{{ 73850,  11125}},{{ 74251,  11019}},
  {{ 74651,  10914}},{{ 75052,  10811}},{{ 75453,  10708}},{{ 75854,  10607}},{{ 76255,  10507}},{{ 76655,  10407}},{{ 77056,  10309}},{{ 77457,  10212}},
  {{ 77858,  10116}},{{ 78259,  10021}},{{ 78659,   9927}},{{ 79060,   9833}},{{ 79461,   9741}},{{ 79862,   9650}},{{ 80263,   9560}},{{ 80663,   9470}},
  {{ 81064,   9382}},{{ 81465,   9294}},{{ 81866,   9208}},{{ 82267,   9122}},{{ 82667,   9037}},{{ 83068,   8953}},{{ 83469,   8870}},{{ 83870,   8788}},
  {{ 84271,   8706}},{{ 84671,   8626}},{{ 85072,   8546}},{{ 85473,   8467}},{{ 85874,   8389}},{{ 86275,   8311}},{{ 86675,   8235}},{{ 87076,   8159}},
  {{ 87477,   8084}},{{ 87878,   8009}},{{ 88279,   7936}},{{ 88679,   7863}},{{ 89080,   7790}},{{ 89481,   7719}},{{ 89882,   7648}},{{ 90283,   7578}},
  {{ 90683,   7509}},{{ 91084,   7440}},{{ 91485,   7372}},{{ 91886,   7305}},{{ 92287,   7238}},{{ 92687,   7172}},{{ 93088,   7107}},{{ 93489,   7042}},
  {{ 93890,   6978}},{{ 94291,   6915}},{{ 94691,   6852}},{{ 95092,   6790}},{{ 95493,   6728}},{{ 95894,   6667}},{{ 96295,   6607}},{{ 96695,   6547}},
  {{ 97096,   6488}},{{ 97497,   6429}},{{ 97898,   6371}},{{ 98299,   6313}},{{ 98699,   6256}},{{ 99100,   6200}},{{ 99501,   6144}},{{ 99902,   6089}},
  {{100303,   6034}},{{100703,   5980}},{{101104,   5926}},{{101505,   5872}},{{101906,   5820}},{{102307,   5767}},{{102707,   5716}},{{103108,   5664}},
  {{103509,   5614}},{{103910,   5563}},{{104311,   5514}},{{104711,   5464}},{{105112,   5415}},{{105513,   5367}},{{105914,   5319}},{{106315,   5271}},
  {{106715,   5224}},{{107116,   5178}},{{107517,   5132}},{{107918,   5086}},{{108319,   5041}},{{108719,   4996}},{{109120,   4951}},{{109521,   4907}},
  {{109922,   4864}},{{110323,   4820}},{{110723,   4778}},{{111124,   4735}},{{111525,   4693}},{{111926,   4651}},{{112327,   4610}},{{112727,   4569}},
  {{113128,   4529}},{{113529,   4489}},{{113930,   4449}},{{114331,   4410}},{{114731,   4371}},{{115132,   4332}},{{115533,   4294}},{{115934,   4256}},
  {{116335,   4218}},{{116735,   4181}},{{117136,   4144}},{{117537,   4107}},{{117938,   4071}},{{118339,   4035}},{{118739,   4000}},{{119140,   3965}},
  {{119541,   3930}},{{119942,   3895}},{{120343,   3861}},{{120743,   3827}},{{121144,   3793}},{{121545,   3760}},{{121946,   3727}},{{122347,   3694}},
  {{122747,   3662}},{{123148,   3630}},{{123549,   3598}},{{123950,   3566}},{{124351,   3535}},{{124751,   3504}},{{125152,   3473}},{{125553,   3443}},
  {{125954,   3413}},{{126355,   3383}},{{126755,   3353}},{{127156,   3324}},{{127557,   3295}},{{127958,   3266}},{{128359,   3238}},{{128759,   3209}},
  {{129160,   3181}},{{129561,   3153}},{{129962,   3126}},{{130363,   3099}},{{130763,   3072}},{{131164,   3045}},{{131565,   3018}},{{131966,   2992}},
  {{132367,   2966}},{{132767,   2940}},{{133168,   2915}},{{133569,   2889}},{{133970,   2864}},{{134371,   2839}},{{134771,   2814}},{{135172,   2790}},
  {{135573,   2766}},{{135974,   2742}},{{136375,   2718}},{{136775,   2694}},{{137176,   2671}},{{137577,   2648}},{{137978,   2625}},{{138379,   2602}},
  {{138779,   2579}},{{139180,   2557}},{{139581,   2535}},{{139982,   2513}},{{140383,   2491}},{{140783,   2469}},{{141184,   2448}},{{141585,   2427}},
  {{141986,   2406}},{{142387,   2385}},{{142787,   2364}},{{143188,   2344}},{{143589,   2324}},{{143990,   2303}},{{144391,   2283}},{{144791,   2264}},
  {{145192,   2244}},{{145593,   2225}},{{145994,   2206}},{{146395,   2186}},{{146795,   2168}},{{147196,   2149}},{{147597,   2130}},{{147998,   2112}},
  {{148398,   2094}},{{148799,   2076}},{{149200,   2058}},{{149601,   2040}},{{150002,   2022}},{{150402,   2005}},{{150803,   1988}},{{151204,   1971}},
  {{151605,   1954}},{{152006,   1937}},{{152406,   1920}},{{152807,   1903}},{{153208,   1887}},{{153609,   1871}},{{154010,   1855}},{{154410,   1839}},
  {{154811,   1823}},{{155212,   1807}},{{155613,   1792}},{{156014,   1776}},{{156414,   1761}},{{156815,   1746}},{{157216,   1731}},{{157617,   1716}},
  {{158018,   1701}},{{158418,   1687}},{{158819,   1672}},{{159220,   1658}},{{159621,   1643}},{{160022,   1629}},{{160422,   1615}},{{160823,   1601}},
  {{161224,   1588}},{{161625,   1574}},{{162026,   1561}},{{162426,   1547}},{{162827,   1534}},{{163228,   1521}},{{163629,   1508}},{{164030,   1495}},
  {{164430,   1482}},{{164831,   1469}},{{165232,   1457}},{{165633,   1444}},{{166034,   1432}},{{166434,   1420}},{{166835,   1407}},{{167236,   1395}},
  {{167637,   1383}},{{168038,   1372}},{{168438,   1360}},{{168839,   1348}},{{169240,   1337}},{{169641,   1325}},{{170042,   1314}},{{170442,   1303}},
  {{170843,   1291}},{{171244,   1280}},{{171645,   1269}},{{172046,   1259}},{{172446,   1248}},{{172847,   1237}},{{173248,   1226}},{{173649,   1216}},
  {{174050,   1206}},{{174450,   1195}},{{174851,   1185}},{{175252,   1175}},{{175653,   1165}},{{176054,   1155}},{{176454,   1145}},{{176855,   1135}},
  {{177256,   1126}},{{177657,   1116}},{{178058,   1106}},{{178458,   1097}},{{178859,   1088}},{{179260,   1078}},{{179661,   1069}},{{180062,   1060}},
  {{180462,   1051}},{{180863,   1042}},{{181264,   1033}},{{181665,   1024}},{{182066,   1016}},{{182466,   1007}},{{182867,    998}},{{183268,    990}},
  {{183669,    981}},{{184070,    973}},{{184470,    965}},{{184871,    956}},{{185272,    948}},{{185673,    940}},{{186074,    932}},{{186474,    924}},
  {{186875,    916}},{{187276,    909}},{{187677,    901}},{{188078,    893}},{{188478,    886}},{{188879,    878}},{{189280,    870}},{{189681,    863}},
  {{190082,    856}},{{190482,    848}},{{190883,    841}},{{191284,    834}},{{191685,    827}},{{192086,    820}},{{192486,    813}},{{192887,    806}},
  {{193288,    799}},{{193689,    792}},{{194090,    786}},{{194490,    779}},{{194891,    772}},{{195292,    766}},{{195693,    759}},{{196094,    753}},
  {{196494,    746}},{{196895,    740}},{{197296,    734}},{{197697,    727}},{{198098,    721}},{{198498,    715}},{{198899,    709}},{{199300,    703}},
  {{199701,    697}},{{200102,    691}},{{200502,    685}},{{200903,    679}},{{201304,    674}},{{201705,    668}},{{202106,    662}},{{202506,    657}},
  {{202907,    651}},{{203308,    645}},{{203709,    640}},{{204110,    635}},{{204510,    629}},{{204911,    624}},{{205312,    618}},{{205713,    613}},
  {{206114,    608}},{{206514,    603}},{{206915,    598}},{{207316,    593}},{{207717,    588}},{{208118,    583}},{{208518,    578}},{{208919,    573}},
  {{209320,    568}},{{209721,    563}},{{210122,    558}},{{210522,    554}},{{210923,    549}},{{211324,    544}},{{211725,    540}},{{212126,    535}},
  {{212526,    530}},{{212927,    526}},{{213328,    521}},{{213729,    517}},{{214130,    513}},{{214530,    508}},{{214931,    504}},{{215332,    500}},
  {{215733,    495}},{{216134,    491}},{{216534,    487}},{{216935,    483}},{{217336,    479}},{{217737,    475}},{{218138,    471}},{{218538,    467}},
  {{218939,    463}},{{219340,    459}},{{219741,    455}},{{220142,    451}},{{220542,    447}},{{220943,    444}},{{221344,    440}},{{221745,    436}},
  {{222146,    432}},{{222546,    429}},{{222947,    425}},{{223348,    421}},{{223749,    418}},{{224149,    414}},{{224550,    411}},{{224951,    407}},
  {{225352,    404}},{{225753,    400}},{{226153,    397}},{{226554,    394}},{{226955,    390}},{{227356,    387}},{{227757,    384}},{{228157,    381}},
  {{228558,    377}},{{228959,    374}},{{229360,    371}},{{229761,    368}},{{230161,    365}},{{230562,    362}},{{230963,    358}},{{231364,    355}},
  {{231765,    352}},{{232165,    349}},{{232566,    346}},{{232967,    344}},{{233368,    341}},{{233769,    338}},{{234169,    335}},{{234570,    332}},
  {{234971,    329}},{{235372,    326}},{{235773,    324}},{{236173,    321}},{{236574,    318}},{{236975,    316}},{{237376,    313}},{{237777,    310}},
};

BinScaledEstBits SBMPCtx::getBits() const
{
//    CHECK( 512 + (S0plusS1 >> 3 ) > 1023, "Too large (reg)" );
//    CHECK( 512 + (S0plusS1 >> 3 ) < 0,  "Too small (reg)" );
    return SBMPScaledEstBits[512 + (S0plusS1 >> 3 )];
}

BinScaledEstBits SBMPCtxOptimizer::getBits(uint8_t ecoIdx) const
{
//  CHECK(512 + (S0plusS1[ecoIdx] >> 3) > 1023, "Too large");
//  CHECK(512 + (S0plusS1[ecoIdx] >> 3) < 0, "Too small:");
  return SBMPScaledEstBits[512 + (S0plusS1[ecoIdx] >> 3)];
}

void SBMPCtxOptimizer::accumulateBits( int32_t minusBin )
{
    int32_t bin = -minusBin;
    for(int i = 0; i < 9; i++)
    {
      accBits[i] += getBits(i).scaledEstBits[bin];
    }
}