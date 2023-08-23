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
#ifndef __CABACENC__
#define __CABACENC__

#include "CommonLib/ContextModel.h"
#include "CommonLib/ContextModeler.h"
#include "CommonLib/Quant.h"
#include "CommonLib/Scan.h"
#include "BinEncoder.h"
#include <bitset>
#include <limits>
#include <iostream>

template <typename TBinEnc>
class TCABACEncoder
{
protected:
  __inline void xInitCtxModels(uint32_t numGtxFlags)
  {
    m_NumGtxFlags = numGtxFlags;
    m_CtxStore.resize(8 * 3 + 3 + m_NumGtxFlags * 2 + 32 + 4);
    for (uint32_t ctxId = 0; ctxId < m_CtxStore.size(); ctxId++)
    {
      m_CtxStore[ctxId].initState();
    }
    m_CtxModeler.init();
  }

  __inline void xResetCtxModels()
  {
    for (uint32_t ctxId = 0; ctxId < m_CtxStore.size(); ctxId++)
    {
      m_CtxStore[ctxId].resetState();
    }
  }

  template <uint32_t (TBinEnc::*FuncBinEnc)(uint32_t, SBMPCtx &)>
  __inline uint32_t xEncRemAbs(int32_t value)
  {
    uint32_t scaledBits = 0;
    uint32_t log2NumElemNextGroup = 0;
    int32_t remAbsBaseLevel = 0;
    uint32_t ctxIdx = (8 * 3 + 3 + m_NumGtxFlags * 2);
    if (value > 0)
    {
      scaledBits += (m_BinEncoder.*FuncBinEnc)(1, m_CtxStore[ctxIdx]);
      remAbsBaseLevel += (1 << log2NumElemNextGroup);
      ctxIdx++;
      log2NumElemNextGroup++;
    }
    else
    {
      return (m_BinEncoder.*FuncBinEnc)(0, m_CtxStore[ctxIdx]);
    }
    while (value > (remAbsBaseLevel + (1 << log2NumElemNextGroup) - 1))
    {
      scaledBits += (m_BinEncoder.*FuncBinEnc)(1, m_CtxStore[ctxIdx]);
      remAbsBaseLevel += (1 << log2NumElemNextGroup);
      ctxIdx++;
      log2NumElemNextGroup++;
    }
    scaledBits += (m_BinEncoder.*FuncBinEnc)(0, m_CtxStore[ctxIdx]);
    scaledBits += m_BinEncoder.encodeBinsEP(value - remAbsBaseLevel, log2NumElemNextGroup);
    return scaledBits;
  }

  template <uint32_t (TBinEnc::*FuncBinEnc)(uint32_t, SBMPCtx &)>
  __inline uint32_t xEncWeight(int32_t value, int32_t stateId)
  {
    uint32_t sigFlag = value != 0 ? 1 : 0;
    int32_t sigctx = m_CtxModeler.getSigCtxId(stateId);

    uint32_t scaledBits = (m_BinEncoder.*FuncBinEnc)(sigFlag, m_CtxStore[sigctx]);

    if (sigFlag)
    {
      uint32_t signFlag = value < 0 ? 1 : 0;

      int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
      scaledBits += (m_BinEncoder.*FuncBinEnc)(signFlag, m_CtxStore[signCtx]);

      uint32_t remAbsLevel = abs(value) - 1;
      uint32_t grXFlag = remAbsLevel ? 1 : 0; // greater1
      int32_t ctxIdx = m_CtxModeler.getGtxCtxId(value, 0, stateId);

      scaledBits += (m_BinEncoder.*FuncBinEnc)(grXFlag, m_CtxStore[ctxIdx]);

      uint32_t numGreaterFlagsCoded = 1;

      while (grXFlag && (numGreaterFlagsCoded < m_NumGtxFlags))
      {
        remAbsLevel--;
        grXFlag = remAbsLevel ? 1 : 0;
        ctxIdx = m_CtxModeler.getGtxCtxId(value, numGreaterFlagsCoded, stateId);
        scaledBits += (m_BinEncoder.*FuncBinEnc)(grXFlag, m_CtxStore[ctxIdx]);
        numGreaterFlagsCoded++;
      }

      if (grXFlag)
      {
        remAbsLevel--;
        scaledBits += xEncRemAbs<FuncBinEnc>(remAbsLevel);
      }
    }
    return scaledBits;
  }

protected:
  std::vector<SBMPCtx> m_CtxStore;
  ContextModeler m_CtxModeler;
  TBinEnc m_BinEncoder;
  uint32_t m_NumGtxFlags;
  uint8_t m_ParamOptFlag;
  std::vector<SBMPCtxOptimizer> m_CtxStoreOpt;
};

class CABACEncoder : protected TCABACEncoder<BinEnc>
{
public:
  CABACEncoder() {}
  ~CABACEncoder() {}

  void startCabacEncoding(std::vector<uint8_t> *pBytestream);
  void initCtxMdls(uint32_t numGtxFlags, uint8_t param_opt_flag);
  void resetCtxMdls();

  void initOptimizerCtxMdls(uint32_t numGtxFlags);
  void resetOptimizerMdls();
  void setBestParamsAndInit();
  void pseudoEncodeWeightVal(int32_t value, int32_t stateId, const int32_t maxValue=-1);
  void pseudoEncodeRemAbsLevelNew(uint32_t value);

  void terminateCabacEncoding();
  void iae_v(uint8_t v, int32_t value);
  void uae_v(uint8_t v, uint32_t value);
  int32_t encodeWeights(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, const uint8_t dq_flag, const int32_t scan_order, const int32_t maxValue=-1);

  template <class trellisDef>
  int32_t encodeWeights(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, const uint8_t dq_flag, const int32_t scan_order, const int32_t maxValue=-1);

private:
  __inline void encodeWeightVal(int32_t weightInt, int32_t stateId)
  {
    TCABACEncoder<BinEnc>::xEncWeight<&BinEnc::encodeBin>(weightInt, stateId);
  }
};

#endif // !__CABACENCIF__
