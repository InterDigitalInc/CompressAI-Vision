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

/*------------------------------------------------------------------------------------------
Parts of the original file under above license were modified under the following terms.
They are identified by comments

Copyright (c) 2022-2023, InterDigital Communications, Inc
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
------------------------------------------------------------------------------------------*/

#include "CABACEncoder.h"
#include <iostream>
#include <cstdlib>
#include <cmath>

void CABACEncoder::startCabacEncoding(std::vector<uint8_t> *pBytestream)
{
  m_BinEncoder.setByteStreamBuf(pBytestream);
  m_BinEncoder.startBinEncoder();
}

void CABACEncoder::initCtxMdls(uint32_t numGtxFlags, uint8_t param_opt_flag)
{
  TCABACEncoder<BinEnc>::xInitCtxModels(numGtxFlags);
  initOptimizerCtxMdls(numGtxFlags);

  m_ParamOptFlag = param_opt_flag;
}

void CABACEncoder::resetCtxMdls()
{
  TCABACEncoder<BinEnc>::xResetCtxModels();
}

void CABACEncoder::initOptimizerCtxMdls(uint32_t numGtxFlags)
{
  m_CtxStoreOpt.resize(8 * 3 + 3 + m_NumGtxFlags * 2 + 32 + 4);

  for (uint32_t ctxId = 0; ctxId < m_CtxStoreOpt.size(); ctxId++)
  {
    m_CtxStoreOpt[ctxId].initStates();
  }
}

void CABACEncoder::resetOptimizerMdls()
{
  for (uint32_t ctxId = 0; ctxId < m_CtxStoreOpt.size(); ctxId++)
  {
    m_CtxStoreOpt[ctxId].resetStates();
  }
}

void CABACEncoder::iae_v(uint8_t v, int32_t value)
{
  uint32_t pattern = uint32_t(value) & (uint32_t(0xFFFFFFFF) >> (32 - v));
  m_BinEncoder.encodeBinsEP(pattern, v);
}

void CABACEncoder::uae_v(uint8_t v, uint32_t value)
{
  m_BinEncoder.encodeBinsEP(value, v);
}

void CABACEncoder::setBestParamsAndInit()
{
  for (uint32_t ctxId = 0; ctxId < m_CtxStore.size() - 4; ctxId++)
  {
    uint8_t bestIdx = m_CtxStoreOpt[ctxId].getBestIdx();
    m_CtxStore[ctxId].initState(bestIdx);
  }
}

void CABACEncoder::terminateCabacEncoding()
{
  m_BinEncoder.encodeBinTrm(1);
  m_BinEncoder.finish();
}

void CABACEncoder::pseudoEncodeRemAbsLevelNew(uint32_t value)
{
  int32_t remAbsBaseLevel = 0;
  uint32_t log2NumElemNextGroup = 0;
  uint32_t ctxIdx = (8 * 3 + 3 + m_NumGtxFlags * 2);

  if (value > 0)
  {
    m_BinEncoder.pseudoEncodeBin(1, m_CtxStoreOpt[ctxIdx]);
    remAbsBaseLevel += (1 << log2NumElemNextGroup);
    ctxIdx++;
    log2NumElemNextGroup++;
  }
  else
  {
    m_BinEncoder.pseudoEncodeBin(0, m_CtxStoreOpt[ctxIdx]);
    return;
  }
  while (value > (remAbsBaseLevel + (1 << log2NumElemNextGroup) - 1))
  {
    m_BinEncoder.pseudoEncodeBin(1, m_CtxStoreOpt[ctxIdx]);
    remAbsBaseLevel += (1 << log2NumElemNextGroup);
    ctxIdx++;
    log2NumElemNextGroup++;
  }

  m_BinEncoder.pseudoEncodeBin(0, m_CtxStoreOpt[ctxIdx]);
  // no pseudoEncode of EP bins
}

void CABACEncoder::pseudoEncodeWeightVal(int32_t value, int32_t stateId, const int32_t maxValue)
{
  // beginning of modification
  if (maxValue >0)
  {
    uint32_t maxflag = abs(value) == maxValue ? 1 : 0;
    int32_t maxctx = m_CtxModeler.getMaxCtxId(stateId);

    m_BinEncoder.pseudoEncodeBin(maxctx, m_CtxStoreOpt[maxctx]);

    if (maxflag)
    {
      uint32_t signFlag = value < 0 ? 1 : 0;

      int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
      m_BinEncoder.pseudoEncodeBin(signFlag, m_CtxStoreOpt[signCtx]);
      return;
    }
  }
  // end of modification

  uint32_t sigFlag = value != 0 ? 1 : 0;
  int32_t sigctx = m_CtxModeler.getSigCtxId(stateId);

  m_BinEncoder.pseudoEncodeBin(sigFlag, m_CtxStoreOpt[sigctx]);

  if (sigFlag)
  {
    uint32_t signFlag = value < 0 ? 1 : 0;

    int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
    m_BinEncoder.pseudoEncodeBin(signFlag, m_CtxStoreOpt[signCtx]);

    uint32_t remAbsLevel = abs(value) - 1;
    uint32_t grXFlag = remAbsLevel ? 1 : 0; // greater1
    int32_t ctxIdx = m_CtxModeler.getGtxCtxId(value, 0, stateId);

    m_BinEncoder.pseudoEncodeBin(grXFlag, m_CtxStoreOpt[ctxIdx]);

    uint32_t numGreaterFlagsCoded = 1;

    while (grXFlag && (numGreaterFlagsCoded < m_NumGtxFlags))
    {
      remAbsLevel--;
      grXFlag = remAbsLevel ? 1 : 0;
      ctxIdx = m_CtxModeler.getGtxCtxId(value, numGreaterFlagsCoded, stateId);
      m_BinEncoder.pseudoEncodeBin(grXFlag, m_CtxStoreOpt[ctxIdx]);
      numGreaterFlagsCoded++;
    }

    if (grXFlag)
    {
      remAbsLevel--;
      pseudoEncodeRemAbsLevelNew(remAbsLevel);
    }
  }
}

template <class trellisDef>
int32_t CABACEncoder::encodeWeights(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, const int32_t maxValue)
{
  typename trellisDef::stateTransTab sttab = trellisDef::getStateTransTab();
  m_CtxModeler.resetNeighborCtx();
  int32_t stateId = 0;

  Scan scanIterator(ScanType(scan_order), numWeights, layerWidth);
  if (m_ParamOptFlag)
  {
    for (int i = 0; i < numWeights; i++)
    {
      int32_t value = pWeights[scanIterator.posInMat()];

      if (dq_flag && value != 0)
      {
        value += value < 0 ? -(stateId & 1) : (stateId & 1);
        value >>= 1;
      }

      pseudoEncodeWeightVal(pWeights[scanIterator.posInMat()], stateId, maxValue);
      m_CtxModeler.updateNeighborCtx(pWeights[scanIterator.posInMat()]);

      if (dq_flag)
      {
        stateId = sttab[stateId][value & 1];
      }

      if (scanIterator.isLastPosOfBlockRowButNotLastPosOfBlock())
      {
        resetOptimizerMdls();
        m_CtxModeler.resetNeighborCtx();
      }

      scanIterator++;
    }
  }

  for (int i = 0; i < m_CtxStore.size() - 4; i++)
  {
    if (!dq_flag && (i > 2 && i < 24))
    {
      continue; // skip unused context models, when DQ is disabled
    }
    uint8_t bestEcoIdx = m_CtxStoreOpt[i].getBestIdx();

    m_BinEncoder.encodeBin(bestEcoIdx ? 1 : 0, m_CtxStore[8 * 3 + 3 + m_NumGtxFlags * 2 + 32 + 2]); // second last ctx model

    if (bestEcoIdx != 0)
    {
      m_BinEncoder.encodeBinsEP(bestEcoIdx - 1, 3);
    }
  }

  setBestParamsAndInit();

  m_CtxModeler.resetNeighborCtx();
  stateId = 0;

  scanIterator.resetScan();
  if (scan_order != 0)
  {
    m_BinEncoder.entryPointStart();
  }

  for (int i = 0; i < numWeights; i++)
  {
    int32_t value = pWeights[scanIterator.posInMat()];

    if (dq_flag && value != 0)
    {
      value += value < 0 ? -(stateId & 1) : (stateId & 1);
      value >>= 1;
    }

    encodeWeightVal(value, stateId);

    m_CtxModeler.updateNeighborCtx(value);

    if (dq_flag)
    {
      stateId = sttab[stateId][value & 1];
    }

    if (scanIterator.isLastPosOfBlockRowButNotLastPosOfBlock())
    {
      resetCtxMdls();
      m_CtxModeler.resetNeighborCtx();
      m_BinEncoder.entryPointStart();
    }

    scanIterator++;
  }

  return m_NumGtxFlags;
}

int32_t CABACEncoder::encodeWeights(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, const uint8_t dq_flag, const int32_t scan_order, const int32_t maxValue)
{
  const QuantType qtype = QuantType(dq_flag);

  if (qtype == URQ || qtype == TCQ8States)
  {
    return encodeWeights<Trellis8States>(pWeights, layerWidth, numWeights, dq_flag, scan_order);
  }
  assert(!"Unsupported TCQType");
}
