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

#include "CABACDecoder.h"
#include <iostream>

void CABACDecoder::startCabacDecoding(uint8_t *pBytestream)
{
  m_BinDecoder.setByteStreamBuf(pBytestream);
  m_BinDecoder.startBinDecoder();
}

void CABACDecoder::initCtxMdls(uint32_t cabac_unary_length)
{
  m_NumGtxFlags = cabac_unary_length;

  m_CtxStore.resize(8 * 3 + 3 + m_NumGtxFlags * 2 + 32 + 4);

  for (uint32_t ctxId = 0; ctxId < m_CtxStore.size(); ctxId++)
  {
    m_CtxStore[ctxId].initState();
  }
  m_CtxModeler.init();
}

void CABACDecoder::resetCtxMdls()
{
  for (uint32_t ctxId = 0; ctxId < m_CtxStore.size(); ctxId++)
  {
    m_CtxStore[ctxId].resetState();
  }
}

int32_t CABACDecoder::iae_v(uint8_t v)
{
  uint32_t pattern = m_BinDecoder.decodeBinsEP(v);
  return int32_t(pattern << (32 - v)) >> (32 - v);
}

uint32_t CABACDecoder::uae_v(uint8_t v)
{
  return m_BinDecoder.decodeBinsEP(v);
}

template <class trellisDef>
void CABACDecoder::decodeWeights(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, const int32_t maxValue)
{
  typename trellisDef::stateTransTab sttab = trellisDef::getStateTransTab();

  for (int i = 0; i < m_CtxStore.size() - 4; i++)
  {
    if (!dq_flag && (i > 2 && i < 24))
    {
      continue; // skip unused context models, when DQ is disabled
    }

    uint8_t bestEcoIdx = 0;

    if (m_BinDecoder.decodeBin(m_CtxStore[8 * 3 + 3 + m_NumGtxFlags * 2 + 32 + 2]))
    {
      bestEcoIdx += (1 + m_BinDecoder.decodeBinsEP(3));
    }

    m_CtxStore[i].initState(bestEcoIdx);
  }

  m_CtxModeler.resetNeighborCtx();
  int32_t stateId = 0;

  Scan scanIterator(ScanType(scan_order), numWeights, layerWidth);

  if (scan_order != 0)
  {
    if (true)
    {
      uint32_t bytesReadBefore = m_BinDecoder.getBytesRead();
      uint8_t *byteStreamPtrBefore = m_BinDecoder.getByteStreamPtr();
      uint8_t *byteStreamPtrAfter = nullptr;
      EntryPoint firstEp = m_BinDecoder.getEntryPoint();
      firstEp.dqState = stateId;

      uint64_t lastBitOffset = firstEp.totalBitOffset;
      // convert from differential entry points to absolute entry points
      for (int epIdx = 0; epIdx < m_EntryPoints.size(); epIdx++)
      {
        EntryPoint ep;
        ep.setEntryPointInt(m_EntryPoints[epIdx]);
        ep.totalBitOffset += lastBitOffset;
        lastBitOffset = ep.totalBitOffset;
        m_EntryPoints[epIdx] = ep.getEntryPointInt();
      }
      m_EntryPoints.insert(m_EntryPoints.begin(), firstEp.getEntryPointInt());

      EntryPoint finalEntryPoint;

      for (int epIdx = m_EntryPoints.size() - 1; epIdx >= 0; epIdx--)
      {
        // int epIdx2 = epIdx == -1 ? m_EntryPoints.size() - 1 : epIdx;
        scanIterator.seekBlockRow(epIdx);
        EntryPoint ep;
        ep.setEntryPointInt(m_EntryPoints[epIdx]);
        m_BinDecoder.setEntryPoint(ep);
        stateId = ep.dqState;
        resetCtxMdls();
        m_CtxModeler.resetNeighborCtx();

        while (true)
        {
          pWeights[scanIterator.posInMat()] = 0;
          decodeWeightVal(pWeights[scanIterator.posInMat()], stateId, maxValue);
          m_CtxModeler.updateNeighborCtx(pWeights[scanIterator.posInMat()]);
          if (dq_flag)
          {
            int32_t newState = sttab[stateId][pWeights[scanIterator.posInMat()] & 1];

            if (pWeights[scanIterator.posInMat()] != 0)
            {
              pWeights[scanIterator.posInMat()] <<= 1;
              pWeights[scanIterator.posInMat()] += pWeights[scanIterator.posInMat()] < 0 ? (stateId & 1) : -(stateId & 1);
            }

            stateId = newState;
          }
          if (scanIterator.isLastPosOfBlockRow())
          {
            if (epIdx == m_EntryPoints.size() - 1) // last Entry Point
            {
              finalEntryPoint = m_BinDecoder.getEntryPoint();
              byteStreamPtrAfter = m_BinDecoder.getByteStreamPtr();
            }
            break;
          }
          scanIterator++;
        }
      }

      m_BinDecoder.setByteStreamPtr(byteStreamPtrAfter);
      m_BinDecoder.setBytesRead(bytesReadBefore + (byteStreamPtrAfter - byteStreamPtrBefore));
      m_BinDecoder.setEntryPointWithRange(finalEntryPoint);
    }
    else
    {
      CHECK(true, "Entry point vector is empty!");
    }
  }
  else
  {
    for (int i = 0; i < numWeights; i++)
    {
      pWeights[scanIterator.posInMat()] = 0;
      decodeWeightVal(pWeights[scanIterator.posInMat()], stateId, maxValue);

      m_CtxModeler.updateNeighborCtx(pWeights[scanIterator.posInMat()]);

      if (dq_flag)
      {
        int32_t newState = sttab[stateId][pWeights[scanIterator.posInMat()] & 1];

        if (pWeights[scanIterator.posInMat()] != 0)
        {
          pWeights[scanIterator.posInMat()] <<= 1;
          pWeights[scanIterator.posInMat()] += pWeights[scanIterator.posInMat()] < 0 ? (stateId & 1) : -(stateId & 1);
        }

        stateId = newState;
      }

      scanIterator++;
    }
  }
}

void CABACDecoder::decodeWeights(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, const int32_t maxValue)
{
  const QuantType qtype = QuantType(dq_flag);

  if (qtype == URQ || qtype == TCQ8States)
  {
    return decodeWeights<Trellis8States>(pWeights, layerWidth, numWeights, dq_flag, scan_order);
  }
  assert(!"Unsupported TCQType");
}

template <class trellisDef>
void CABACDecoder::decodeWeightsAndCreateEPs(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, std::vector<uint64_t> &entryPoints, const int32_t maxValue)
{
  typename trellisDef::stateTransTab sttab = trellisDef::getStateTransTab();

  for (int i = 0; i < m_CtxStore.size() - 4; i++)
  {
    if (!dq_flag && (i > 2 && i < 24))
    {
      continue; // skip unused context models, when DQ is disabled
    }

    uint8_t bestEcoIdx = 0;

    if (m_BinDecoder.decodeBin(m_CtxStore[8 * 3 + 3 + m_NumGtxFlags * 2 + 32 + 2]))
    {
      bestEcoIdx += (1 + m_BinDecoder.decodeBinsEP(3));
    }

    m_CtxStore[i].initState(bestEcoIdx);
  }

  m_CtxModeler.resetNeighborCtx();
  int32_t stateId = 0;

  Scan scanIterator(ScanType(scan_order), numWeights, layerWidth);

  uint64_t lastBitOffset = 0;
  if (scan_order != 0)
  {
    m_BinDecoder.entryPointStart();
    EntryPoint ep = m_BinDecoder.getEntryPoint();
    ep.dqState = stateId;
    lastBitOffset = ep.totalBitOffset;
  }
  for (int i = 0; i < numWeights; i++)
  {
    pWeights[scanIterator.posInMat()] = 0;
    decodeWeightVal(pWeights[scanIterator.posInMat()], stateId, maxValue);

    m_CtxModeler.updateNeighborCtx(pWeights[scanIterator.posInMat()]);

    if (dq_flag)
    {
      int32_t newState = sttab[stateId][pWeights[scanIterator.posInMat()] & 1];

      if (pWeights[scanIterator.posInMat()] != 0)
      {
        pWeights[scanIterator.posInMat()] <<= 1;
        pWeights[scanIterator.posInMat()] += pWeights[scanIterator.posInMat()] < 0 ? (stateId & 1) : -(stateId & 1);
      }

      stateId = newState;
    }
    if (scanIterator.isLastPosOfBlockRowButNotLastPosOfBlock())
    {
      resetCtxMdls();
      m_CtxModeler.resetNeighborCtx();
      m_BinDecoder.entryPointStart();
      EntryPoint ep = m_BinDecoder.getEntryPoint();
      ep.dqState = stateId;
      uint64_t deltaOffset = ep.totalBitOffset - lastBitOffset;
      lastBitOffset = ep.totalBitOffset;
      ep.totalBitOffset = deltaOffset;
      entryPoints.push_back(ep.getEntryPointInt());
    }

    scanIterator++;
  }
}

void CABACDecoder::decodeWeightsAndCreateEPs(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, std::vector<uint64_t> &entryPoints, const int32_t maxValue)
{
  const QuantType qtype = QuantType(dq_flag);

  if (qtype == URQ || qtype == TCQ8States)
  {
    return decodeWeightsAndCreateEPs<Trellis8States>(pWeights, layerWidth, numWeights, dq_flag, scan_order, entryPoints, maxValue);
  }
  assert(!"Unsupported TCQType");
}

void CABACDecoder::setEntryPoints(uint64_t *pEntryPoints, uint64_t numEntryPoints)
{
  m_EntryPoints.resize(numEntryPoints);

  for (int i = 0; i < m_EntryPoints.size(); i++)
  {
    m_EntryPoints[i] = pEntryPoints[i];
  }
}

void CABACDecoder::decodeWeightVal(int32_t &decodedIntVal, int32_t stateId, const int32_t maxValue)
{
  // beginning of modification
  if (maxValue >0)
  {
    int32_t maxctx = m_CtxModeler.getMaxCtxId(stateId);
    uint32_t maxFlag = m_BinDecoder.decodeBin(m_CtxStore[maxctx]);

    if (maxFlag)
    {
      decodedIntVal++;
      uint32_t signFlag = 0;
      int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
      signFlag = m_BinDecoder.decodeBin(m_CtxStore[signCtx]);

      decodedIntVal = signFlag ? -maxValue : maxValue;
      return;
    }
  }
  // end of modification

  int32_t sigctx = m_CtxModeler.getSigCtxId(stateId);
  uint32_t sigFlag = m_BinDecoder.decodeBin(m_CtxStore[sigctx]);

  if (sigFlag)
  {
    decodedIntVal++;
    uint32_t signFlag = 0;
    int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
    signFlag = m_BinDecoder.decodeBin(m_CtxStore[signCtx]);

    int32_t intermediateVal = signFlag ? -1 : 1;

    int32_t ctxIdx = m_CtxModeler.getGtxCtxId(intermediateVal, 0, stateId);
    uint32_t grXFlag = 0;

    grXFlag = m_BinDecoder.decodeBin(m_CtxStore[ctxIdx]); // greater1

    uint32_t numGreaterFlagsDecoded = 1;

    while (grXFlag && (numGreaterFlagsDecoded < m_NumGtxFlags))
    {
      decodedIntVal++;
      ctxIdx = m_CtxModeler.getGtxCtxId(intermediateVal, numGreaterFlagsDecoded, stateId);
      grXFlag = m_BinDecoder.decodeBin(m_CtxStore[ctxIdx]);
      numGreaterFlagsDecoded++;
    }

    if (grXFlag)
    {
      decodedIntVal++;
      uint32_t remAbsLevel = decodeRemAbsLevel();
      decodedIntVal += remAbsLevel;
    }
    decodedIntVal = signFlag ? -decodedIntVal : decodedIntVal;
  }
}

int32_t CABACDecoder::decodeRemAbsLevel()
{
  int32_t remAbsLevel = 0;
  uint32_t log2NumElemNextGroup = 0;
  uint32_t ctxIdx = (8 * 3 + 3 + m_NumGtxFlags * 2);

  if (m_BinDecoder.decodeBin(m_CtxStore[ctxIdx]))
  {
    remAbsLevel += (1 << log2NumElemNextGroup);
    ctxIdx++;
    log2NumElemNextGroup++;
  }
  else
  {
    return remAbsLevel;
  }

  while (m_BinDecoder.decodeBin(m_CtxStore[ctxIdx]))
  {
    remAbsLevel += (1 << log2NumElemNextGroup);
    ctxIdx++;
    log2NumElemNextGroup++;
  }

  remAbsLevel += (int32_t)m_BinDecoder.decodeBinsEP(log2NumElemNextGroup);
  return remAbsLevel;
}

uint32_t CABACDecoder::getBytesRead()
{
  return m_BinDecoder.getBytesRead();
}

uint32_t CABACDecoder::terminateCabacDecoding()
{
  if (m_BinDecoder.decodeBinTrm())
  {
    m_BinDecoder.finish();
    return m_BinDecoder.getBytesRead();
  }
  CHECK(1, "Terminating Bin not received!");
}
