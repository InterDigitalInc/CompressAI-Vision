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
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <cassert>

#include "Quant.h"
#include "../EncLib/CABACEncoder.h"

namespace TCQ
{
  template <DistType distType>
  class Dist
  {
  public:
    Dist(double scale) : dscale(scale) {}
    double setOrg(double org)
    {
      orgval = org;
      return dscale * dist(orgval);
    }
    double operator()(double rec) { return dscale * dist(orgval - rec); }

  private:
    double dist(double diff)
    {
      if (distType == DIST_MSE)
      {
        return diff * diff;
      }
      return 0.0;
    };

  private:
    const double dscale;
    double orgval;
  };

  struct QData
  {
    double dist;
    int32_t level;
  };

  template <DistType distType>
  class PreQuant
  {
  public:
    PreQuant(double qstep, double lambdaFactor)
        : dist(lambdaFactor <= 0.0 ? 1.0 : double(1 << 15) / lambdaFactor), qscale(1.0 / qstep)
    {
    }
    std::array<QData, 5> operator()(double value)
    {
      std::array<QData, 5> qdarray;
      const int32_t sign = (value < 0 ? -1 : 1);
      const double org = qscale * value * double(sign);
      const int32_t minQIdx = std::max(1, (int32_t)org - 1);
      QData &qdataZero = qdarray[4];
      qdataZero.level = 0;
      qdataZero.dist = dist.setOrg(org);
      for (int32_t qidx = minQIdx; qidx < minQIdx + 4;)
      {
        QData &qdata = qdarray[qidx & 3];
        qdata.dist = dist(qidx);
        qdata.level = sign * (++qidx >> 1);
      }
      return qdarray;
    }

  private:
    Dist<distType> dist;
    double qscale;
  };

  struct Decision
  {
    int32_t level;
    int32_t prevId;
  };

  template <class rateEst>
  class State
  {
  public:
    State() : getRate(nullptr), rdCost(0.0) {}
    ~State() { delete getRate; }
    void init(int32_t stateId, const typename rateEst::pars &repar)
    {
      delete getRate;
      getRate = new rateEst(stateId, repar);
      rdCost = 0.5 * std::numeric_limits<double>::max();
    }
    void setStart() { rdCost = 0.0; }
    void update(const State &prevState, int32_t level, double cost)
    {
      getRate->copyCtx(prevState.getRate);
      getRate->updateCtx(level);
      rdCost = cost;
    }
    double getCost() const { return rdCost; }
    double getCost(const QData &qdata) const
    {
      return rdCost + qdata.dist + (*getRate)(qdata.level);
    }

  private:
    rateEst *getRate;
    double rdCost;
  };

  template <class trellisDef, DistType distType, class rateEst>
  class Trellis
  {
  private:
    typedef std::array<Decision, std::tuple_size<typename trellisDef::stateTransTab>::value> decArray;
    struct Branch
    {
      int prevId;
      int qindex;
    };

  public:
    Trellis(double qstep, double lambdaFactor, const typename rateEst::pars &repar) : quant(qstep, lambdaFactor)
    {
      // compile checks for correct state transition table sizes
#define IS_POWER_OF_TWO(n) (n && !(n & (n - 1)))
      static_assert(IS_POWER_OF_TWO(std::tuple_size<typename trellisDef::stateTransTab>::value), "Trellis::Trellis<def>: array size must be a power of 2");
      static_assert(std::tuple_size<typename trellisDef::stateTransTab::value_type>::value == 2, "Trellis::Trellis<def>: inner array size must be equal to 2");
#undef IS_POWER_OF_TWO

      //===== set connections =====
      {
        typename trellisDef::stateTransTab sttab = trellisDef::getStateTransTab();
        connections.resize(sttab.size());
        for (int32_t prevId = 0; prevId < (int32_t)sttab.size(); prevId++)
        {
          const int32_t quantId = prevId & 1;       // quantizer id (0 or 1)
          const int32_t qindex0 = 3 * quantId;      // quantization index (factor of step size) for parity 0
          const int32_t qindex1 = 2 - quantId;      // quantization index (factor of step size) for parity 1
          const int32_t currId0 = sttab[prevId][0]; // preceding state for parity 0
          const int32_t currId1 = sttab[prevId][1]; // preceding state for parity 1
          assert(currId0 >= 0 && currId0 < std::tuple_size<typename trellisDef::stateTransTab>::value && currId1 >= 0 && currId1 < std::tuple_size<typename trellisDef::stateTransTab>::value);
          connections[currId0].push_back({prevId, qindex0});
          connections[currId0].push_back({prevId, 4}); // special for "0" (see PreQuant)
          connections[currId1].push_back({prevId, qindex1});
        }
      }

      //===== init states =====
      prev.resize(std::tuple_size<typename trellisDef::stateTransTab>::value);
      curr.resize(std::tuple_size<typename trellisDef::stateTransTab>::value);
      for (int32_t stateId = 0; stateId < (int32_t)std::tuple_size<typename trellisDef::stateTransTab>::value; stateId++)
      {
        prev[stateId].init(stateId, repar);
        curr[stateId].init(stateId, repar);
      }
      curr[0].setStart();
    }

    decArray decideUpdate(double value)
    {
      decArray decarray;
      std::array<QData, 5> qdata = quant(value);

      curr.swap(prev);
      for (int32_t currId = 0; currId < (int32_t)connections.size(); currId++)
      {
        const std::vector<Branch> &branches = connections[currId];
        Decision &decision = decarray[currId];
        double minCost = std::numeric_limits<double>::max(), cost;
        for (const auto &branch : branches)
        {
          if ((cost = prev[branch.prevId].getCost(qdata[branch.qindex])) < minCost)
          {
            minCost = cost;
            decision = {qdata[branch.qindex].level, branch.prevId};
          }
        }
        curr[currId].update(prev[decision.prevId], decision.level, minCost);
      }
      return decarray;
    }

    int32_t getMinCostPathId()
    {
      int32_t bestId = 0;
      double minCost = curr[bestId].getCost(), cost;
      for (int32_t id = 1; id < (int32_t)connections.size(); id++)
      {
        if ((cost = curr[id].getCost()) < minCost)
        {
          minCost = cost;
          bestId = id;
        }
      }
      return bestId;
    }

  private:
    PreQuant<distType> quant;
    std::vector<std::vector<Branch>> connections;
    std::vector<State<rateEst>> prev;
    std::vector<State<rateEst>> curr;
  };

  template <class trellisDef, DistType distType, class rateEst>
  class TCQ
  {
  public:
    typedef typename rateEst::pars pars;
    static uint32_t quant(const float32_t *weights, int32_t *level, const int32_t numTotal, const uint32_t stride, const double qstep, const double lambdaFactor, const pars &rateEstPars, const int32_t scan_order)
    {
      // init, populate trellis
      Trellis<trellisDef, distType, rateEst> trellis(qstep, lambdaFactor, rateEstPars);
      std::vector<decArray> decisions;
      decisions.reserve(numTotal);

      Scan scanIterator(ScanType(scan_order), numTotal, stride);

      const float32_t scale = float32_t(1.0) / qstep;

      for (int i = 0; i < numTotal; i++)
      {
        // CHECK for Possible int32_t overflow
        double scaledVal = round(weights[scanIterator.posInMat()] * scale);
        if (int64_t(scaledVal) > ((int64_t(1) << 31) - 3) || int64_t(scaledVal) < (-(int64_t(1) << 31) - 2))
        {
          return 0;
        }
        decisions.push_back(trellis.decideUpdate(weights[scanIterator.posInMat()]));
        scanIterator++;
      }
      // backward scanning and write back
      int32_t stateId = trellis.getMinCostPathId();

      // scanIterator.resetScan();

      // for (int i = 0; i < numTotal; i++)
      for (int32_t k = numTotal - 1; k >= 0; k--)
      {
        const Decision &dec = decisions[k][stateId];
        level[scanIterator.posInMat()] = dec.level;
        stateId = dec.prevId;
        if (dec.level != 0)
        {
          level[scanIterator.posInMat()] <<= 1;
          level[scanIterator.posInMat()] += level[scanIterator.posInMat()] < 0 ? (stateId & 1) : -(stateId & 1);
        }
        scanIterator--;
      }
      return 1;
    }

  private:
    typedef std::array<Decision, std::tuple_size<typename trellisDef::stateTransTab>::value> decArray;
  };
};

// rate estimators
class IgnoreRate
{
public:
  struct pars
  {
  };

public:
  // the constructor and the functions must have exactly this form
  IgnoreRate(int32_t stateId, const pars &p) {}
  void copyCtx(const IgnoreRate *other) {}
  void updateCtx(int32_t level) {}
  double operator()(int32_t level) { return 0.0; }
};

class CabacRate : protected TCABACEncoder<BinEst>
{
public:
  struct pars
  {
    // int layerwidth;
    uint32_t maxNumNoRem;
  };

public:
  // the constructor and the functions must have exactly this form
  CabacRate(int32_t stateId, const pars &p)
      : m_stateId(stateId)
  {
    TCABACEncoder<BinEst>::xInitCtxModels(p.maxNumNoRem);
    m_CtxModeler.resetNeighborCtx();
  }
  void copyCtx(const CabacRate *other)
  {
    m_CtxStore = other->m_CtxStore;
  }
  void updateCtx(int32_t level)
  {
    TCABACEncoder<BinEst>::xEncWeight<&BinEst::updateBin>(level, m_stateId);
    m_CtxModeler.updateNeighborCtx(level);
  }
  double operator()(int32_t level)
  {
    return (double)TCABACEncoder<BinEst>::xEncWeight<&BinEst::encodeBin>(level, m_stateId);
  }

private:
  const int32_t m_stateId;
};

template <class trellisDef, DistType distType>
uint32_t quantizeTCQ(const float32_t *weights, int32_t *level, const float32_t qstep, const int32_t stride, const int32_t numTotal, const double lambdaFactor, const uint32_t maxNumNoRem, const int32_t scan_order)
{
  if (lambdaFactor <= 0.0)
  {
    return TCQ::TCQ<trellisDef, distType, IgnoreRate>::quant(weights, level, numTotal, stride, qstep, 0.0, {}, scan_order);
  }
  return TCQ::TCQ<trellisDef, distType, CabacRate>::quant(weights, level, numTotal, stride, qstep, lambdaFactor, {maxNumNoRem}, scan_order);
}

template <class trellisDef>
uint32_t quantizeTCQ(const float32_t *weights, int32_t *level, const float32_t qstep, const int32_t stride, const int32_t numTotal, const DistType distType, const double lambdaScale, const uint32_t maxNumNoRem, const int32_t scan_order)
{
  if (distType == DIST_MSE)
  {
    const double lambdaFactor = lambdaScale * 4.0 * (log(2.) / 6.) * 0.7; // the last 0.7 seems to be a special TCQ thing
    return quantizeTCQ<trellisDef, DIST_MSE>(weights, level, qstep, stride, numTotal, lambdaFactor, maxNumNoRem, scan_order);
  }
  assert(!"Unsupported DistType");
  return 0;
}

uint32_t quantizeFeatures(float32_t *features, int32_t *level, const float32_t qstep, const int32_t stride, const int32_t numTotal, const int32_t scan_order)
{

  Scan scanIterator(ScanType(scan_order), numTotal, stride);
  int32_t maxValue = -1;
  const float32_t scale = float32_t(1.0) / qstep;
  for (int i = 0; i < numTotal; i++)
  {
    double scaledVal = round(scale * features[scanIterator.posInMat()]);

    if (int64_t(scaledVal) > (int64_t(1) << 31) - 1 || int64_t(scaledVal) < (-int64_t(1) << 31))
    {
      return 0; // check for int32_t overflow
    }
    // TODO fracape: that's the max per tensor
    if (scaledVal > maxValue)
    {
      maxValue = std::abs(scaledVal);
    }
    level[scanIterator.posInMat()] = (int32_t)scaledVal;
    scanIterator++;
  }
  assert(maxValue > 0);
  return maxValue;
}

template <DistType distType>
uint32_t quantizeURQ(float32_t *weights, int32_t *level, const float32_t qstep, const int32_t stride, const int32_t numTotal, double lambdaScale, uint32_t maxNumNoRem, const int32_t scan_order)
{

  Scan scanIterator(ScanType(scan_order), numTotal, stride);

  if (lambdaScale <= 0.0)
  {
    const float32_t scale = float32_t(1.0) / qstep;
    for (int i = 0; i < numTotal; i++)
    {
      double scaledVal = round(scale * weights[scanIterator.posInMat()]);

      if (int64_t(scaledVal) > (int64_t(1) << 31) - 1 || int64_t(scaledVal) < (-int64_t(1) << 31))
      {
        return 0; // check for int32_t overflow
      }

      level[scanIterator.posInMat()] = (int32_t)scaledVal;
      scanIterator++;
    }
    return 1;
  }

  const double lambdaFactor = lambdaScale * (distType == DIST_MSE ? (log(2.) / 6.) : 1.);
  const double distScale = double(1 << 15) / lambdaFactor;
  const double qscale = 1.0 / qstep;
  TCQ::Dist<distType> dist(distScale);
  CabacRate rateEst(0, {maxNumNoRem});
  for (int i = 0; i < numTotal; i++)
  {
    const double scaledVal = round(weights[scanIterator.posInMat()] * qscale);
    if (int64_t(scaledVal) > (int64_t(1) << 31) - 1 || int64_t(scaledVal) < (-int64_t(1) << 31))
    {
      return 0; // check for int32_t overflow
    }

    const int sign = (weights[scanIterator.posInMat()] < 0 ? -1 : 1);
    const double absw = double(sign) * double(weights[scanIterator.posInMat()]) * qscale;
    int32_t bestIdx = 0;
    double minCost = dist.setOrg(absw) + rateEst(0);
    int32_t maxQIdx = (int32_t)round(absw);
    for (int32_t qIdx = std::max<int32_t>(1, maxQIdx - 1); qIdx <= maxQIdx; qIdx++)
    {
      double cost = dist(qIdx) + rateEst(qIdx);
      if (cost < minCost)
      {
        bestIdx = qIdx;
        minCost = cost;
      }
    }
    bestIdx *= sign;
    level[scanIterator.posInMat()] = bestIdx;
    rateEst.updateCtx(bestIdx);
    scanIterator++;
  }
  return 1;
}

uint32_t quantizeURQ(float32_t *weights, int32_t *level, const float32_t qstep, const int32_t stride, const int32_t numTotal, const DistType distType, double lambdaScale, uint32_t maxNumNoRem, const int32_t scan_order)
{
  if (distType == DIST_MSE)
  {
    return quantizeURQ<DIST_MSE>(weights, level, qstep, stride, numTotal, lambdaScale, maxNumNoRem, scan_order);
  }
  assert(!"Unsupported DistType");
  return 0;
}

uint32_t quantize(float32_t *weights, int32_t *level, const float32_t qstep, const int32_t stride, const int32_t numTotal, const DistType distType, double lambdaScale, const uint8_t dq_flag, const uint32_t maxNumNoRem, const int32_t scan_order)
{
  assert(weights && qstep > 0.0 && stride > 0);

  const QuantType qtype = QuantType(dq_flag);

  if (qtype == URQ)
  {
    return quantizeURQ(weights, level, qstep, stride, numTotal, distType, lambdaScale, maxNumNoRem, scan_order);
  }
  if (qtype == TCQ8States)
  {
    return quantizeTCQ<Trellis8States>(weights, level, qstep, stride, numTotal, distType, lambdaScale, maxNumNoRem, scan_order);
  }
  assert(!"Unsupported TCQType");
  return 0;
}

void deQuantize(float32_t *weights, int32_t *level, const float32_t qstep, const uint32_t numWeights, const int32_t stride, const int32_t scan_order)
{
  assert(weights && level && qstep > 0.0 && stride > 0);

  Scan scanIterator(ScanType(scan_order), numWeights, stride);

  for (uint32_t i = 0; i < numWeights; i++)
  {
    weights[scanIterator.posInMat()] = qstep * float32_t(level[scanIterator.posInMat()]);
    scanIterator++;
  }
}
