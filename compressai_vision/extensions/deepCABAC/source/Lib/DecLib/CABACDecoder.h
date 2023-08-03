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
#ifndef __CABACDEC__
#define __CABACDEC__

#include <vector>

#include "CommonLib/ContextModel.h"
#include "CommonLib/ContextModeler.h"
#include "CommonLib/Quant.h"
#include "CommonLib/Scan.h"
#include "BinDecoder.h"

class CABACDecoder
{
public:
    CABACDecoder() {}
    ~CABACDecoder() {}

    void     startCabacDecoding    ( uint8_t* pBytestream );
    void     initCtxMdls           ( uint32_t cabac_unary_length );
    void     resetCtxMdls          ();
    uint32_t terminateCabacDecoding();
    int32_t  iae_v                 ( uint8_t v );
    uint32_t uae_v                 ( uint8_t v );
    void decodeWeights             ( int32_t* pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order );

    template <class trellisDef>
    void decodeWeights             (int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order);

    void decodeWeightsAndCreateEPs(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, std::vector<uint64_t>& entryPoints);

    template <class trellisDef>
    void decodeWeightsAndCreateEPs(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, std::vector<uint64_t> &entryPoints);

    void setEntryPoints           (uint64_t* pEntryPoints, uint64_t numEntryPoints);

uint32_t
getBytesRead();

protected:
    void decodeWeightVal           ( int32_t &decodedIntVal, int32_t stateId  );
    int32_t decodeRemAbsLevel      ();

private:
    std::vector<SBMPCtx>  m_CtxStore;
    ContextModeler        m_CtxModeler;
    BinDec                m_BinDecoder;
    uint32_t              m_NumGtxFlags;
    std::vector<uint64_t> m_EntryPoints;
};
#endif // __CABACDEC__
