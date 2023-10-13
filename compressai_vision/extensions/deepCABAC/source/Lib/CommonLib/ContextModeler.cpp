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
#include "ContextModeler.h"


void ContextModeler::init()
{
    neighborWeightVal = 0;
}

void ContextModeler::resetNeighborCtx()
{
    init();
}


int32_t ContextModeler::getSigCtxId( int32_t stateId )
{
  int32_t offset = 3 * stateId;
    int32_t ctxId = 0;

    if (neighborWeightVal != 0)
    {
        ctxId = neighborWeightVal < 0 ? 1 : 2;
    }

    return ctxId+offset;
}

int32_t ContextModeler::getSignFlagCtxId()
{
    int32_t ctxId = 8*3;

    if (neighborWeightVal != 0)
    {
        ctxId += neighborWeightVal < 0 ? 1 : 2;
    }

    return ctxId;
}

int32_t ContextModeler::getMaxCtxId( int32_t stateId )
{
  int32_t offset = 3 * stateId;
    int32_t ctxId = 0;

    if (neighborWeightVal != 0)
    {
        ctxId = neighborWeightVal < 0 ? 1 : 2;
    }

    return ctxId+offset;
}

int32_t ContextModeler::getGtxCtxId( int32_t currWeighVal, uint32_t numGtxFlagsCoded, int32_t stateId )
{
    int32_t offset =  8*3+3;
    int32_t ctxId  = 0;

    ctxId = currWeighVal > 0 ? (numGtxFlagsCoded << 1) : 1 + (numGtxFlagsCoded << 1);

    return (ctxId + offset);
}


void ContextModeler::updateNeighborCtx( int32_t currWeightVal )
{
    neighborWeightVal = currWeightVal;
}
