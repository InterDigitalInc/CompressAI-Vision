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
#ifndef __CONTEXTMODEL__
#define __CONTEXTMODEL__

#include "TypeDef.h"

struct BinScaledEstBits
{
    uint32_t scaledEstBits[ 2 ];
};

class SBMPCtx
{
public:
    void initState(uint8_t ecoInit = 0);
    void resetState();
    void updateState( int32_t minusBin );
    uint32_t getRLPS( uint32_t range ) const;
    int32_t getMinusMPS() const;
    BinScaledEstBits getBits() const;
private:
    int8_t  S0;
    int16_t S0plusS1;
    uint8_t r;
    uint8_t StoreInitIdx;
};

class SBMPCtxOptimizer
{
public:
    void initStates();
    void resetStates();
    void updateStates( int32_t minusBin );
    
    void accumulateBits( int32_t minusBin );

    uint8_t getBestIdx( );
    BinScaledEstBits getBits(uint8_t ecoIdx) const;

private:
    int8_t S0[9];
    int16_t S0plusS1[9];
    uint8_t r[9];
    uint64_t accBits[9];
};

#endif // __CONTEXTMODEL__
