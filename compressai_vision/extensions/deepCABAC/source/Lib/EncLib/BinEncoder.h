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
#ifndef __BINENC__
#define __BINENC__

#include "CommonLib/ContextModel.h"
#include <iostream>

class BinEnc
{
public:
    BinEnc  () {}
    ~BinEnc () {}

    void      startBinEncoder      ();
    void      setByteStreamBuf     ( std::vector<uint8_t> *byteStreamBuf );

    uint32_t  encodeBin            ( uint32_t bin,  SBMPCtx &ctxMdl  );
    void      entryPointStart      () { m_Range = 256; }

    void      pseudoEncodeBin      ( uint32_t bin,       SBMPCtxOptimizer &ctxMdl );

    uint32_t  encodeBinEP          ( uint32_t bin                    );
    uint32_t  encodeBinsEP         ( uint32_t bins, uint32_t numBins );

    void      encodeBinTrm         ( unsigned bin );
    void      finish               (              );
    void      terminate_write      (              );
protected:
    void      write_out         ();
private:
    std::vector<uint8_t>   *m_ByteBuf;
    uint32_t                m_Low;
    uint32_t                m_Range;
    uint8_t                 m_BufferedByte;
    uint32_t                m_NumBufferedBytes;
    uint32_t                m_BitsLeft;
    static const uint32_t   m_auiGoRiceRange[ 10 ];
};


class BinEst
{
public:
  uint32_t encodeBin    ( uint32_t bin,  SBMPCtx& ctxMdl  )   { return ctxMdl.getBits().scaledEstBits[ bin ]; }
  uint32_t updateBin    ( uint32_t bin,  SBMPCtx& ctxMdl  )   { ctxMdl.updateState( -(int32_t)bin ); return 0; }
  uint32_t encodeBinEP  ( uint32_t bin )                      { return (1<<15); }
  uint32_t encodeBinsEP ( uint32_t bins, uint32_t numBins )   { return (numBins<<15); }
};

#endif // !__BINENC__
