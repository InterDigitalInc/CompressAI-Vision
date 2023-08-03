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
#ifndef __BINDEC__
#define __BINDEC__

#include "CommonLib/ContextModel.h"
#include <iostream>

struct EntryPoint
{
    uint64_t totalBitOffset;
    uint32_t m_Value;
    int32_t dqState;
    uint32_t m_Range;
    
    uint64_t getEntryPointInt() const
    {
        return (totalBitOffset<<11) + ((m_Value&255)<<3) + (dqState&7);
    }
    
    void setEntryPointInt( uint64_t ep )
    {
        totalBitOffset = ep >> 11;
        m_Value    = (ep >>  3) & 255;
        dqState    =  ep & 7;
    }
};

class BinDec
{
public:
    BinDec() : m_Bytes( nullptr ), m_ByteStreamPtr( nullptr ) {}
    ~BinDec() {}

public:
    void          startBinDecoder      (                                     );
    void          setByteStreamBuf     ( uint8_t* byteStreamBuf              );

    uint32_t      decodeBin            ( SBMPCtx &ctxMdl                     );
    uint32_t      decodeBinEP          (                                     );
    uint32_t      decodeBinsEP         ( uint32_t numBins                    );
    void          entryPointStart      () { m_Range = 256; }
    EntryPoint    getEntryPoint();
    void          setEntryPoint(EntryPoint ep);
    void          setEntryPointWithRange(EntryPoint ep);

    unsigned      decodeBinTrm();
    void          finish();

    uint32_t      getBytesRead() { return m_BytesRead; }
    void          setBytesRead(uint32_t bytesRead) { m_BytesRead=bytesRead; }
    void          setByteStreamPtr(uint8_t* byteStreamPtr ) { m_ByteStreamPtr = byteStreamPtr; }
    uint8_t*      getByteStreamPtr() {return m_ByteStreamPtr;}

private:
    uint32_t m_Range;
    int32_t  m_BitsNeeded;
    uint32_t m_Value;
    uint32_t m_BytesRead;
    uint8_t *m_Bytes;
    uint8_t *m_ByteStreamPtr;
    static const uint32_t  m_auiGoRiceRange[ 10 ];
};

#endif // __BINDEC__
