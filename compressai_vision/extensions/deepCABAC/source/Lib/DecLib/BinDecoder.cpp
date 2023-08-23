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
#include <random>
#include <algorithm>
#include <iostream>
#include "BinDecoder.h"

#if _WIN32
inline uint32_t __builtin_clz(uint32_t x)
{
    unsigned long position;
    _BitScanReverse(&position, x);
    return 31 - position;
}
#endif

const uint32_t BinDec::m_auiGoRiceRange[10] =
    {
        6, 5, 6, 3, 3, 3, 3, 3, 3, 3};

void BinDec::setByteStreamBuf(uint8_t *byteStreamBuf)
{
    m_Bytes = byteStreamBuf;
}

void BinDec::startBinDecoder()
{
    m_BytesRead = 0;
    m_BitsNeeded = -8;

    m_Range = 510;

    CHECK(m_Bytes == nullptr, "Bitstream is not initialized!");

    m_Value = 256 * m_Bytes[0] + m_Bytes[1];
    m_ByteStreamPtr = m_Bytes + 2;
    m_BytesRead += 2;
}

uint32_t BinDec::decodeBin(SBMPCtx &ctxMdl)
{
    uint32_t rlps = ctxMdl.getRLPS(m_Range);
    uint32_t rmps = m_Range - rlps;
    int32_t is_lps = ((int32_t)(rmps + ~(m_Value >> 7))) >> 31;
    m_Range = rmps ^ ((rmps ^ rlps) & is_lps);
    m_Value -= (rmps << 7) & is_lps;
    int32_t minusBin = ctxMdl.getMinusMPS() ^ is_lps;

    uint32_t n = __builtin_clz(m_Range) - 23;

    m_Range <<= n;
    m_Value <<= n;
    m_BitsNeeded += n;
    if (m_BitsNeeded >= 0)
    {
        m_Value += (*m_ByteStreamPtr++) << m_BitsNeeded;
        m_BitsNeeded -= 8;
        m_BytesRead++;
    }
    ctxMdl.updateState(minusBin);
    return minusBin & 1;
}

uint32_t BinDec::decodeBinEP()
{
    m_Value += m_Value;
    if (++m_BitsNeeded >= 0)
    {
        m_Value += (*m_ByteStreamPtr++);
        m_BitsNeeded = -8;
        m_BytesRead++;
    }
    uint32_t bin = 0;
    uint32_t SR = m_Range << 7;
    if (m_Value >= SR)
    {
        m_Value -= SR;
        bin = 1;
    }
    return bin;
}

uint32_t BinDec::decodeBinsEP(uint32_t numBins)
{
    if (m_Range == 256)
    {
        uint32_t remBins = numBins;
        uint32_t bins = 0;
        while (remBins > 0)
        {
            uint32_t binsToRead = std::min<uint32_t>(remBins, 8); // read bytes if able to take advantage of the system's byte-read function
            uint32_t binMask = (1 << binsToRead) - 1;
            uint32_t newBins = (m_Value >> (15 - binsToRead)) & binMask;
            bins = (bins << binsToRead) | newBins;
            m_Value = (m_Value << binsToRead) & 0x7FFF;
            remBins -= binsToRead;
            m_BitsNeeded += binsToRead;
            if (m_BitsNeeded >= 0)
            {
                m_Value |= (*m_ByteStreamPtr++) << m_BitsNeeded;
                m_BitsNeeded -= 8;
                m_BytesRead++;
            }
        }
        return bins;
    }
    uint32_t remBins = numBins;
    uint32_t bins = 0;
    while (remBins > 8)
    {
        m_Value = (m_Value << 8) + ((*m_ByteStreamPtr++) << (8 + m_BitsNeeded));
        uint32_t SR = m_Range << 15;
        m_BytesRead++;

        for (int i = 0; i < 8; i++)
        {
            bins += bins;
            SR >>= 1;
            if (m_Value >= SR)
            {
                bins++;
                m_Value -= SR;
            }
        }
        remBins -= 8;
    }
    m_BitsNeeded += remBins;
    m_Value <<= remBins;
    if (m_BitsNeeded >= 0)
    {
        m_Value += (*m_ByteStreamPtr++) << m_BitsNeeded;
        m_BitsNeeded -= 8;
        m_BytesRead++;
    }
    uint32_t SR = m_Range << (remBins + 7);
    for (uint32_t i = 0; i < remBins; i++)
    {
        bins += bins;
        SR >>= 1;
        if (m_Value >= SR)
        {
            bins++;
            m_Value -= SR;
        }
    }
    return bins;
}

EntryPoint BinDec::getEntryPoint()
{
    EntryPoint e;
    e.totalBitOffset = 8 * (m_ByteStreamPtr - m_Bytes) - 8 + m_BitsNeeded;
    uint64_t byteOff = (e.totalBitOffset + 9) >> 3;
    uint64_t bitOff = (e.totalBitOffset + 9) & 7;
    uint64_t bytesReadCheck = byteOff;
    int64_t bitsNeededCheck = int64_t(bitOff) - 9;
    if (bitOff != 0)
    {
        bytesReadCheck++;
    }
    else
    {
        bitsNeededCheck += 8;
    }
    CHECK(bytesReadCheck != m_ByteStreamPtr - m_Bytes, "Mismatch");
    CHECK(bitsNeededCheck != m_BitsNeeded, "Mismatch2");

    e.m_Value = m_Value >> 7;
    e.m_Range = m_Range;
    return e;
}

void BinDec::setEntryPoint(EntryPoint ep)
{
    m_Range = 256;
    uint64_t byteOff = (ep.totalBitOffset + 9) >> 3;
    uint64_t bitOff = (ep.totalBitOffset + 9) & 7;
    uint64_t bytesReadCheck = byteOff;
    int64_t bitsNeededCheck = int64_t(bitOff) - 9;
    if (bitOff != 0)
    {
        bytesReadCheck++;
    }
    else
    {
        bitsNeededCheck += 8;
    }

    m_ByteStreamPtr = bytesReadCheck + m_Bytes;
    m_BitsNeeded = bitsNeededCheck;
    uint32_t bsValue = (uint32_t(m_ByteStreamPtr[-1]) << (8 + m_BitsNeeded)) & 127;
    m_Value = (ep.m_Value << 7) + bsValue;
}

void BinDec::setEntryPointWithRange(EntryPoint ep)
{
    setEntryPoint(ep);
    m_Range = ep.m_Range;
}

unsigned BinDec::decodeBinTrm()
{
    m_Range -= 2;
    unsigned SR = m_Range << 7;
    if (m_Value >= SR)
    {
        return 1;
    }
    else
    {
        if (m_Range < 256)
        {
            m_Range += m_Range;
            m_Value += m_Value;
            if (++m_BitsNeeded == 0)
            {
                m_Value += (*m_ByteStreamPtr++);
                m_BitsNeeded = -8;
                m_BytesRead++;
            }
        }
        return 0;
    }
}

void BinDec::finish()
{
    unsigned lastByte;
    lastByte = *(--m_ByteStreamPtr);
    if (((lastByte << (8 + m_BitsNeeded)) & 0xff) != 0x80)
    {
        std::cout << "No proper stop/alignment pattern at end of CABAC stream." << std::endl;
    }

    //  CHECK( ( ( lastByte << ( 8 + m_bitsNeeded ) ) & 0xff ) != 0x80,
    //        "No proper stop/alignment pattern at end of CABAC stream." );
}
