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

#pragma once

#include "TypeDef.h"
#include <iostream>

enum ScanType
{
    MATRIX_SCAN      = 0, //Row Wise
    BLOCK_SCAN_8x8   = 1, //Block Wise with block size 8x8
    BLOCK_SCAN_16x16 = 2, //Block Wise with block size 8x8
    BLOCK_SCAN_32x32 = 3, //Block Wise with block size 8x8
    BLOCK_SCAN_64x64 = 4, //Block Wise with block size 8x8
};

class Scan
{
public:
    Scan(ScanType scanType , uint32_t numWeights, uint32_t stride)
        : m_scanType(scanType),
          m_stride(stride),
          m_PosX(0),
          m_PosY(0),
          m_currScanIndex(0),
          m_currPosInMat(0),
          m_numWeights(numWeights)
    {

        m_height = numWeights / m_stride;

        if( scanType == MATRIX_SCAN )
        {
            m_BlockWidth = m_stride;
            m_BlockHeight = m_height;
        }
        else if (scanType == BLOCK_SCAN_8x8)
        {
            m_BlockWidth = 8;
            m_BlockHeight = 8;
        }
        else if (scanType == BLOCK_SCAN_16x16)
        {
            m_BlockWidth = 16;
            m_BlockHeight = 16;
        }
        else if (scanType == BLOCK_SCAN_32x32)
        {
            m_BlockWidth = 32;
            m_BlockHeight = 32;
        }
        else if (scanType == BLOCK_SCAN_64x64)
        {
            m_BlockWidth = 64;
            m_BlockHeight = 64;
        }
        else
        {
            CHECK(1, "Unsupported ScanType");
        }
        
        m_borderBlkHeight = m_height % m_BlockHeight;
        m_borderBlkWidth = m_stride % m_BlockWidth;


    } //stride == layderWidth
    ~Scan() {};

    uint32_t operator++ (int) { return getNextPosition();}
    uint32_t operator-- (int) { return getPreviousPosition();}
    uint32_t posInMat() { return m_currPosInMat; }
    uint32_t isLastPosOfBlockRowButNotLastPosOfBlock()
    {
        if(m_scanType == MATRIX_SCAN || m_currScanIndex + 1 == m_numWeights)
        {
            return 0;
        }
        if((m_currScanIndex + 1) % (m_stride * m_BlockHeight) == 0)
        {
            return 1;
        }
        return 0;
    }
    uint32_t isLastPosOfBlockRow()
    {
        if(m_scanType == MATRIX_SCAN)
        {
            return 0;
        }
        if((m_currScanIndex + 1) % (m_stride * m_BlockHeight) == 0|| m_currScanIndex + 1 == m_numWeights)
        {
            return 1;
        }
        return 0;
    }
    void seekBlockRow(int blockRow)
    {
        m_PosX = 0;
        m_PosY = m_BlockHeight * blockRow;
        m_currScanIndex = m_PosY * m_stride;
        m_currPosInMat = m_currScanIndex;
    }
    
    void     resetScan()
    {
        m_PosX = 0;
        m_PosY = 0;
        m_currScanIndex = 0;
        m_currPosInMat = 0;
    }

private:
    uint32_t getNextPosition()
    {
        if ((m_PosX + 1 < m_stride) && ((m_PosX + 1) % m_BlockWidth != 0))
        {
            m_PosX++;
            m_currScanIndex++;
        }
        else if ((m_PosY + 1 < m_height) && (((m_PosY + 1) % m_BlockHeight) != 0))
        {
            if (m_PosX + m_borderBlkWidth >= m_stride)
            {
                m_PosX -=  m_borderBlkWidth - 1;
            }
            else
            {
                m_PosX -= m_BlockWidth - 1;
            }
            m_PosY++;
            m_currScanIndex++;
        }
        else if (m_PosX + 1 < m_stride)
        {
            m_PosX++;

            if (m_PosY + m_borderBlkHeight >= m_height )
            {
                m_PosY -= m_borderBlkHeight - 1;
            }
            else
            {
                m_PosY -= m_BlockHeight - 1;
            }
            m_currScanIndex++;
        }
        else if (m_PosY + 1 < m_height)
        {
            m_PosX = 0;
            m_PosY++;
            m_currScanIndex++;
        }
        //else : Scan Position can not be further incremented to avoid out of bounds error!

        m_currPosInMat = m_PosY * m_stride + m_PosX;
        return m_currPosInMat;
    }

    uint32_t getPreviousPosition()
    {
        if ((m_PosX > 0) && ((m_PosX % m_BlockWidth) != 0))
        {
            m_PosX--;
            m_currScanIndex--;
        }
        else if ((m_PosY > 0) && (m_PosY % m_BlockHeight) != 0)
        {
            m_PosX += m_BlockWidth - 1;
            m_PosX = m_PosX >= m_stride ? m_stride - 1 : m_PosX;
            m_PosY--;
            m_currScanIndex--;
        }
        else if (m_PosX > 0)
        {
            m_PosX--;
            m_PosY += m_BlockHeight - 1;
            m_PosY = m_PosY >= m_height ? m_height - 1 : m_PosY;
            m_currScanIndex--;
        }
        else if (m_PosY > 0)
        {
            m_PosX = m_stride - 1;
            m_PosY--;
            m_currScanIndex--;
        }
        //else : Scan Position can not be further decremented to avoid out of bounds error!

        m_currPosInMat = m_PosY * m_stride + m_PosX;
        return m_currPosInMat;
    }

    uint32_t m_BlockWidth;
    uint32_t m_BlockHeight;
    uint32_t m_borderBlkHeight;
    uint32_t m_borderBlkWidth;

    ScanType m_scanType;
    uint32_t m_stride;
    uint32_t m_height;
    uint32_t m_PosX;
    uint32_t m_PosY;
    uint32_t m_currScanIndex;
    uint32_t m_currPosInMat;
    uint32_t m_numWeights;
};