/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_device.h"

template<typename T>
class mergeKernel {
public:
    __aicore__ inline mergeKernel() {}

    __aicore__ inline void Init(TPipe* pipe,
                                __gm__ uint8_t* finalHeader,
                                __gm__ uint8_t* finalMs,
                                __gm__ uint8_t* finalMbl,
                                __gm__ uint8_t* finalCompPrefix,
                                __gm__ uint8_t* compedExp,
                                __gm__ uint8_t* finalExp, //output
                                __gm__ uint8_t* histogramDevice,
                                __gm__ uint8_t* blockCompSize, //output
                                uint32_t dataBlockSize,
                                uint32_t dataBlockNum,
                                uint32_t threadBlockNum,
                                uint32_t compLevel,
                                uint32_t totalUncompressedBytes_Origin,
                                uint32_t totalUncompressedBytes,
                                uint32_t totalCompressedBytes,
                                uint32_t tileLength,
                                uint32_t dataType,
                                uint32_t mblLength,
                                uint32_t options,
                                uint32_t bufferSize) {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->datablocksize = dataBlockSize;
        this->datablocknum = dataBlockNum;
        this->threadblocknum = threadBlockNum;
        this->complevel = compLevel;
        this->totaluncompressedbytes_Origin = totalUncompressedBytes_Origin;
        this->totaluncompressedbytes = totalUncompressedBytes;
        this->totalcompressedbytes = totalCompressedBytes;
        this->tilelength = tileLength;
        this->datatype = dataType;
        this->mbllength = mblLength;
        this->options = options;
        this->buffersize = bufferSize;

        finalheader.SetGlobalBuffer((__gm__ T*)(finalHeader));
        finalms.SetGlobalBuffer((__gm__ T*)(finalMs));
        finalmbl.SetGlobalBuffer((__gm__ T*)(finalMbl));
        finalcompprefix.SetGlobalBuffer((__gm__ T*)(finalCompPrefix));
        compedexp.SetGlobalBuffer((__gm__ uint8_t*)(compedExp));
        finalexp.SetGlobalBuffer((__gm__ uint8_t*)(finalExp));
        histogram.SetGlobalBuffer((__gm__ T*)(histogramDevice));
        blockcompsize.SetGlobalBuffer((__gm__ T*)(blockCompSize));
    }

public:
    __aicore__ inline void Process()
    {
        pipe->InitBuffer(temp, 32 * 1024);
        LocalTensor<T> tempLocal = temp.Get<T>();

        pipe->InitBuffer(datacopy, 128 * 1024);// 128KB传输缓冲区
        LocalTensor<uint8_t> datacopyLocal = datacopy.Get<uint8_t>();

        pipe->InitBuffer(offset, 64);
        LocalTensor<T> offsetLocal = offset.Get<T>();
        AIV_WITH_BARRIER(Duplicate, offsetLocal, (T)16843009, 64 / sizeof(T));

        DataCopy(tempLocal[8], blockcompsize, BLOCK_NUM * 32 / sizeof(uint32_t));
        AIV_WITH_BARRIER(Duplicate, tempLocal, (T)0, 32 / sizeof(T));

        uint32_t thisblocksize = tempLocal((blockId + 1) * 8);

        AIV_WITH_BARRIER(Add, tempLocal[8].template ReinterpretCast<int32_t>(), tempLocal.template ReinterpretCast<int32_t>(), tempLocal[8].template ReinterpretCast<int32_t>(), 8 * 64 - 8);// 2
        AIV_WITH_BARRIER(Add, tempLocal[8 * 2].template ReinterpretCast<int32_t>(), tempLocal.template ReinterpretCast<int32_t>(), tempLocal[8 * 2].template ReinterpretCast<int32_t>(), 8 * 64 - 8 * 2);// 4
        AIV_WITH_BARRIER(Add, tempLocal[8 * 4].template ReinterpretCast<int32_t>(), tempLocal.template ReinterpretCast<int32_t>(), tempLocal[8 * 4].template ReinterpretCast<int32_t>(), 8 * 64 - 8 * 4);// 8
        AIV_WITH_BARRIER(Add, tempLocal[8 * 8].template ReinterpretCast<int32_t>(), tempLocal.template ReinterpretCast<int32_t>(), tempLocal[8 * 8].template ReinterpretCast<int32_t>(), 8 * 64 - 8 * 8);// 16
        AIV_WITH_BARRIER(Add, tempLocal[8 * 16].template ReinterpretCast<int32_t>(), tempLocal.template ReinterpretCast<int32_t>(), tempLocal[8 * 16].template ReinterpretCast<int32_t>(), 8 * 64 - 8 * 16);// 32
        AIV_WITH_BARRIER(Add, tempLocal[8 * 32].template ReinterpretCast<int32_t>(), tempLocal.template ReinterpretCast<int32_t>(), tempLocal[8 * 32].template ReinterpretCast<int32_t>(), 8 * 64 - 8 * 32);// 64
        uint64_t tempNum = 0;
        AIV_WITH_BARRIER(GatherMask, tempLocal.template ReinterpretCast<float>(), tempLocal.template ReinterpretCast<float>(),
                   offsetLocal.template ReinterpretCast<uint32_t>(), true, 64 * 8, {1, 1, 1, 0}, tempNum);
        uint32_t totalcompsize = tempLocal(48 * 8);
        // if(blockId == 0){
        //     DumpTensor(tempLocal, 1, 49 * 8);
        // }

        uint32_t thisblockprefix = tempLocal(blockId);
        uint32_t cyclenum = (thisblocksize + 128 * 1024 - 1) / 128 / 1024;
        uint32_t copystart0 = 
            blockId * buffersize
            ;
        uint32_t copyend0 = thisblockprefix;
        for(int i = 0; i < cyclenum; i ++){
            uint32_t offset = i * 128 * 1024;
            if(offset >= totalcompsize) break;
            uint32_t size = (offset + 128 * 1024 > thisblocksize) ? (thisblocksize - offset) : 128 * 1024;
            uint32_t copystart1 = copystart0 + offset;
            uint32_t copyend1 = copyend0 + offset;
            PipeBarrier<PIPE_ALL>();

            DataCopy(datacopyLocal, compedexp[copystart1], size);
            // if(blockId == 0 && i == 0){
            //     DumpTensor(datacopyLocal.template ReinterpretCast<uint16_t>(), 1, 32);
            // }
            int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

            DataCopy(finalexp[copyend1], datacopyLocal, size);
            int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        }

        uint32_t totalcompressedsize = 0;
        if (datatype == 0 | datatype == 1)
        {
            totalcompressedsize = 32 +
                                totaluncompressedbytes / 2 +
                                datablocknum * (datablocksize / sizeof(uint16_t) * 3 / 8) +  
                                datablocknum * (datablocksize / (tilelength * sizeof(uint16_t)) / 8) +
                                threadblocknum * 4 +
                                totalcompsize;
        }
        else
        {
            totalcompressedsize = 32 +
                                totaluncompressedbytes / 2 +
                                datablocknum * (datablocksize / sizeof(uint16_t) * 3 / 8) +  
                                datablocknum * (datablocksize / (tilelength * sizeof(float))) / 8 +
                                threadblocknum * 4 +
                                totalcompsize;
        }

        if(blockId == 0){
            // prefix
            DataCopy(finalcompprefix, tempLocal, threadblocknum);

            // header
            tempLocal[64](0) = datablocksize;                // dataBlockSize
            tempLocal[64](1) = datablocknum;                 // dataBlockNum
            tempLocal[64](2) = threadblocknum | (complevel << 16); // threadBlockNum (low 16), compLevel (high 16)
            tempLocal[64](3) = totaluncompressedbytes_Origin;
            tempLocal[64](4) = totaluncompressedbytes;       // totalUncompressedBytes
            // assert(totaluncompressedbytes == 32 * 1024 * 1024);
            tempLocal[64](5) = totalcompressedsize;          // totalCompressedBytes
            tempLocal[64](6) = tilelength | (datatype << 16); // tileLength (low 16), dataType (high 16)
            tempLocal[64](7) = mbllength | (options << 16);   // mblLength (low 16), options (high 16)
            DataCopy(finalheader, tempLocal[64], 8);

        }
    }

private:
    TPipe* pipe;

    TBuf<TPosition::VECCALC> temp;
    TBuf<TPosition::VECCALC> datacopy;
    TBuf<TPosition::VECCALC> offset;

    GlobalTensor<T> finalheader;
    GlobalTensor<T> finalms;    
    GlobalTensor<T> finalmbl;
    GlobalTensor<T> finalcompprefix;
    GlobalTensor<uint8_t> compedexp; // output
    GlobalTensor<uint8_t> finalexp;
    GlobalTensor<T> histogram;
    GlobalTensor<T> blockcompsize;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t datablocksize;
    uint32_t datablocknum;
    uint32_t threadblocknum;
    uint32_t complevel;
    uint32_t totaluncompressedbytes_Origin;
    uint32_t totaluncompressedbytes;
    uint32_t totalcompressedbytes;
    uint32_t tilelength;
    uint32_t datatype;
    uint32_t mbllength;
    uint32_t options;
    uint32_t buffersize;
};

__global__ __aicore__ void merge(   
                                    __gm__ uint8_t* finalHeader,
                                    __gm__ uint8_t* finalMs,
                                    __gm__ uint8_t* finalMbl,
                                    __gm__ uint8_t* finalCompPrefix,
                                    __gm__ uint8_t* compedexp,
                                    __gm__ uint8_t* finalExp, //output
                                    __gm__ uint8_t* histogramDevice,
                                    __gm__ uint8_t* blockCompSize,
                                    uint32_t dataBlockSize,
                                    uint32_t dataBlockNum,
                                    uint32_t threadBlockNum,
                                    uint32_t compLevel,
                                    uint32_t totalUncompressedBytes_Origin,
                                    uint32_t totalUncompressedBytes,
                                    uint32_t totalCompressedBytes,
                                    uint32_t tileLength,
                                    uint32_t dataType,
                                    uint32_t mblLength,
                                    uint32_t options,
                                    uint32_t bufferSize
                                    )
{
    TPipe pipe;
    mergeKernel<uint32_t> op;
    op.Init(&pipe, finalHeader, finalMs, finalMbl, finalCompPrefix, compedexp, finalExp, histogramDevice, blockCompSize, dataBlockSize, dataBlockNum, threadBlockNum, compLevel, totalUncompressedBytes_Origin, totalUncompressedBytes, totalCompressedBytes, tileLength, dataType, mblLength, options, bufferSize);
    op.Process();
}

extern "C" void enec_merge(Header *cphd, void *stream, uint8_t *compressedDevice, uint8_t *compressedFinal, uint8_t *histogramDevice, uint8_t *blockCompSizeDevice, uint32_t bufferSize){
    merge<<<BLOCK_NUM, nullptr, stream>>>(compressedFinal, getMsdata(cphd, compressedFinal), getMbl(cphd, compressedFinal), getCompSizePrefix(cphd, compressedFinal), getCompressed_exp(cphd, compressedDevice), getCompressed_exp(cphd, compressedFinal), histogramDevice, blockCompSizeDevice, cphd->dataBlockSize, cphd->dataBlockNum, cphd->threadBlockNum, cphd->compLevel, cphd->totalUncompressedBytes_Origin, cphd->totalUncompressedBytes, cphd->totalCompressedBytes, cphd->tileLength, cphd->dataType, cphd->mblLength, cphd->options, bufferSize);
}
