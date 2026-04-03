/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_device.h"

template <typename T>
class DecompressKernelBF16
{
public:
    __aicore__ inline DecompressKernelBF16() {}

    __aicore__ inline void Init(TPipe *pipe,
                                uint32_t BUFFER_NUM,
                                uint32_t elementNum,
                                uint32_t tileLength,
                                uint32_t tileNum,
                                uint32_t threadblockNum,
                                uint32_t datablockNum,
                                uint32_t datablockSize,
                                uint32_t totalCompressedBytes,
                                __gm__ uint8_t *msGlobal,          // ms_input
                                __gm__ uint8_t *eGlobal0,           // e_input
                                __gm__ uint8_t *mblGlobal,         // mbl_input
                                __gm__ uint8_t *compSizePrefix,    // compSizePrefix
                                __gm__ uint8_t *eGlobal1,           // e_input
                                __gm__ uint8_t *decompressedGlobal // output
    )
    {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->computeNum = elementNum;
        this->tileLength = tileLength;
        this->tileNum = computeNum / tileLength;
        this->BLOCK_NUM = threadblockNum;
        this->datablockNum = datablockNum;
        this->datablockSize = datablockSize;

        srcShape_1[0] = 128;
        srcShape_1[1] = 1;
        dstShape_1[0] = 128;
        dstShape_1[1] = 8;

        srcShape_prefix[0] = 1;
        srcShape_prefix[1] = tileNum;
        dstShape_prefix[0] = tileLength;
        dstShape_prefix[1] = tileNum;

        srcShape_offset[0] = tileLength;
        srcShape_offset[1] = 1;
        dstShape_offset[0] = tileLength;
        dstShape_offset[1] = tileNum;

        // table_input.SetGlobalBuffer((__gm__ T *)(tableGlobal));
        e_input0.SetGlobalBuffer((__gm__ T *)(eGlobal0));
        ms_input.SetGlobalBuffer((__gm__ T *)(msGlobal));
        mbl_input.SetGlobalBuffer((__gm__ T *)(mblGlobal));
        compSizePrefix_input.SetGlobalBuffer((__gm__ uint32_t *)(compSizePrefix));
        // e_input1.SetGlobalBuffer((__gm__ T *)(eGlobal1));
        output.SetGlobalBuffer((__gm__ T *)(decompressedGlobal));

        // pipe->InitBuffer(e_inQueue0, BUFFER_NUM, computeNum * sizeof(T));// 32KB
        // pipe->InitBuffer(outQueue, BUFFER_NUM, computeNum * sizeof(T));// 32KB
        // pipe->InitBuffer(ms_inQueue, BUFFER_NUM, computeNum);// 16KB
        pipe->InitBuffer(mbl_inQueue, BUFFER_NUM, tileLength * tileNum / 8);// 2KB

        pipe->InitBuffer(compPrefix, BLOCK_NUM * sizeof(uint32_t));// 192b
        LocalTensor<uint32_t> compPrefixLocal = compPrefix.Get<uint32_t>();
        AIV_WITH_BARRIER(DataCopy, compPrefixLocal, compSizePrefix_input, BLOCK_NUM);
        e_input1.SetGlobalBuffer((__gm__ T *)(eGlobal1 + compPrefixLocal(blockId)));

        int FinalOtherSize = 32 + // 头
                            datablockSize / 2 * datablockNum + // ms
                            (datablockSize / sizeof(uint16_t)) * 4 / 8 * datablockNum + // low bits
                            tileNum / 8 * datablockNum + // mbl compareMask
                            BLOCK_NUM * 4; // prefix

        this->threadcompedNum = ((blockId == BLOCK_NUM - 1 ? totalCompressedBytes - FinalOtherSize : compPrefixLocal(blockId + 1)) - compPrefixLocal(blockId)) * 8 / 2;
    }

    __aicore__ inline void Process()
    {

        // pipe->InitBuffer(cmbl, tileNum * sizeof(T));
        pipe->InitBuffer(merge, computeNum * sizeof(T));// 32KB
        // pipe->InitBuffer(temp0, computeNum * sizeof(T));
        // pipe->InitBuffer(temp1, computeNum * sizeof(T));
        pipe->InitBuffer(temp2, tileNum * sizeof(T));// 2KB
        // pipe->InitBuffer(temp3, tileNum * sizeof(T));
        pipe->InitBuffer(offset0, tileNum / 8);// 128B
        pipe->InitBuffer(offset1, tileLength * sizeof(T));// 32B
        pipe->InitBuffer(mask1, tileNum * sizeof(T));// 2KB

        pipe->InitBuffer(temp0, computeNum * sizeof(float));// 64KB
        pipe->InitBuffer(temp1, 32 + computeNum * sizeof(float));// 64KB + 32B

        LocalTensor<T> tempLocal0 = temp0.Get<T>();
        LocalTensor<T> tempLocal1 = temp1.Get<T>();

        // LocalTensor<T> cmblLocal = cmbl.Get<T>();
        LocalTensor<T> mergeLocal = merge.Get<T>();
        // LocalTensor<T> tempLocal0 = temp0.Get<T>();
        // LocalTensor<T> tempLocal1 = temp1.Get<T>();
        LocalTensor<T> tempLocal2 = temp2.Get<T>();
        // LocalTensor<T> tempLocal3 = temp3.Get<T>();
        LocalTensor<T> offset0Local = offset0.Get<T>();
        LocalTensor<T> offset1Local = offset1.Get<T>();
        LocalTensor<T> mask1Local = mask1.Get<T>();
        
        // for(int i = 0; i < tileNum / 8 / sizeof(T); i++){
        //     offset0Local(i) = 32896;
        // }

        AIV_WITH_BARRIER(CreateVecIndex, offset1Local.template ReinterpretCast<int32_t>(), 0, tileLength);
        AIV_WITH_BARRIER(Duplicate, mask1Local, (T)1, tileNum);
        AIV_WITH_BARRIER(Duplicate, offset0Local, (T)32768, tileNum / 8 / sizeof(T));
        AIV_WITH_BARRIER(Duplicate, tempLocal1, (T)0, 16);

        AIV_WITH_BARRIER(DataCopy, mergeLocal, e_input1, computeNum * 2 / 16);
        int32_t eventIDMTE2ToV0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);

        uint64_t tempNum = 0;
        uint32_t outerNum = 0;
        uint32_t accouterNum = 0;
        uint32_t accCompressed = 0;
            
        assert(threadcompedNum % 256 == 0);
        uint32_t computeNum0 = computeNum >= threadcompedNum ? threadcompedNum : computeNum;

        AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 8], mergeLocal, (uint16_t)8, computeNum0 / 8);
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)8, computeNum0 / 8);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)8, computeNum0 / 8);

        AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 4], mergeLocal, (uint16_t)4, computeNum0 / 4);
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)4, computeNum0 / 4);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)4, computeNum0 / 4);

        AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 2], mergeLocal, (uint16_t)2, computeNum0 / 2);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)14, computeNum0);

        accCompressed = computeNum0;

        for(uint32_t i = blockId; i < datablockNum; i += blockNum)
        {
            // if(i == 0){
            CopyIn_mbl(i);
            // CopyIn_ms(i);
            Compute(
                i,
                tempNum,
                outerNum,
                accouterNum,
                accCompressed,
                mergeLocal,
                tempLocal0,
                tempLocal1,
                tempLocal2,
                offset0Local,
                offset1Local,
                mask1Local
            );
            // CopyOut(i);
            // PipeBarrier<PIPE_ALL>();
        // }
        }
    }

private:
    __aicore__ inline void CopyIn_mbl(uint32_t i)
    {
        LocalTensor<T> mbl_inLocal = mbl_inQueue.AllocTensor<T>();
        AIV_WITH_BARRIER(DataCopy, mbl_inLocal, mbl_input[i * (tileNum / 8 / sizeof(T))], tileNum / 8 / sizeof(T));
        mbl_inQueue.EnQue(mbl_inLocal);
    }

    __aicore__ inline void Compute(int32_t i,
                                   uint64_t &tempNum,
                                   uint32_t &outerNum,
                                   uint32_t &accouterNum,
                                   uint32_t &accCompressed,
                                   LocalTensor<T> &mergeLocal,// 32KB
                                   LocalTensor<T> &tempLocal0,// 64KB
                                   LocalTensor<T> &tempLocal1,// 64KB + 32B
                                   LocalTensor<T> &tempLocal2,// 2KB
                                   LocalTensor<T> &offset0Local,// 32B
                                   LocalTensor<T> &offset1Local,// 32B
                                   LocalTensor<T> &mask1Local// 4KB
                                )
    {
        LocalTensor<T> mbl_inLocal = mbl_inQueue.DeQue<T>();
        AIV_WITH_BARRIER(Select, tempLocal1[16].template ReinterpretCast<half>(), mbl_inLocal, mask1Local.template ReinterpretCast<half>(), (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileNum);

        auto src0Float = tempLocal1[16].template ReinterpretCast<half>();
        auto dst0Float = tempLocal0.template ReinterpretCast<half>();
        auto lastRowFloat = tempLocal0[computeNum].template ReinterpretCast<half>();
        auto sharedTmp = tempLocal1[16 + computeNum].template ReinterpretCast<uint8_t>();

        const CumSumInfo cumSumInfo{
            64,
            16
        };
        AIV_WITH_BARRIER((CumSum<half, cumSumConfig>), dst0Float, lastRowFloat, src0Float, sharedTmp, cumSumInfo);

        tempLocal0[computeNum].template ReinterpretCast<int16_t>()(15) = tempLocal0.template ReinterpretCast<int16_t>()(15);
        // AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<int16_t>()[8], tempLocal0.template ReinterpretCast<int16_t>(), tempLocal0.template ReinterpretCast<int16_t>()[8], tileNum - 8);// 2
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[16], tempLocal0.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<half>()[16], tileNum - 16);// 4
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[32], tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>()[32], tileNum - 32);// 8
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[64], tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>()[64], tileNum - 64);// 16
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[128], tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>()[128], tileNum - 128);// 32
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[256], tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>()[256], tileNum - 256);// 64
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[512], tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>()[512], tileNum - 512);// 128


        AIV_WITH_BARRIER(GatherMask, tempLocal2.template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>(), offset0Local.template ReinterpretCast<uint16_t>(), true, tileNum, {1, 1, 1, 0}, tempNum);
        // AIV_WITH_BARRIER(GatherMask, tempLocal2.template ReinterpretCast<half>(), tempLocal1[16].template ReinterpretCast<half>(), offset0Local.template ReinterpretCast<uint16_t>(), true, tileNum, {1, 1, 1, 0}, tempNum);
        // 广播至tileNum宽度，存在tempLocal0[computeNum]

        // if(i == 0){
        //     DumpTensor(tempLocal2, 1, 64);
        // }
        
        srcShape_0[0] = 64;
        srcShape_0[1] = 1;
        dstShape_0[0] = 64;
        dstShape_0[1] = 16;
        AIV_WITH_BARRIER((Broadcast<half, 2, 1>), tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal2.template ReinterpretCast<half>(), dstShape_0, srcShape_0);

        // 更新每8元素局部前缀和得到tileNum元素的前缀和，存在tempLocal0
        AIV_WITH_BARRIER(Add, tempLocal0.template ReinterpretCast<half>()[16], tempLocal0.template ReinterpretCast<half>()[16], tempLocal0[computeNum].template ReinterpretCast<half>(), tileNum - 16);

        Cast(tempLocal0.template ReinterpretCast<float>(), tempLocal0.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, tileNum);
        Cast(tempLocal0.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<float>(), RoundMode::CAST_TRUNC, tileNum);

        float lastnum = (float)(tempLocal0.template ReinterpretCast<float>()(tileNum - 1));

        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum), tileNum);

        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[tileNum << 1], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 2), tileNum);

        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 1) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 3), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[tileNum << 2], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 4), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 2) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 5), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 2) + (tileNum << 1)], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 6), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[7 * tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 7), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3)], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 8), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 9), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 1)], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 10), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 1) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 11), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 2)], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 12), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 2) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 13), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 2) + (tileNum << 1)], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 14), tileNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 2) + (tileNum << 1) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 15), tileNum);

        outerNum = tempLocal0.template ReinterpretCast<int32_t>()(computeNum - 1);

        PipeBarrier<PIPE_ALL>();
        if(accouterNum + outerNum >= computeNum)
        {
            uint32_t remainNum = computeNum - accouterNum;
            uint32_t nextreadNum = outerNum - remainNum;

            AIV_WITH_BARRIER(ShiftRight, tempLocal1[16].template ReinterpretCast<int16_t>(), mergeLocal[accouterNum].template ReinterpretCast<int16_t>(), (int16_t)10, remainNum);

            uint32_t computeNum0 = accCompressed + computeNum >= threadcompedNum ? threadcompedNum - accCompressed : computeNum;

            AIV_WITH_BARRIER(DataCopy, mergeLocal, e_input1[accCompressed  * 2 / 8 / sizeof(T)], computeNum0 * 2 / 8 / sizeof(T));
            accCompressed = accCompressed + computeNum0;

            // 处理mergeLocal
            AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 8], mergeLocal, (uint16_t)8, computeNum0 / 8);
            // AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)8, computeNum0 / 8);
            // AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)8, computeNum0 / 8);

            AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 4], mergeLocal, (uint16_t)4, computeNum0 / 4);
            // AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)12, computeNum0 / 4);
            // AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)12, computeNum0 / 4);

            AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 2], mergeLocal, (uint16_t)2, computeNum0 / 2);
            AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)14, computeNum0);

            AIV_WITH_BARRIER(ShiftRight, tempLocal1[16 + remainNum].template ReinterpretCast<int16_t>(), mergeLocal.template ReinterpretCast<int16_t>(), (int16_t)10, nextreadNum);
            accouterNum  = nextreadNum;
        }
        else {
            AIV_WITH_BARRIER(ShiftRight, tempLocal1[16].template ReinterpretCast<int16_t>(), mergeLocal[accouterNum].template ReinterpretCast<int16_t>(), (int16_t)10, outerNum);
            accouterNum = accouterNum + outerNum;
        }

        // 计算反向gather的索引
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint32_t>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)1, computeNum);
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)30, computeNum);
        AIV_WITH_BARRIER(DataCopy, mbl_inLocal[tileNum / 8 / sizeof(T)], mbl_inLocal, tileNum / 8 / sizeof(T));
        AIV_WITH_BARRIER(DataCopy, mbl_inLocal[tileNum / 8 / sizeof(T) * 2], mbl_inLocal, tileNum / 8 / sizeof(T) * 2);
        AIV_WITH_BARRIER(DataCopy, mbl_inLocal[tileNum / 8 / sizeof(T) * 4], mbl_inLocal, tileNum / 8 / sizeof(T) * 4);
        AIV_WITH_BARRIER(DataCopy, mbl_inLocal[tileNum / 8 / sizeof(T) * 8], mbl_inLocal, tileNum / 8 / sizeof(T) * 8);
        AIV_WITH_BARRIER(Select, tempLocal0.template ReinterpretCast<float>(), mbl_inLocal, tempLocal0.template ReinterpretCast<float>(), (float)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, computeNum);

        AIV_WITH_BARRIER(Gather, tempLocal0.template ReinterpretCast<half>(), tempLocal1.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)0, (uint32_t)computeNum); 

        // 输入指数部分
        AIV_WITH_BARRIER(DataCopy, tempLocal0[computeNum], e_input0[i * computeNum * 4 / 8 / sizeof(T)], computeNum * 4 / 8 / sizeof(T));

        // 展开指数部分
        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum * 4 / 16], tempLocal0[computeNum], (uint16_t)8, computeNum * 4 / 16);
        // AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)8, computeNum * 4 / 16);
        // AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)8, computeNum * 4 / 16);

        // AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum / 4 + computeNum / 8], tempLocal0[computeNum][computeNum / 4], (uint16_t)4, computeNum / 8);
        // AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum][computeNum / 4], tempLocal0[computeNum][computeNum / 4], (uint16_t)8, computeNum / 4);
        // AIV_WITH_BARRIER(Or, tempLocal0[computeNum], tempLocal0[computeNum], tempLocal0[computeNum][computeNum / 4], computeNum / 4);

        // AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum / 4], tempLocal0[computeNum], (uint16_t)6, computeNum / 4);
        // AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum / 2], tempLocal0[computeNum], (uint16_t)3, computeNum / 2);

        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum / 2], tempLocal0[computeNum], (uint16_t)4, computeNum / 2);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)12, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)12, computeNum);

    //    // 输入指数部分
    //     AIV_WITH_BARRIER(DataCopy, tempLocal0[computeNum], e_input0[i * computeNum * 3 / 8 / sizeof(T)], computeNum * 3 / 8 / sizeof(T));

    //     // 展开指数部分
    //     AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum * 3 / 16], tempLocal0[computeNum], (uint16_t)8, computeNum * 3 / 16);
    //     AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)8, computeNum * 3 / 16);
    //     AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)8, computeNum * 3 / 16);

    //     AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum / 4 + computeNum / 8], tempLocal0[computeNum][computeNum / 4], (uint16_t)4, computeNum / 8);
    //     AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum][computeNum / 4], tempLocal0[computeNum][computeNum / 4], (uint16_t)8, computeNum / 4);
    //     AIV_WITH_BARRIER(Or, tempLocal0[computeNum], tempLocal0[computeNum], tempLocal0[computeNum][computeNum / 4], computeNum / 4);

    //     AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum / 4], tempLocal0[computeNum], (uint16_t)6, computeNum / 4);
    //     AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum / 2], tempLocal0[computeNum], (uint16_t)3, computeNum / 2);

    //     AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)13, computeNum);
    //     AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)13, computeNum);

        // 指数数据恢复，逆线性拟合
        AIV_WITH_BARRIER(Or, tempLocal0[computeNum], tempLocal0[computeNum], tempLocal0, computeNum);
        AIV_WITH_BARRIER(Muls, tempLocal0[computeNum].template ReinterpretCast<int16_t>(), tempLocal0[computeNum].template ReinterpretCast<int16_t>(), (int16_t)(-1), computeNum);
        AIV_WITH_BARRIER(Adds, tempLocal0[computeNum].template ReinterpretCast<int16_t>(), tempLocal0[computeNum].template ReinterpretCast<int16_t>(), (int16_t)122, computeNum);

        // 恢复原始数据，尾数，符号，指数进行组合
        AIV_WITH_BARRIER(DataCopy, tempLocal0, ms_input[computeNum / 2 * i], computeNum / 2);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum / 2], tempLocal0, (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0, (uint16_t)8, computeNum);
        // AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)8, computeNum / 2);

        AIV_WITH_BARRIER(Or, tempLocal0[computeNum], tempLocal0[computeNum], tempLocal0, computeNum);

    //     // AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0[computeNum], (uint16_t)15, computeNum);
    //     // AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)1, computeNum);
    //     // AIV_WITH_BARRIER(Or, tempLocal0[computeNum], tempLocal0[computeNum], tempLocal0, computeNum);

        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0[computeNum], (uint16_t)9, computeNum);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)7, computeNum);
        AIV_WITH_BARRIER(Or, tempLocal0[computeNum], tempLocal0[computeNum], tempLocal0, computeNum);

        AIV_WITH_BARRIER(DataCopy, output[computeNum * i], tempLocal0[computeNum], computeNum);

        mbl_inQueue.FreeTensor(mbl_inLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECOUT, 1> mbl_inQueue;

    TBuf<TPosition::VECCALC> compPrefix;
    TBuf<TPosition::VECCALC> e_in;
    TBuf<TPosition::VECCALC> merge;
    TBuf<TPosition::VECCALC> temp0;
    TBuf<TPosition::VECCALC> temp1;
    TBuf<TPosition::VECCALC> temp2;
    TBuf<TPosition::VECCALC> temp3;
    TBuf<TPosition::VECCALC> offset0;
    TBuf<TPosition::VECCALC> offset1;
    TBuf<TPosition::VECCALC> mask1;

    GlobalTensor<T> ms_input;
    GlobalTensor<T> e_input0;
    GlobalTensor<T> mbl_input;
    GlobalTensor<uint32_t> compSizePrefix_input;
    GlobalTensor<T> e_input1;
    GlobalTensor<T> output;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t computeNum;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t BLOCK_NUM;
    uint32_t datablockNum;
    uint32_t datablockSize;
    uint32_t totalCompressed;
    uint32_t threadcompedNum;

    uint32_t srcShape_0[2];
    uint32_t dstShape_0[2];
    uint32_t srcShape_1[2];
    uint32_t dstShape_1[2];
    uint32_t dstShape_prefix[2];
    uint32_t srcShape_prefix[2];
    uint32_t dstShape_offset[2];
    uint32_t srcShape_offset[2];

    static constexpr CumSumConfig cumSumConfig{true, false, false};
    const CumSumInfo cumSumInfo0{
        128,
        8
    };

};

__global__ __aicore__ void decompBF16(
    uint32_t BUFFER_NUM,
    uint32_t elementNum,
    uint32_t tileLength,
    uint32_t tileNum,
    uint32_t threadblockNum,
    uint32_t datablockNum,
    uint32_t datablockSize,
    uint32_t totalCompressedBytes,
    __gm__ uint8_t* msGlobal,
    __gm__ uint8_t* eGlobal0,
    __gm__ uint8_t* mblGlobal,
    __gm__ uint8_t* compSizePrefix,
    __gm__ uint8_t* eGlobal1,
    __gm__ uint8_t* decompressedGlobal)
{
    TPipe pipe;
    DecompressKernelBF16<uint16_t> op;
    op.Init(&pipe, BUFFER_NUM, elementNum, tileLength, tileNum, threadblockNum, datablockNum, datablockSize, totalCompressedBytes,
            msGlobal, eGlobal0, mblGlobal, compSizePrefix, eGlobal1, decompressedGlobal);
    op.Process();
}

extern "C" void enec_decompress(Header* cphd, void* stream, uint8_t* compressed, uint8_t* decompressed)
{
    switch (cphd->dataType)
    {
    case 0:
    { // BF16
        uint32_t elementNum = cphd->dataBlockSize / sizeof(uint16_t);
        uint32_t tileNum = elementNum / cphd->tileLength;
        decompBF16<<<cphd->threadBlockNum, nullptr, stream>>>(1, elementNum, cphd->tileLength, tileNum, cphd->threadBlockNum, cphd->dataBlockNum, cphd->dataBlockSize, cphd->totalCompressedBytes,
                                                            getMsdata(cphd, compressed), getEdata(cphd, compressed), getMbl(cphd, compressed), getCompSizePrefix(cphd, compressed), getCompressed_exp(cphd, compressed), decompressed);
        break;
    }
    case 1:
    { // FP16

        break;
    }
    case 2:
    { // FP32

        break;
    }
    default:
    {

        return;
    }
    }
}