/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_device.h"

template <typename T>
class DecompressKernelFP32
{
public:
    __aicore__ inline DecompressKernelFP32() {}

    __aicore__ inline void Init(TPipe *pipe,
                                uint32_t BUFFER_NUM,
                                uint32_t elementNum,
                                uint32_t tileLength,
                                uint32_t tileNum,
                                uint32_t threadblockNum,
                                uint32_t datablockNum,
                                uint32_t datablockSize,
                                uint32_t totalCompressedBytes,
                                __gm__ uint8_t *ms0Global,          // ms_input
                                __gm__ uint8_t *ms1Global, 
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

        ms_input0.SetGlobalBuffer((__gm__ T *)(ms0Global));
        ms_input1.SetGlobalBuffer((__gm__ T *)(ms1Global));
        e_input0.SetGlobalBuffer((__gm__ T *)(eGlobal0));
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
                            datablockSize / 2 * datablockNum + // ms0
                            datablockSize / 4 * datablockNum + // ms1
                            (datablockSize / sizeof(uint32_t)) * 3 / 8 * datablockNum + // low bits
                            tileNum / 8 * datablockNum + // mbl compareMask
                            BLOCK_NUM * 4; // prefix

        this->threadcompedNum = ((blockId == BLOCK_NUM - 1 ? totalCompressedBytes - FinalOtherSize : compPrefixLocal(blockId + 1)) - compPrefixLocal(blockId)) * 8 / 4;
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

        AIV_WITH_BARRIER(DataCopy, mergeLocal, e_input1, computeNum * 4 / 16);
        int32_t eventIDMTE2ToV0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);

        // if(blockId == 0){
        //     DumpTensor(mergeLocal
        //         // [computeNum * 3 / 16 - 32]
        //         , 1, 32);
        // }

        uint64_t tempNum = 0;
        uint32_t outerNum = 0;
        uint32_t accouterNum = 0;
        uint32_t accCompressed = 0;
            
        // assert(threadcompedNum % 256 == 0);
        uint32_t computeNum0 = computeNum >= threadcompedNum ? threadcompedNum : computeNum;
        // uint32_t computeNum0 = computeNum;

        AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 * 4 / 16], mergeLocal, (uint16_t)8, computeNum0 * 4 / 16);
        // AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)8, computeNum0 * 3 / 16);
        // AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)8, computeNum0 * 3 / 16);

        // if(blockId == 0) {
        //     DumpTensor(mergeLocal, 1, 32);
        // }

        // AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 4 + computeNum0 / 8], mergeLocal[computeNum0 / 4], (uint16_t)4, computeNum0 / 8);
        // AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum0 / 4], mergeLocal[computeNum0 / 4], (uint16_t)12, computeNum0 / 8);
        // AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 4], mergeLocal[computeNum0 / 4], (uint16_t)12, computeNum0 / 8);

        // if(blockId == 0) {
        //     DumpTensor(mergeLocal, 1, 32);
        // }

        // AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum0 / 4], mergeLocal[computeNum0 / 4], (uint16_t)8, computeNum0 / 4);
        // AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum0 / 4], computeNum0 / 4);

        // if(blockId == 0){
        //     DumpTensor(mergeLocal, 1, 32);
        // }

        // AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 2], mergeLocal, (uint16_t)6, computeNum0 / 4);

        // AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)10, computeNum0);
        // AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)10, computeNum0);

        // if(blockId == 0) {
        //     DumpTensor(mergeLocal, 1, 32);
        // }

        AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 2], mergeLocal, (uint16_t)4, computeNum0 / 2);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)12, computeNum0);

        accCompressed = computeNum0;
        // AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)13, computeNum);

        // if(blockId == 0) {
        //     DumpTensor(mergeLocal, 1, 32);
        // }

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
        // LocalTensor<T> e_inLocal0 = e_inQueue0.DeQue<T>();// 32KB
        // LocalTensor<T> ms_inLocal = ms_inQueue.DeQue<T>();
        LocalTensor<T> mbl_inLocal = mbl_inQueue.DeQue<T>();
        // LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
        // if(i == 0){
        //     DumpTensor(mbl_inLocal, 1, 32);
        // }
        // PipeBarrier<PIPE_ALL>();        
        // AIV_WITH_BARRIER(Select, tempLocal0[computeNum].template ReinterpretCast<half>(), mbl_inLocal, mask1Local.template ReinterpretCast<half>(), (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileNum);
        AIV_WITH_BARRIER(Select, tempLocal1[16].template ReinterpretCast<half>(), mbl_inLocal, mask1Local.template ReinterpretCast<half>(), (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0[computeNum], 1, 32);
        // }

        // auto src0Float = tempLocal0[computeNum].template ReinterpretCast<half>();
        auto src0Float = tempLocal1[16].template ReinterpretCast<half>();
        auto dst0Float = tempLocal0.template ReinterpretCast<half>();
        // auto lastRowFloat = tempLocal1[16].template ReinterpretCast<half>();
        auto lastRowFloat = tempLocal0[computeNum].template ReinterpretCast<half>();
        auto sharedTmp = tempLocal1[16 + computeNum].template ReinterpretCast<uint8_t>();

        const CumSumInfo cumSumInfo{
            64,
            16
        };
        // 计算每8元素局部前缀和，结果存在tempLocal0
        AIV_WITH_BARRIER((CumSum<half, cumSumConfig>), dst0Float, lastRowFloat, src0Float, sharedTmp, cumSumInfo);
        // if(i == 0){
        //     DumpTensor(tempLocal0, 1, 1024);
        // }

        // if(i == 0)
        //     assert(tempLocal0.template ReinterpretCast<int16_t>()(15) == 5);

        tempLocal0[computeNum].template ReinterpretCast<int16_t>()(15) = tempLocal0.template ReinterpretCast<int16_t>()(15);
        // AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<int16_t>()[8], tempLocal0.template ReinterpretCast<int16_t>(), tempLocal0.template ReinterpretCast<int16_t>()[8], tileNum - 8);// 2
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[16], tempLocal0.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<half>()[16], tileNum - 16);// 4
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[32], tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>()[32], tileNum - 32);// 8
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[64], tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>()[64], tileNum - 64);// 16
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[128], tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>()[128], tileNum - 128);// 32
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[256], tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>()[256], tileNum - 256);// 64
        AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<half>()[512], tempLocal0[computeNum].template ReinterpretCast<half>(), tempLocal0[computeNum].template ReinterpretCast<half>()[512], tileNum - 512);// 128

        // tempLocal1[16].template ReinterpretCast<int16_t>()(15) = tempLocal0.template ReinterpretCast<int16_t>()(15);
        // // AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<int16_t>()[8], tempLocal0.template ReinterpretCast<int16_t>(), tempLocal0.template ReinterpretCast<int16_t>()[8], tileNum - 8);// 2
        // AIV_WITH_BARRIER(Add, tempLocal1[16].template ReinterpretCast<half>()[16], tempLocal0.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<half>()[16], tileNum - 16);// 4
        // AIV_WITH_BARRIER(Add, tempLocal1[16].template ReinterpretCast<half>()[32], tempLocal1[16].template ReinterpretCast<half>(), tempLocal1[16].template ReinterpretCast<half>()[32], tileNum - 32);// 8
        // AIV_WITH_BARRIER(Add, tempLocal1[16].template ReinterpretCast<half>()[64], tempLocal1[16].template ReinterpretCast<half>(), tempLocal1[16].template ReinterpretCast<half>()[64], tileNum - 64);// 16
        // AIV_WITH_BARRIER(Add, tempLocal1[16].template ReinterpretCast<half>()[128], tempLocal1[16].template ReinterpretCast<half>(), tempLocal1[16].template ReinterpretCast<half>()[128], tileNum - 128);// 32
        // AIV_WITH_BARRIER(Add, tempLocal1[16].template ReinterpretCast<half>()[256], tempLocal1[16].template ReinterpretCast<half>(), tempLocal1[16].template ReinterpretCast<half>()[256], tileNum - 256);// 64
        // AIV_WITH_BARRIER(Add, tempLocal1[16].template ReinterpretCast<half>()[512], tempLocal1[16].template ReinterpretCast<half>(), tempLocal1[16].template ReinterpretCast<half>()[512], tileNum - 512);// 128

        // if(i == 0){
        //     DumpTensor(tempLocal0[computeNum], 1, 1024);
        // }

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
        // if(i == 0){
        //     DumpTensor(tempLocal0, 1, 1024);
        // }
        Cast(tempLocal0.template ReinterpretCast<float>(), tempLocal0.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, tileNum);
        Cast(tempLocal0.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<float>(), RoundMode::CAST_TRUNC, tileNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<int32_t>(), 1, 1024);
        // }
        float lastnum = (float)(tempLocal0.template ReinterpretCast<float>()(tileNum - 1));
        // if(i == 0)
        //     assert(lastnum == 397);
        // 0.068

        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[0], tempLocal0.template ReinterpretCast<float>(), (float)(0), tileNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<int32_t>(), 1, 1024);
        // }
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum), tileNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<int32_t>()[tileNum], 1, 1024);
        // }
        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[tileNum << 1], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum * 2), tileNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<int32_t>()[tileNum * 2], 1, 1024);
        // }
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

        // // 计算反向gather的索引
        // AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint32_t>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)2, computeNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)28, computeNum);
        
        // AIV_WITH_BARRIER(ShiftLeft, tempLocal1[16].template ReinterpretCast<uint16_t>(), tempLocal1[16].template ReinterpretCast<uint16_t>(), (uint16_t)15, tileNum);
        // AIV_WITH_BARRIER(ShiftRight, tempLocal1[16].template ReinterpretCast<int16_t>(), tempLocal1[16].template ReinterpretCast<int16_t>(), (int16_t)15, tileNum);
        // AIV_WITH_BARRIER(DataCopy, tempLocal1[16 + tileNum], tempLocal1[16], tileNum);
        // AIV_WITH_BARRIER(DataCopy, tempLocal1[16 + (tileNum << 1)], tempLocal1[16], tileNum << 1);
        // AIV_WITH_BARRIER(DataCopy, tempLocal1[16 + (tileNum << 2)], tempLocal1[16], tileNum << 2);
        // AIV_WITH_BARRIER(DataCopy, tempLocal1[16 + (tileNum << 3)], tempLocal1[16], tileNum << 3);
        // AIV_WITH_BARRIER(And, tempLocal0.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), tempLocal1[16].template ReinterpretCast<uint16_t>(), computeNum);

        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<int32_t>()[tileNum * 15], 1, 1024);
        // }

        // Cast(tempLocal0.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<float>(), RoundMode::CAST_TRUNC, computeNum);

        // AIV_WITH_BARRIER(Select, tempLocal0[computeNum].template ReinterpretCast<float>(), mbl_inLocal, mask1Local.template ReinterpretCast<float>(), (float)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileNum);

        // auto src0Float = tempLocal0[computeNum].template ReinterpretCast<float>();
        // auto dst0Float = tempLocal0.template ReinterpretCast<float>();
        // auto lastRowFloat = tempLocal1[16].template ReinterpretCast<float>();
        // auto sharedTmp = tempLocal1[16 + computeNum].template ReinterpretCast<uint8_t>();

        // // 计算每8元素局部前缀和，结果存在tempLocal0
        // AIV_WITH_BARRIER((CumSum<float, cumSumConfig>), dst0Float, lastRowFloat, src0Float, sharedTmp, cumSumInfo0);
        
        // // 计算每8元素和的前缀和，结果存在tempLocal0[computeNum]
        // tempLocal0[computeNum].template ReinterpretCast<int32_t>()(8) = tempLocal0.template ReinterpretCast<int32_t>()(8);
        // AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<int32_t>()[8], tempLocal0.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<int32_t>()[8], tileNum - 8);// 2
        // AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<int32_t>()[16], tempLocal0[computeNum].template ReinterpretCast<int32_t>(), tempLocal0[computeNum].template ReinterpretCast<int32_t>()[16], tileNum - 16);// 4
        // AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<int32_t>()[32], tempLocal0[computeNum].template ReinterpretCast<int32_t>(), tempLocal0[computeNum].template ReinterpretCast<int32_t>()[32], tileNum - 32);// 8
        // AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<int32_t>()[64], tempLocal0[computeNum].template ReinterpretCast<int32_t>(), tempLocal0[computeNum].template ReinterpretCast<int32_t>()[64], tileNum - 64);// 16
        // AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<int32_t>()[128], tempLocal0[computeNum].template ReinterpretCast<int32_t>(), tempLocal0[computeNum].template ReinterpretCast<int32_t>()[128], tileNum - 128);// 32
        // AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<int32_t>()[256], tempLocal0[computeNum].template ReinterpretCast<int32_t>(), tempLocal0[computeNum].template ReinterpretCast<int32_t>()[256], tileNum - 256);// 64
        // AIV_WITH_BARRIER(Add, tempLocal0[computeNum].template ReinterpretCast<int32_t>()[512], tempLocal0[computeNum].template ReinterpretCast<int32_t>(), tempLocal0[computeNum].template ReinterpretCast<int32_t>()[512], tileNum - 512);// 128
        // // 取出每8元素和的前缀和，存在tempLocal2
        // AIV_WITH_BARRIER(GatherMask, tempLocal2.template ReinterpretCast<float>(), tempLocal0[computeNum].template ReinterpretCast<float>(), offset0Local.template ReinterpretCast<uint32_t>(), true, tileNum, {1, 1, 1, 0}, tempNum);
        // // 广播至tileNum宽度，存在tempLocal0[computeNum]
        // AIV_WITH_BARRIER((Broadcast<float, 2, 1>), tempLocal0[computeNum].template ReinterpretCast<float>(), tempLocal2.template ReinterpretCast<float>(), dstShape_1, srcShape_1);
        // // 更新每8元素局部前缀和得到tileNum元素的前缀和，存在tempLocal0
        // AIV_WITH_BARRIER(Add, tempLocal0.template ReinterpretCast<int32_t>()[8], tempLocal0.template ReinterpretCast<int32_t>()[8], tempLocal0[computeNum].template ReinterpretCast<int32_t>(), tileNum - 8);

        // // 取出tileNum前缀和最后一个元素
        // int lastnum = (int32_t)(tempLocal0.template ReinterpretCast<int32_t>()(tileNum - 1));
        // // 0.069
        // 将tileNum的前缀和扩展为computeNum长度
        // // AIV_WITH_BARRIER((Broadcast<float, 2, 0>), tempLocal1.template ReinterpretCast<float>()[8], tempLocal0.template ReinterpretCast<float>()[computeNum], dstShape_prefix, srcShape_prefix);
        // AIV_WITH_BARRIER(Muls, tempLocal2.template ReinterpretCast<int32_t>(), offset1Local.template ReinterpretCast<int32_t>(), lastnum, tileLength);
        // ArithProgression<int32_t>(tempLocal2.template ReinterpretCast<int32_t>(), static_cast<int32_t>(0), static_cast<int32_t>(lastnum), tileLength);

        // // int cum = 0;
        // // for(int i = 0; i < tileLength; i ++){
        // //     tempLocal2.template ReinterpretCast<int32_t>()(i) = i * lastnum;
        // //     // tempLocal2.template ReinterpretCast<int32_t>()(i) = cum;
        // //     // cum = cum + lastnum;
        // // }
        // // tempLocal2.template ReinterpretCast<int32_t>()(0) = 0;
        // // tempLocal2.template ReinterpretCast<int32_t>()(1) = lastnum;
        // // tempLocal2.template ReinterpretCast<int32_t>()(2) = lastnum << 1;
        // // tempLocal2.template ReinterpretCast<int32_t>()(3) = (lastnum << 1) + lastnum; 
        // // tempLocal2.template ReinterpretCast<int32_t>()(4) = lastnum << 2;
        // // tempLocal2.template ReinterpretCast<int32_t>()(5) = (lastnum << 2) + lastnum;
        // // tempLocal2.template ReinterpretCast<int32_t>()(6) = (lastnum << 2) + (lastnum << 1);
        // // tempLocal2.template ReinterpretCast<int32_t>()(7) = (lastnum << 2) + (lastnum << 1) + lastnum;
        // // tempLocal2.template ReinterpretCast<int32_t>()(8) = lastnum << 3;
        // // tempLocal2.template ReinterpretCast<int32_t>()(9) = (lastnum << 3) + lastnum;
        // // tempLocal2.template ReinterpretCast<int32_t>()(10) = (lastnum << 3) + (lastnum << 1);
        // // tempLocal2.template ReinterpretCast<int32_t>()(11) = (lastnum << 3) + (lastnum << 1) + lastnum;
        // // tempLocal2.template ReinterpretCast<int32_t>()(12) = (lastnum << 3) + (lastnum << 2);
        // // tempLocal2.template ReinterpretCast<int32_t>()(13) = (lastnum << 3) + (lastnum << 2) + lastnum;
        // // tempLocal2.template ReinterpretCast<int32_t>()(14) = (lastnum << 3) + (lastnum << 2) + (lastnum << 1);
        // // tempLocal2.template ReinterpretCast<int32_t>()(15) = (lastnum << 3) + (lastnum << 2) + (lastnum << 1) + lastnum;

        // tempLocal2.template ReinterpretCast<int32_t>()(0) = 0;
        // tempLocal2.template ReinterpretCast<int32_t>()(1) = lastnum;
        // tempLocal2.template ReinterpretCast<int32_t>()(2) = lastnum << 1;
        // tempLocal2.template ReinterpretCast<int32_t>()(3) = tempLocal2.template ReinterpretCast<int32_t>()(2) + lastnum; 
        // tempLocal2.template ReinterpretCast<int32_t>()(4) = lastnum << 2;
        // tempLocal2.template ReinterpretCast<int32_t>()(5) = tempLocal2.template ReinterpretCast<int32_t>()(4) + lastnum; 
        // tempLocal2.template ReinterpretCast<int32_t>()(6) = tempLocal2.template ReinterpretCast<int32_t>()(5) + lastnum;
        // tempLocal2.template ReinterpretCast<int32_t>()(7) = tempLocal2.template ReinterpretCast<int32_t>()(6) + lastnum;
        // tempLocal2.template ReinterpretCast<int32_t>()(8) = lastnum << 3;
        // tempLocal2.template ReinterpretCast<int32_t>()(9) = tempLocal2.template ReinterpretCast<int32_t>()(8) + lastnum;
        // tempLocal2.template ReinterpretCast<int32_t>()(10) = tempLocal2.template ReinterpretCast<int32_t>()(9) + lastnum;
        // tempLocal2.template ReinterpretCast<int32_t>()(11) = tempLocal2.template ReinterpretCast<int32_t>()(10) + lastnum;
        // tempLocal2.template ReinterpretCast<int32_t>()(12) = tempLocal2.template ReinterpretCast<int32_t>()(11) + lastnum;
        // tempLocal2.template ReinterpretCast<int32_t>()(13) = tempLocal2.template ReinterpretCast<int32_t>()(12) + lastnum;
        // tempLocal2.template ReinterpretCast<int32_t>()(14) = tempLocal2.template ReinterpretCast<int32_t>()(13) + lastnum;
        // tempLocal2.template ReinterpretCast<int32_t>()(15) = tempLocal2.template ReinterpretCast<int32_t>()(14) + lastnum;
        // AIV_WITH_BARRIER(Adds, tempLocal2.template ReinterpretCast<int32_t>()[8], tempLocal2.template ReinterpretCast<int32_t>(), (int32_t)(lastnum << 3), 8);
        // // for(int j = 0; j < tileLength; j ++){
        // //     // AIV_WITH_BARRIER(Duplicate, tempLocal0.template ReinterpretCast<int32_t>()[j * tileNum], j * lastnum, tileNum);
        // //     // AIV_WITH_BARRIER(Duplicate, tempLocal0.template ReinterpretCast<int32_t>()[j * tileNum], (int32_t)(tempLocal2.template ReinterpretCast<int32_t>()(j)), tileNum);
        // //     DataCopy(tempLocal1.template ReinterpretCast<int32_t>()[8 + j * tileNum], tempLocal0.template ReinterpretCast<int32_t>()[computeNum], tileNum);
        // // }
        // // 优化成logn次
        // DataCopy(tempLocal0.template ReinterpretCast<int32_t>()[tileNum], tempLocal0.template ReinterpretCast<int32_t>(), tileNum);
        // DataCopy(tempLocal0.template ReinterpretCast<int32_t>()[tileNum << 1], tempLocal0.template ReinterpretCast<int32_t>(), tileNum << 1);
        // DataCopy(tempLocal0.template ReinterpretCast<int32_t>()[tileNum << 2], tempLocal0.template ReinterpretCast<int32_t>(), tileNum << 2);
        // DataCopy(tempLocal0.template ReinterpretCast<int32_t>()[tileNum << 3], tempLocal0.template ReinterpretCast<int32_t>(), tileNum << 3);
        // DataCopy(tempLocal1.template ReinterpretCast<int32_t>()[8 + 0 * tileNum], tempLocal0.template ReinterpretCast<int32_t>()[computeNum], tileNum);
        // DataCopy(tempLocal1.template ReinterpretCast<int32_t>()[8 + 1 * tileNum], tempLocal1.template ReinterpretCast<int32_t>()[8 + 0 * tileNum], 1 * tileNum);
        // DataCopy(tempLocal1.template ReinterpretCast<int32_t>()[8 + 2 * tileNum], tempLocal1.template ReinterpretCast<int32_t>()[8 + 0 * tileNum], 2 * tileNum);
        // DataCopy(tempLocal1.template ReinterpretCast<int32_t>()[8 + 4 * tileNum], tempLocal1.template ReinterpretCast<int32_t>()[8 + 0 * tileNum], 4 * tileNum);
        // DataCopy(tempLocal1.template ReinterpretCast<int32_t>()[8 + 8 * tileNum], tempLocal1.template ReinterpretCast<int32_t>()[8 + 0 * tileNum], 8 * tileNum);
        // // AIV_WITH_BARRIER(Duplicate, tempLocal1.template ReinterpretCast<int32_t>()[8], (int32_t)0, computeNum);
        // // for(int j = 0; j < tileLength; j ++){
        // //     AIV_WITH_BARRIER(Or, tempLocal1.template ReinterpretCast<int32_t>()[8 + j * tileNum], tempLocal1.template ReinterpretCast<int32_t>()[8 + j * tileNum], tempLocal0.template ReinterpretCast<int32_t>()[computeNum], tileNum * 2);
        // // }
        // AIV_WITH_BARRIER((Broadcast<float, 2, 1>), tempLocal1.template ReinterpretCast<float>()[8], tempLocal2.template ReinterpretCast<float>(), dstShape_offset, srcShape_offset);
        // AIV_WITH_BARRIER(Add, tempLocal0.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<int32_t>(), tempLocal1.template ReinterpretCast<int32_t>()[8], computeNum);
        // for(int j = 0; j < tileLength; j ++){
        //     AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[j * tileNum], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(j * lastnum), tileNum);
        // }
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[0], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(0), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[tileNum], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[tileNum << 1], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(lastnum << 1), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 1) + tileNum], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)((lastnum << 1) + lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[tileNum << 2], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(lastnum << 2), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 2) + tileNum], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(5 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 2) + (tileNum << 1)], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(6 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[7 * tileNum], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(7 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 2) + (tileNum << 1) +tileNum], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(lastnum << 3), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 3) + tileNum], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(9 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 3) + (tileNum << 1)], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(10 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 3) + (tileNum << 1) + tileNum], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(11 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 3) + (tileNum << 2)], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(12 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 3) + (tileNum << 2) + tileNum], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(13 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 3) + (tileNum << 2) + (tileNum << 1)], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(14 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int32_t>()[(tileNum << 3) + (tileNum << 2) + (tileNum << 1) + tileNum], tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(15 * lastnum), tileNum);

        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[0], tempLocal0.template ReinterpretCast<float>(), (float)(0), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[tileNum << 1], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum << 1), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 1) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)((lastnum << 1) + lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[tileNum << 2], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum << 2), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 2) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(5 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 2) + (tileNum << 1)], tempLocal0.template ReinterpretCast<float>(), (float)(6 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[7 * tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(7 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 2) + (tileNum << 1) +tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(lastnum << 3), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(9 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 1)], tempLocal0.template ReinterpretCast<float>(), (float)(10 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 1) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(11 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 2)], tempLocal0.template ReinterpretCast<float>(), (float)(12 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 2) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(13 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 2) + (tileNum << 1)], tempLocal0.template ReinterpretCast<float>(), (float)(14 * lastnum), tileNum);
        // AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<float>()[(tileNum << 3) + (tileNum << 2) + (tileNum << 1) + tileNum], tempLocal0.template ReinterpretCast<float>(), (float)(15 * lastnum), tileNum);
        // 0.085

        // // 读取需要的码字
        // // SCALAR_WITH_BARRIER(outerNum = tempLocal0(computeNum - 1));
        // outerNum = tempLocal0.template ReinterpretCast<int32_t>()(computeNum - 1);
        PipeBarrier<PIPE_ALL>();
        if(accouterNum + outerNum >= computeNum)
        {
            uint32_t remainNum = computeNum - accouterNum;
            uint32_t nextreadNum = outerNum - remainNum;

            AIV_WITH_BARRIER(ShiftRight, tempLocal1[16].template ReinterpretCast<int16_t>(), mergeLocal[accouterNum].template ReinterpretCast<int16_t>(), (int16_t)9, remainNum);

            uint32_t computeNum0 = accCompressed + computeNum >= threadcompedNum ? threadcompedNum - accCompressed : computeNum;

            AIV_WITH_BARRIER(DataCopy, mergeLocal, e_input1[accCompressed  * 4 / 8 / sizeof(T)], computeNum0 * 4 / 8 / sizeof(T));
            accCompressed = accCompressed + computeNum0;

            // 处理mergeLocal
            AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 * 4 / 16], mergeLocal, (uint16_t)8, computeNum0 * 4 / 16);
            // AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)8, computeNum0 * 3 / 16);
            // AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)8, computeNum0 * 3 / 16);

            // AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 4 + computeNum0 / 8], mergeLocal[computeNum0 / 4], (uint16_t)4, computeNum0 / 8);
            // AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum0 / 4], mergeLocal[computeNum0 / 4], (uint16_t)8, computeNum0 / 4);
            // AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum0 / 4], computeNum0 / 4);

            // AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 4], mergeLocal, (uint16_t)6, computeNum0 / 4);
            AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum0 / 2], mergeLocal, (uint16_t)4, computeNum0 / 2);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)12, computeNum0);

            // int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            // SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            // WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

            AIV_WITH_BARRIER(ShiftRight, tempLocal1[16 + remainNum].template ReinterpretCast<int16_t>(), mergeLocal.template ReinterpretCast<int16_t>(), (int16_t)9, nextreadNum);
            accouterNum  = nextreadNum;
        }
        else {
            AIV_WITH_BARRIER(ShiftRight, tempLocal1[16].template ReinterpretCast<int16_t>(), mergeLocal[accouterNum].template ReinterpretCast<int16_t>(), (int16_t)9, outerNum);
            // if(i == 0){
            //     DumpTensor(tempLocal1[16], 1, outerNum);
            // }
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
        // if(i == 0)
        //     assert(tempLocal0.template ReinterpretCast<int32_t>()(computeNum - 1) == 12734);
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<int32_t>()[15 * 1024], 1, 1024);
        // }

        // if(i == 0){
        //     DumpTensor(tempLocal1[16].template ReinterpretCast<int16_t>(), 1, 64);
        // }
        // AIV_WITH_BARRIER(Duplicate, tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)12734, computeNum);
        // AIV_WITH_BARRIER(Gather, tempLocal1[16 + computeNum].template ReinterpretCast<half>(), tempLocal1.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)0, (uint32_t)computeNum);

        // Cast(tempLocal1[16].template ReinterpretCast<float>(), tempLocal1[16].template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, outerNum);
        // Cast(tempLocal1[16].template ReinterpretCast<int32_t>(), tempLocal1[16].template ReinterpretCast<float>(), RoundMode::CAST_TRUNC, outerNum);
        // if(i == 0){
        //     DumpTensor(tempLocal1.template ReinterpretCast<int32_t>(), 1, 64);
        // }
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<uint32_t>()[15 * 1024], 1, 1024);
        // }
        // AIV_WITH_BARRIER(Duplicate, tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)76, computeNum);
        AIV_WITH_BARRIER(Gather, tempLocal0.template ReinterpretCast<half>(), tempLocal1.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)0, (uint32_t)computeNum); 
        // AIV_WITH_BARRIER(Gather, tempLocal0.template ReinterpretCast<half>(), tempLocal1.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)0, (uint32_t)computeNum / 2); 
        // AIV_WITH_BARRIER(Gather, tempLocal0.template ReinterpretCast<half>()
        // [computeNum / 2]
        // , tempLocal1.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<uint32_t>()
        // // [computeNum / 2]
        // , (uint32_t)0, (uint32_t)computeNum / 2); 
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<int32_t>(), 1, 64);
        // }
        // AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint32_t>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)8, computeNum);
        // AIV_WITH_BARRIER(ShiftRight, tempLocal0.template ReinterpretCast<uint32_t>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)8, computeNum);

        // AIV_WITH_BARRIER(Cast, tempLocal0.template ReinterpretCast<int16_t>(), tempLocal0.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, computeNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0, 1, 64);
        // }
        // for(int j = 0; j < tileLength; j ++){
        //     AIV_WITH_BARRIER(Select, tempLocal0.template ReinterpretCast<half>()[j * tileNum], mbl_inLocal, tempLocal0.template ReinterpretCast<half>()[j * tileNum], (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileNum);
        // }
        
        // AIV_WITH_BARRIER(DataCopy, mbl_inLocal[tileNum / 8 / sizeof(T)], mbl_inLocal, tileNum / 8 / sizeof(T));
        // AIV_WITH_BARRIER(DataCopy, mbl_inLocal[tileNum / 8 / sizeof(T) * 2], mbl_inLocal, tileNum / 8 / sizeof(T) * 2);
        // AIV_WITH_BARRIER(DataCopy, mbl_inLocal[tileNum / 8 / sizeof(T) * 4], mbl_inLocal, tileNum / 8 / sizeof(T) * 4);
        // AIV_WITH_BARRIER(DataCopy, mbl_inLocal[tileNum / 8 / sizeof(T) * 8], mbl_inLocal, tileNum / 8 / sizeof(T) * 8);
        // AIV_WITH_BARRIER(Select, tempLocal0.template ReinterpretCast<half>(), mbl_inLocal, tempLocal0.template ReinterpretCast<half>(), (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, computeNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<int16_t>(), 1, computeNum);
        // }
        // if(i == 0){
        //     DumpTensor(tempLocal0[computeNum].template ReinterpretCast<int32_t>(), 1, computeNum / 2);
        // }
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<int16_t>(), 1, 32);
        // }


        // 输入指数部分
        AIV_WITH_BARRIER(DataCopy, tempLocal0[computeNum], e_input0[i * computeNum * 3 / 8 / sizeof(T)], computeNum * 3 / 8 / sizeof(T));

        // AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum * 3 / 16], mergeLocal, (uint16_t)8, computeNum * 3 / 16);
        // AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)8, computeNum * 3 / 16);
        // AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)8, computeNum * 3 / 16);

        // AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum / 4 + computeNum / 8], mergeLocal[computeNum / 4], (uint16_t)4, computeNum / 8);
        // AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 4], mergeLocal[computeNum / 4], (uint16_t)8, computeNum / 4);
        // AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 4], computeNum / 4);

        // AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum / 4], mergeLocal, (uint16_t)6, computeNum / 4);
        // AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum / 2], mergeLocal, (uint16_t)3, computeNum / 2);

        // AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)13, computeNum);
        // AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)13, computeNum);

        // 展开指数部分
        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum * 3 / 16], tempLocal0[computeNum], (uint16_t)8, computeNum * 3 / 16);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)8, computeNum * 3 / 16);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)8, computeNum * 3 / 16);

        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum / 4 + computeNum / 8], tempLocal0[computeNum][computeNum / 4], (uint16_t)4, computeNum / 8);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum][computeNum / 4], tempLocal0[computeNum][computeNum / 4], (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(Or, tempLocal0[computeNum], tempLocal0[computeNum], tempLocal0[computeNum][computeNum / 4], computeNum / 4);

        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum / 4], tempLocal0[computeNum], (uint16_t)6, computeNum / 4);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum][computeNum / 2], tempLocal0[computeNum], (uint16_t)3, computeNum / 2);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)13, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)13, computeNum);

        // if(i == 0){
        //     DumpTensor(tempLocal0[computeNum], 1, computeNum);
        // }

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
        // if(i == 0){
        //     DumpTensor(tempLocal0[computeNum].template ReinterpretCast<int16_t>()[computeNum - 512], 1, 512);
        // }
        AIV_WITH_BARRIER(Muls, tempLocal0[computeNum].template ReinterpretCast<int16_t>(), tempLocal0[computeNum].template ReinterpretCast<int16_t>(), (int16_t)(-1), computeNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0[computeNum].template ReinterpretCast<int32_t>(), 1, 64);
        // }
        AIV_WITH_BARRIER(Adds, tempLocal0[computeNum].template ReinterpretCast<int16_t>(), tempLocal0[computeNum].template ReinterpretCast<int16_t>(), (int16_t)121, computeNum);

        // if(i == 0){
        //     DumpTensor(tempLocal0[computeNum].template ReinterpretCast<int16_t>()[computeNum - 512], 1, 512);
        // }
        // AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)8, computeNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0[computeNum + computeNum / 2], 1, computeNum / 2);
        // }

        // 恢复原始数据，尾数，符号，指数进行组合
        AIV_WITH_BARRIER(DataCopy, tempLocal0, ms_input1[computeNum / 2 * i], computeNum / 2);
        // if(i == 0){
        //     DumpTensor(tempLocal0
        //         , 1, 512);
        // }
        AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum / 2], tempLocal0, (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0, (uint16_t)8, computeNum);
        // AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)8, computeNum / 2);

        AIV_WITH_BARRIER(Or, tempLocal0[computeNum], tempLocal0[computeNum], tempLocal0, computeNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0[computeNum]
        //         [computeNum - 512]
        //         , 1, 512);
        // }
    //     // AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0[computeNum], (uint16_t)15, computeNum);
    //     // AIV_WITH_BARRIER(ShiftRight, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)1, computeNum);
    //     // AIV_WITH_BARRIER(Or, tempLocal0[computeNum], tempLocal0[computeNum], tempLocal0, computeNum);

        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0[computeNum], (uint16_t)9, computeNum);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0[computeNum], tempLocal0[computeNum], (uint16_t)7, computeNum);
        AIV_WITH_BARRIER(Or, tempLocal0[computeNum], tempLocal0[computeNum], tempLocal0, computeNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0[computeNum].template ReinterpretCast<uint16_t>()[computeNum - 512], 1, 512);
        // }
        AIV_WITH_BARRIER(Cast, tempLocal1[16].template ReinterpretCast<float>(), tempLocal0[computeNum].template ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, computeNum);
        AIV_WITH_BARRIER(DataCopy, tempLocal0.template ReinterpretCast<uint16_t>(), ms_input0[computeNum * i], computeNum);

        // if(i == 1){
        //     DumpTensor(tempLocal0.template ReinterpretCast<uint16_t>(), 1, 512);
        // }
        AIV_WITH_BARRIER(ShiftRight, tempLocal0.template ReinterpretCast<uint32_t>()[computeNum / 2], tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)16, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint32_t>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)16, computeNum / 2);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0.template ReinterpretCast<uint32_t>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)16, computeNum / 2);

        AIV_WITH_BARRIER(Or, tempLocal0.template ReinterpretCast<uint32_t>(), tempLocal0.template ReinterpretCast<uint32_t>(), tempLocal1[16].template ReinterpretCast<uint32_t>(), computeNum * 2);

        // if(i == 1){
        //     DumpTensor(tempLocal0.template ReinterpretCast<uint32_t>()[computeNum - 512], 1, 512);
        // }
        AIV_WITH_BARRIER(DataCopy, output[computeNum * 2 * i], tempLocal0, computeNum * 2);

        mbl_inQueue.FreeTensor(mbl_inLocal);
    }

private:
    TPipe *pipe;

    // TQue<QuePosition::VECIN, 1> outQueue;
    // TQue<QuePosition::VECOUT, 1> e_inQueue0;
    // TQue<QuePosition::VECOUT, 1> ms_inQueue;
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

    GlobalTensor<T> ms_input0;
    GlobalTensor<T> ms_input1;
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

__global__ __aicore__ void decompFP32(
    uint32_t BUFFER_NUM,
    uint32_t elementNum,
    uint32_t tileLength,
    uint32_t tileNum,
    uint32_t threadblockNum,
    uint32_t datablockNum,
    uint32_t datablockSize,
    uint32_t totalCompressedBytes,
    __gm__ uint8_t* ms0Global,
    __gm__ uint8_t* ms1Global,
    __gm__ uint8_t* eGlobal0,
    __gm__ uint8_t* mblGlobal,
    __gm__ uint8_t* compSizePrefix,
    __gm__ uint8_t* eGlobal1,
    __gm__ uint8_t* decompressedGlobal)
{
    TPipe pipe;
    DecompressKernelFP32<uint16_t> op;
    op.Init(&pipe, BUFFER_NUM, elementNum, tileLength, tileNum, threadblockNum, datablockNum, datablockSize, totalCompressedBytes,
            ms0Global, ms1Global, eGlobal0, mblGlobal, compSizePrefix, eGlobal1, decompressedGlobal);
    op.Process();
}

extern "C" void enec_decompress(Header* cphd, void* stream, uint8_t* compressed, uint8_t* decompressed)
{
    switch (cphd->dataType)
    {
    case 0:
    { // BF16
        // uint32_t elementNum = cphd->dataBlockSize / sizeof(uint32_t);
        // uint32_t tileNum = elementNum / cphd->tileLength;
        // decompBF16<<<cphd->threadBlockNum, nullptr, stream>>>(1, elementNum, cphd->tileLength, tileNum, cphd->threadBlockNum, cphd->dataBlockNum, cphd->dataBlockSize, cphd->totalCompressedBytes,
        //                                                     getMs0data(cphd, compressed), getMs1data(cphd, compressed), getEdata(cphd, compressed), getMbl(cphd, compressed), getCompSizePrefix(cphd, compressed), getCompressed_exp(cphd, compressed), decompressed);
        break;
    }
    case 1:
    { // FP16

        break;
    }
    case 2:
    { // FP32
        uint32_t elementNum = cphd->dataBlockSize / sizeof(uint32_t);
        uint32_t tileNum = elementNum / cphd->tileLength;
        decompFP32<<<cphd->threadBlockNum, nullptr, stream>>>(1, elementNum, cphd->tileLength, tileNum, cphd->threadBlockNum, cphd->dataBlockNum, cphd->dataBlockSize, cphd->totalCompressedBytes,
                                                            getMs0data(cphd, compressed), getMs1data(cphd, compressed), getEdata(cphd, compressed), getMbl(cphd, compressed), getCompSizePrefix(cphd, compressed), getCompressed_exp(cphd, compressed), decompressed);
        break;
    }
    default:
    {

        return;
    }
    }
}