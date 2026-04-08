
#include "snec_utils.h"
#include "snec_device.h"

template <typename T>
class CompressKernelFP16
{
public:
    __aicore__ inline CompressKernelFP16() {}

    __aicore__ inline void Init(TPipe *pipe,
                                uint32_t datablockNum,
                                uint32_t datablockSize,
                                uint32_t elementNum,
                                uint32_t tileLength,
                                __gm__ uint8_t *srcDevice,          // e_input
                                __gm__ uint8_t *ms0Global,          // ms0_output
                                __gm__ uint8_t *ms1Global,          // ms1_output
                                __gm__ uint8_t *e0Global,           // e0_output
                                __gm__ uint8_t *mblGlobal,          // mbl_output
                                __gm__ uint8_t *e1Global,           // e1_output
                                __gm__ uint8_t *histogramDevice,    // table_input
                                __gm__ uint8_t *blockCompSize)
    {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->computeNum = elementNum;
        this->tileLength = tileLength;
        this->tileNum = elementNum / tileLength;
        this->datablockNum = datablockNum;
        this->datablockSize = datablockSize;
        int datablockNumPerBLOCK = (datablockNum + blockNum - 1) / blockNum;
        this->bufferSize = (datablockSize * datablockNumPerBLOCK);

        srcShape_cmp[0] = 1;
        srcShape_cmp[1] = tileNum / 8 / sizeof(half);
        dstShape_cmp[0] = tileLength;
        dstShape_cmp[1] = tileNum / 8 / sizeof(half);

        input.SetGlobalBuffer((__gm__ T *)(srcDevice));
        // table_input.SetGlobalBuffer((__gm__ T *)(histogramDevice));
        ms_output0.SetGlobalBuffer((__gm__ T *)(ms0Global));
        ms_output1.SetGlobalBuffer((__gm__ T *)(ms1Global));
        e_output0.SetGlobalBuffer((__gm__ T *)(e0Global));
        mbl_output.SetGlobalBuffer((__gm__ T *)(mblGlobal));
        e_output1.SetGlobalBuffer((__gm__ T *)(e1Global + bufferSize * blockId));
        blockCompSizeOutput.SetGlobalBuffer((__gm__ T *)(blockCompSize + 32 * blockId));

        // pipe->InitBuffer(inQueue, BUFFER_NUM, computeNum * sizeof(T));// 32KB
        // pipe->InitBuffer(e_outQueue0, BUFFER_NUM, computeNum * sizeof(T));// 32KB
        // // pipe->InitBuffer(e_outQueue1, BUFFER_NUM, computeNum * sizeof(T));// 32KB
        // pipe->InitBuffer(ms_outQueue, BUFFER_NUM, computeNum);// 16KB 
        pipe->InitBuffer(mbl_outQueue, BUFFER_NUM, tileLength * tileNum / 8);// 128B
    }

    __aicore__ inline void Process()
    {
        // pipe->InitBuffer(temp0, computeNum * sizeof(T));
        // pipe->InitBuffer(table, HISTOGRAM_BINS * sizeof(T));
        // pipe->InitBuffer(e_out1, computeNum * sizeof(T));// 32KB
        pipe->InitBuffer(merge, computeNum * sizeof(T));// 32KB
        // pipe->InitBuffer(cmbl, tileNum * sizeof(T));
        pipe->InitBuffer(mask7, 32);// 32B
        // pipe->InitBuffer(cmp, computeNum / 8);

        // LocalTensor<T> tempLocal0 = temp0.Get<T>();
        // LocalTensor<T> tableLocal = table.Get<T>();
        // LocalTensor<T> e_outLocal1 = e_out1.Get<T>();
        LocalTensor<T> mergeLocal = merge.Get<T>();
        // LocalTensor<T> cmblLocal = cmbl.Get<T>();
        LocalTensor<T> mask7Local = mask7.Get<T>();
        // LocalTensor<T> compareMask = cmp.Get<T>();

        pipe->InitBuffer(temp0, computeNum * sizeof(float));// 64KB
        pipe->InitBuffer(temp1, computeNum * sizeof(float));// 64KB

        LocalTensor<T> tempLocal0 = temp0.Get<T>();
        LocalTensor<T> tempLocal1 = temp1.Get<T>();

        // AIV_WITH_BARRIER(DataCopy, tableLocal, table_input, HISTOGRAM_BINS);
        // AIV_WITH_BARRIER(Duplicate, tempLocal0, (T)0, computeNum);
        AIV_WITH_BARRIER(Duplicate, mergeLocal, (T)0, computeNum);
        // AIV_WITH_BARRIER(Duplicate, cmblLocal, (T)0, tileNum);
        AIV_WITH_BARRIER(Duplicate, mask7Local, (T)7, 32 / sizeof(T));

        uint64_t outerNum = 0;
        uint32_t totalouterNum = 0;
        uint32_t totalcompressedSize = 0;
        uint32_t cumulated_amount = 0;
        uint32_t new_cumulated_amount = 0;
        uint32_t low_write_num = 0;
        uint32_t high_unwrite_num = 0;
        uint32_t write_offset = 0;
        for (uint32_t i = blockId; i < datablockNum; i += blockNum)
        {
            // if(i <= 48 * 20){
            // CopyIn(i);
            Compute(i,
                    cumulated_amount,
                    low_write_num,
                    high_unwrite_num,
                    write_offset,
                    outerNum,
                    tempLocal0,
                    tempLocal1,
                    // tableLocal,
                    // e_outLocal1,
                    mergeLocal,
                    // cmblLocal,
                    mask7Local
                    // ,
                    // compareMask
                );
            // totalcompressedSize = totalcompressedSize + computeNum * 3 / 8;
            totalouterNum = totalouterNum + outerNum;
            CopyOut(i);
            // }
        }
        totalouterNum = (totalouterNum + 256 - 1) / 256 * 256;// 向上取整到256个元素的倍数，这样折叠会保证32字节对齐
        totalcompressedSize = totalcompressedSize 
        + totalouterNum * 1 / 8
        ;
        // assert(computeNum == 16384);
        // assert(cumulated_amount <= computeNum);
        // assert(cumulated_amount % 16 == 0);
        // assert(cumulated_amount % 16 == 0);
        cumulated_amount = (cumulated_amount + 256 - 1) / 256 * 256;
        // PipeBarrier<PIPE_ALL>();
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)15, cumulated_amount);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)15, cumulated_amount);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 2], mergeLocal[cumulated_amount / 2], (uint16_t)1, cumulated_amount / 2);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 2], cumulated_amount / 2);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 4], mergeLocal[cumulated_amount / 4], (uint16_t)2, cumulated_amount / 4);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 4], cumulated_amount / 4);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 8], mergeLocal[cumulated_amount / 8], (uint16_t)4, cumulated_amount / 8);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 8], cumulated_amount / 8);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 16], mergeLocal[cumulated_amount / 16], (uint16_t)8, cumulated_amount / 16);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 16], cumulated_amount / 16);
        // // if(blockId == 31){
        // //     DumpTensor(mergeLocal, 1, cumulated_amount);
        // // }

        AIV_WITH_BARRIER(DataCopy, e_output1[write_offset], mergeLocal, cumulated_amount * 1 / 16);

        AIV_WITH_BARRIER(Duplicate, mask7Local, (T)0, 32 / sizeof(T));
        mask7Local.template ReinterpretCast<int32_t>()(0) = totalcompressedSize;
        PipeBarrier<PIPE_ALL>();
        AIV_WITH_BARRIER(DataCopy, blockCompSizeOutput, mask7Local, 32 / sizeof(T));
    }

private:
    // __aicore__ inline void CopyIn(uint32_t i)
    // {
    //     uint32_t offset = i * (computeNum * sizeof(uint16_t) / sizeof(T));
    //     LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    //     AIV_WITHOUT_BARRIER(DataCopy, inLocal, input[offset], computeNum);
    //     inQueue.EnQue(inLocal);
    // }

    __aicore__ inline void Compute( uint32_t i,
                                    uint32_t &cumulated_amount,
                                    uint32_t &low_write_num,
                                    uint32_t &high_unwrite_num,
                                    uint32_t &write_offset,
                                    uint64_t &outerNum,
                                    LocalTensor<T> &tempLocal0,
                                    LocalTensor<T> &tempLocal1,// 64KB
                                //    LocalTensor<T> &tableLocal,
                                    // LocalTensor<T> &e_outLocal1,// 32KB
                                    LocalTensor<T> &mergeLocal,// 32KB
                                //    LocalTensor<T> &cmblLocal,
                                    LocalTensor<T> &mask7Local
                                //    ,
                                //    LocalTensor<T> &compareMask
                                )// 3， 6
    {   // tempLocal0: e_inLocal(前半), mergeLocal(后半)
        // tempLocal1: e_outLocal0, ms_outLocal

        LocalTensor<T> compareMask = mbl_outQueue.AllocTensor<T>();// 1024/8 = 128bytes

        uint32_t offset = i * (computeNum * sizeof(uint16_t) / sizeof(T));
        AIV_WITH_BARRIER(DataCopy, tempLocal0, input[offset], computeNum * sizeof(uint16_t) / sizeof(T));
        // if(i == 0){
        //     // tempLocal0(0) = 8224;
        //     DumpTensor(tempLocal0, 1, 512);
        // }
        // assert(computeNum == 16384);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal0, (uint16_t)6, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)10, computeNum);
        AIV_WITH_BARRIER(Or, tempLocal0, tempLocal0, tempLocal1, computeNum);

        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal0, (uint16_t)5, computeNum);
        // if(i == 0){
        //     DumpTensor(tempLocal1, 1, 512);
        // }
        AIV_WITH_BARRIER(ShiftRight, tempLocal1[computeNum], tempLocal1, (uint16_t)8, computeNum);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum + computeNum / 2], tempLocal1[computeNum + computeNum / 2], (uint16_t)3, computeNum / 2);
        AIV_WITH_BARRIER(Or, tempLocal1[computeNum], tempLocal1[computeNum], tempLocal1[computeNum + computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum + computeNum / 4], tempLocal1[computeNum + computeNum / 4], (uint16_t)6, computeNum / 4);
        AIV_WITH_BARRIER(Or, tempLocal1[computeNum], tempLocal1[computeNum], tempLocal1[computeNum + computeNum / 4], computeNum / 4);

        AIV_WITH_BARRIER(ShiftRight, tempLocal1[computeNum + computeNum / 4], tempLocal1[computeNum], (uint16_t)8, computeNum / 4);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum + computeNum / 4 + computeNum / 8], tempLocal1[computeNum + computeNum / 4 + computeNum / 8], (uint16_t)4, computeNum / 8);
        AIV_WITH_BARRIER(Or, tempLocal1[computeNum + computeNum / 4], tempLocal1[computeNum + computeNum / 4], tempLocal1[computeNum + computeNum / 4 + computeNum / 8], computeNum / 8);
        // if(i == 0){
        //     DumpTensor(tempLocal1.template ReinterpretCast<uint16_t>()
        //     // [computeNum - 512]
        //     , 1, 512);
        // }
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal1, (uint16_t)8, computeNum + computeNum / 4 + computeNum / 8);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal1, (uint16_t)8, (computeNum + computeNum / 4 + computeNum / 8) / 2);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[(computeNum + computeNum / 4 + computeNum / 8) / 2], (computeNum + computeNum / 4 + computeNum / 8) / 2);
        // if(i == 0){
        //     DumpTensor(tempLocal1.template ReinterpretCast<uint16_t>()
        //     // [computeNum - 512]
        //     , 1, 512);
        // }
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<uint16_t>()[computeNum - 512], 1, 512);
        // }
        
        AIV_WITH_BARRIER(DataCopy, ms_output1[i * (computeNum * 11 / 8 / sizeof(T))], tempLocal1, computeNum * 11 / 8 / sizeof(T));

        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0, (uint16_t)11, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)11, computeNum);
        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<uint16_t>(), 1, 512);
        // }

        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int16_t>(), tempLocal0.template ReinterpretCast<int16_t>(), (int16_t)(-6), computeNum);
        AIV_WITH_BARRIER(Muls, tempLocal0.template ReinterpretCast<int16_t>(), tempLocal0.template ReinterpretCast<int16_t>(), (int16_t)(-1), computeNum);

        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<uint16_t>()
        //     // [computeNum - 512]
        //     , 1, 512);
        // }

        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0, (uint16_t)12, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)12, computeNum);

        // if(i == 0){
        //     DumpTensor(tempLocal0.template ReinterpretCast<uint16_t>()[computeNum - 512], 1, 512);
        // }

        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal0, tempLocal0[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 4], computeNum / 4);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 8], computeNum / 8);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 16], computeNum / 16);
        // if(i == 0){
        //     DumpTensor(tempLocal1.template ReinterpretCast<uint16_t>(), 1, 1056);
        // }
        // assert(tileNum == 1024);

        AIV_WITH_BARRIER(CompareScalar, compareMask.template ReinterpretCast<uint8_t>(), tempLocal1.template ReinterpretCast<half>(),
                      (mask7Local.template ReinterpretCast<half>())(0), CMPMODE::GT, tileNum);
        AIV_WITH_BARRIER(DataCopy, mbl_output[i * (tileNum / 8 / sizeof(T))], compareMask, tileNum / 8 / sizeof(T));
        // if(i == 0){
        //     DumpTensor(compareMask.template ReinterpretCast<uint16_t>(), 1, 1024 / 8 / 2);
        // }

        AIV_WITH_BARRIER(DataCopy, compareMask[64], compareMask, 64);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 1], compareMask, 64 << 1);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 2], compareMask, 64 << 2);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 3], compareMask, 64 << 3);
        // if(i == 0){
        //     DumpTensor(compareMask.template ReinterpretCast<uint16_t>()[64 * 15], 1, 1024 / 8 / 2);
        // }

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal0, (uint16_t)13, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal1, (uint16_t)13, computeNum);
        // if(i == 0){
        //     DumpTensor(tempLocal1.template ReinterpretCast<uint16_t>(), 1, 512);
        // }
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum / 2], tempLocal1[computeNum / 2], (uint16_t)3, computeNum / 2);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum / 4], tempLocal1[computeNum / 4], (uint16_t)6, computeNum / 4);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 4], computeNum / 4);

        AIV_WITH_BARRIER(ShiftRight, tempLocal1[computeNum / 4], tempLocal1, (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal1, (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal1, (uint16_t)8, computeNum / 4);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum / 4  + computeNum / 8], tempLocal1[computeNum / 4 + computeNum / 8], (uint16_t)4, computeNum / 8);
        AIV_WITH_BARRIER(Or, tempLocal1[computeNum / 4], tempLocal1[computeNum / 4], tempLocal1[computeNum / 4  + computeNum / 8], computeNum / 8);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[(computeNum * 3 / 16)], tempLocal1[(computeNum * 3 / 16)], (uint16_t)8, computeNum * 3 / 16);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[(computeNum * 3 / 16)], computeNum * 3 / 16);

        AIV_WITH_BARRIER(DataCopy, e_output0[i * (computeNum * 3 / 16)], tempLocal1, computeNum * 3 / 16);

        // if(i == 0){
        //     DumpTensor(tempLocal1.template ReinterpretCast<uint16_t>(), 1, 512);
        // }

        AIV_WITH_BARRIER(GatherMask, tempLocal1.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<half>(),
                   compareMask.template ReinterpretCast<uint16_t>(), true, computeNum, {1, 1, 1, 0}, outerNum);
        // assert(outerNum % 16 == 0);
        if(cumulated_amount + outerNum >= computeNum){
            low_write_num = computeNum - cumulated_amount;
            high_unwrite_num = outerNum - low_write_num;

            AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount], tempLocal1, (uint16_t)3, low_write_num);
            // if(i == 0){
            //     DumpTensor(mergeLocal[cumulated_amount], 1, 32);
            // }
            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 2], mergeLocal[computeNum / 2], (uint16_t)1, computeNum / 2);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 2], computeNum / 2);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 4], mergeLocal[computeNum / 4], (uint16_t)2, computeNum / 4);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 4], computeNum / 4);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 8], mergeLocal[computeNum / 8], (uint16_t)4, computeNum / 8);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 8], computeNum / 8);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 16], mergeLocal[computeNum / 16], (uint16_t)8, computeNum / 16);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 16], computeNum / 16);

            // if(write_offset == 0 && blockId == 0){
            //     DumpTensor(mergeLocal
            //         //[computeNum * 3 / 16 - 32]
            //         , 1, 32);
            // }
            AIV_WITH_BARRIER(DataCopy, e_output1[write_offset], mergeLocal, (computeNum * 1 / 16));
            write_offset = write_offset + computeNum * 1 / 16;

            AIV_WITH_BARRIER(ShiftRight, mergeLocal, tempLocal1[low_write_num], (uint16_t)3, high_unwrite_num);
            cumulated_amount = 
            high_unwrite_num;
            // assert(cumulated_amount % 16 == 0);
        }
        else {
            AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount], tempLocal1, (uint16_t)3, outerNum);
            // ,
            cumulated_amount = cumulated_amount + outerNum;
            // assert(computeNum == 16384);
            // assert(cumulated_amount <= computeNum);
            // assert(cumulated_amount % 16 == 0);
        }

        mbl_outQueue.EnQue(compareMask);
    }

    __aicore__ inline void CopyOut(uint32_t i)
    {
        // LocalTensor<T> e_outLocal0 = e_outQueue0.DeQue<T>();
        // LocalTensor<T> ms_outLocal = ms_outQueue.DeQue<T>();
        LocalTensor<T> compareMask = mbl_outQueue.DeQue<T>();

        // AIV_WITH_BARRIER(DataCopy, e_output0[i * (computeNum * 3 / 16)], e_outLocal0, computeNum * 3 / 16);
        // AIV_WITH_BARRIER(DataCopy, ms_output[i * (computeNum / sizeof(T))], ms_outLocal, (computeNum / sizeof(T)));
        AIV_WITH_BARRIER(DataCopy, mbl_output[i * (tileNum / 8 / sizeof(T))], compareMask, tileNum / 8 / sizeof(T));
        // if(i == 0){
        //     DumpTensor(compareMask, 1, tileNum / 8 / sizeof(T));
        // }

        // e_outQueue0.FreeTensor(e_outLocal0);
        // ms_outQueue.FreeTensor(ms_outLocal);
        mbl_outQueue.FreeTensor(compareMask);
    }

private:
    TPipe *pipe;

    // TQue<QuePosition::VECIN, 1> inQueue;
    // TQue<QuePosition::VECOUT, 1> e_outQueue0;
    // // TQue<QuePosition::VECOUT, 1> e_outQueue1;
    // TQue<QuePosition::VECOUT, 1> ms_outQueue;
    TQue<QuePosition::VECOUT, 1> mbl_outQueue;

    TBuf<TPosition::VECCALC> temp0;
    TBuf<TPosition::VECCALC> temp1;
    // TBuf<TPosition::VECCALC> e_out1;
    // TBuf<TPosition::VECCALC> table;
    TBuf<TPosition::VECCALC> merge;
    // TBuf<TPosition::VECCALC> cmbl;
    TBuf<TPosition::VECCALC> mask7;
    // TBuf<TPosition::VECCALC> cmp;

    GlobalTensor<T> input;
    GlobalTensor<T> ms_output0;
    GlobalTensor<T> ms_output1;
    GlobalTensor<T> e_output0;
    GlobalTensor<T> mbl_output;
    GlobalTensor<T> blockCompSizeOutput;
    GlobalTensor<T> e_output1;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t computeNum;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t threadblockNum;
    uint32_t datablockNum;
    uint32_t datablockSize;
    uint32_t bufferSize;

    uint32_t srcShape_cmp[2];
    uint32_t dstShape_cmp[2];
};

__global__ __aicore__ void compFP16(uint32_t datablockNum,
                                    uint32_t datablockSize,
                                    uint32_t elementNum,
                                    uint32_t tileLength,
                                    __gm__ uint8_t* srcDevice,       // e_input
                                    __gm__ uint8_t* ms0Global,        // ms0_output
                                    __gm__ uint8_t* ms1Global,        // ms1_output
                                    __gm__ uint8_t* e0Global,         // e0_output
                                    __gm__ uint8_t* mblGlobal,       // mbl_output
                                    __gm__ uint8_t* e1Global,         // e1_output
                                    __gm__ uint8_t* histogramDevice, // table_input
                                    __gm__ uint8_t* blockCompSize)
{
    TPipe pipe;
    CompressKernelFP16<uint16_t> op;
    op.Init(&pipe, datablockNum, datablockSize, elementNum, tileLength, 
            srcDevice, 
            ms0Global, 
            ms1Global,
            e0Global, 
            mblGlobal, 
            e1Global, 
            histogramDevice, 
            blockCompSize);
    op.Process();
}

extern "C" void enec_compress(Header *cphd, void *stream, uint8_t* srcDevice, uint8_t* compressedDevice, uint8_t* compressedFinal, uint8_t* histogramDevice, uint8_t* blockCompSize)
{
    switch (cphd->dataType)
    {
    case 0:
    { // BF16
        // uint32_t elementNum = cphd->dataBlockSize / sizeof(uint32_t);
        // compBF16<<<BLOCK_NUM, nullptr, stream>>>(
        //     cphd->dataBlockNum, cphd->dataBlockSize, elementNum, cphd->tileLength, 
        //     srcDevice, 
        //     getMs0data(cphd, compressedFinal), 
        //     getMs1data(cphd, compressedFinal), 
        //     getEdata(cphd, compressedFinal),
        //     getMbl(cphd, compressedFinal), 
        //     getCompressed_exp(cphd, compressedDevice), 
        //     histogramDevice, 
        //     blockCompSize);
        break;
    }
    case 1:
    { // FP16
        uint32_t elementNum = cphd->dataBlockSize / sizeof(uint16_t);
        compFP16<<<BLOCK_NUM, nullptr, stream>>>(
            cphd->dataBlockNum, cphd->dataBlockSize, elementNum, cphd->tileLength, 
            srcDevice, 
            getMs0data(cphd, compressedFinal), 
            getMs1data(cphd, compressedFinal), 
            getEdata(cphd, compressedFinal),
            getMbl(cphd, compressedFinal), 
            getCompressed_exp(cphd, compressedDevice), 
            histogramDevice, 
            blockCompSize);
        // break;
        break;
    }
    case 2:
    { // FP32
        // uint32_t elementNum = cphd->dataBlockSize / sizeof(uint32_t);
        // compBF16<<<BLOCK_NUM, nullptr, stream>>>(
        //     cphd->dataBlockNum, cphd->dataBlockSize, elementNum, cphd->tileLength, 
        //     srcDevice, 
        //     getMs0data(cphd, compressedFinal), 
        //     getMs1data(cphd, compressedFinal), 
        //     getEdata(cphd, compressedFinal),
        //     getMbl(cphd, compressedFinal), 
        //     getCompressed_exp(cphd, compressedDevice), 
        //     histogramDevice, 
        //     blockCompSize);
        break;
        // break;
    }
    default:
        return;
    }
}