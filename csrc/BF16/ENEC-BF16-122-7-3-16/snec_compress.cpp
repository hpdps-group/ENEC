
#include "snec_utils.h"
#include "snec_device.h"

template <typename T>
class CompressKernelBF16
{
public:
    __aicore__ inline CompressKernelBF16() {}

    __aicore__ inline void Init(TPipe *pipe,
                                uint32_t datablockNum,
                                uint32_t datablockSize,
                                uint32_t elementNum,
                                uint32_t tileLength,
                                __gm__ uint8_t *srcDevice,       // e_input
                                __gm__ uint8_t *msGlobal,        // ms_output
                                __gm__ uint8_t *e0Global,         // e0_output
                                __gm__ uint8_t *mblGlobal,       // mbl_output
                                __gm__ uint8_t *e1Global,       // e1_output
                                __gm__ uint8_t *histogramDevice, // table_input
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
        table_input.SetGlobalBuffer((__gm__ T *)(histogramDevice));
        ms_output.SetGlobalBuffer((__gm__ T *)(msGlobal));
        e_output0.SetGlobalBuffer((__gm__ T *)(e0Global));
        mbl_output.SetGlobalBuffer((__gm__ T *)(mblGlobal));
        e_output1.SetGlobalBuffer((__gm__ T *)(e1Global + bufferSize * blockId));
        blockCompSizeOutput.SetGlobalBuffer((__gm__ T *)(blockCompSize + 32 * blockId));

        pipe->InitBuffer(inQueue, BUFFER_NUM, computeNum * sizeof(T));// 32KB
        pipe->InitBuffer(e_outQueue0, BUFFER_NUM, computeNum * sizeof(T));// 32KB
        // pipe->InitBuffer(e_outQueue1, BUFFER_NUM, computeNum * sizeof(T));// 32KB
        pipe->InitBuffer(ms_outQueue, BUFFER_NUM, computeNum);// 16KB 
        pipe->InitBuffer(mbl_outQueue, BUFFER_NUM, tileLength * tileNum / 8);// 128B
    }

    __aicore__ inline void Process()
    {
        // pipe->InitBuffer(temp0, computeNum * sizeof(T));
        // pipe->InitBuffer(table, HISTOGRAM_BINS * sizeof(T));
        pipe->InitBuffer(e_out1, computeNum * sizeof(T));// 32KBs
        pipe->InitBuffer(merge, computeNum * sizeof(T));// 32KB
        // pipe->InitBuffer(cmbl, tileNum * sizeof(T));
        pipe->InitBuffer(mask7, 32);// 32B
        // pipe->InitBuffer(cmp, computeNum / 8);

        // LocalTensor<T> tempLocal0 = temp0.Get<T>();
        // LocalTensor<T> tableLocal = table.Get<T>();
        LocalTensor<T> e_outLocal1 = e_out1.Get<T>();
        LocalTensor<T> mergeLocal = merge.Get<T>();
        // LocalTensor<T> cmblLocal = cmbl.Get<T>();
        LocalTensor<T> mask7Local = mask7.Get<T>();
        // LocalTensor<T> compareMask = cmp.Get<T>();

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
            // if(i == 0){
            CopyIn(i);
            Compute(i,
                    cumulated_amount,
                    low_write_num,
                    high_unwrite_num,
                    write_offset,
                    outerNum,
                    // tempLocal0,
                    // tableLocal,
                    e_outLocal1,
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
        + totalouterNum * 4 / 8
        ;

        cumulated_amount = (cumulated_amount + 256 - 1) / 256 * 256;
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)12, cumulated_amount);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)12, cumulated_amount);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 2], mergeLocal[cumulated_amount / 2], (uint16_t)4, cumulated_amount / 2);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 2], cumulated_amount / 2);
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 4], mergeLocal[cumulated_amount / 4], (uint16_t)8, cumulated_amount / 4);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 4], cumulated_amount / 4);

        AIV_WITH_BARRIER(DataCopy, e_output1[write_offset], mergeLocal, cumulated_amount * 4 / 16);

        AIV_WITH_BARRIER(Duplicate, mask7Local, (T)0, 32 / sizeof(T));
        mask7Local.template ReinterpretCast<int32_t>()(0) = totalcompressedSize;
        AIV_WITH_BARRIER(DataCopy, blockCompSizeOutput, mask7Local, 32 / sizeof(T));
    }

private:
    __aicore__ inline void CopyIn(uint32_t i)
    {
        uint32_t offset = i * (computeNum * sizeof(uint16_t) / sizeof(T));
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        AIV_WITHOUT_BARRIER(DataCopy, inLocal, input[offset], computeNum);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute( uint32_t i,
                                    uint32_t &cumulated_amount,
                                    uint32_t &low_write_num,
                                    uint32_t &high_unwrite_num,
                                    uint32_t &write_offset,
                                    uint64_t &outerNum,
                                //    LocalTensor<T> &tempLocal0,
                                //    LocalTensor<T> &tableLocal,
                                    LocalTensor<T> &e_outLocal1,
                                    LocalTensor<T> &mergeLocal,
                                //    LocalTensor<T> &cmblLocal,
                                    LocalTensor<T> &mask7Local
                                //    ,
                                //    LocalTensor<T> &compareMask
                                )// 3， 6
    {

        LocalTensor<T> e_inLocal = inQueue.DeQue<T>();
        LocalTensor<T> e_outLocal0 = e_outQueue0.AllocTensor<T>();
        LocalTensor<T> ms_outLocal = ms_outQueue.AllocTensor<T>();
        LocalTensor<T> compareMask = mbl_outQueue.AllocTensor<T>();

        // AIV_WITH_BARRIER(ShiftRight, e_outLocal0, e_inLocal, (uint16_t)15, computeNum);
        // AIV_WITH_BARRIER(ShiftLeft, e_inLocal, e_inLocal, (uint16_t)1, computeNum);
        // AIV_WITH_BARRIER(Or, e_inLocal, e_inLocal, e_outLocal0, computeNum);

        // if(i == 0){
        //     DumpTensor(e_inLocal, 1, computeNum / 2);
        // }
        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0, e_inLocal, (uint16_t)9, computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal, e_inLocal, (uint16_t)7, computeNum);
        AIV_WITH_BARRIER(Or, e_inLocal, e_inLocal, e_outLocal0, computeNum);
        // if(i == 0){
        //     DumpTensor(e_inLocal, 1, 32);
        // }

        // AIV_WITH_BARRIER(ShiftLeft, e_outLocal0, e_inLocal, (uint16_t)8, computeNum);
        // AIV_WITH_BARRIER(ShiftRight, e_outLocal0, e_outLocal0, (uint16_t)8, (int)(computeNum / 2));
        // AIV_WITH_BARRIER(Or, ms_outLocal, e_outLocal0, e_outLocal0[computeNum / 2], computeNum / 2);

        AIV_WITH_BARRIER(ShiftRight, e_outLocal0, e_inLocal, (uint16_t)8, computeNum);
        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0[computeNum / 2], e_outLocal0[computeNum / 2], (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(Or, ms_outLocal, e_outLocal0, e_outLocal0[computeNum / 2], computeNum / 2);

        AIV_WITH_BARRIER(ShiftLeft, e_inLocal, e_inLocal, (uint16_t)8, computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal, e_inLocal, (uint16_t)8, computeNum);


        // if(i == 0){
        //     DumpTensor(e_inLocal[computeNum / 2], 1, computeNum / 4);
        // }


        // e_inLocal(0) = 124;
        AIV_WITH_BARRIER(Adds, e_inLocal.template ReinterpretCast<int16_t>(), e_inLocal.template ReinterpretCast<int16_t>(), (int16_t)(-122), computeNum);
        AIV_WITH_BARRIER(Muls, e_inLocal.template ReinterpretCast<int16_t>(), e_inLocal.template ReinterpretCast<int16_t>(), (int16_t)(-1), computeNum);
        // 取低7位
        AIV_WITH_BARRIER(ShiftLeft, e_inLocal, e_inLocal, (uint16_t)9, computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal, e_inLocal, (uint16_t)9, computeNum);

        // tileLength = 16
        AIV_WITH_BARRIER(Or, e_outLocal0, e_inLocal, e_inLocal[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[computeNum / 4], computeNum / 4);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[computeNum / 8], computeNum / 8);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[computeNum / 16], computeNum / 16);

        AIV_WITH_BARRIER(CompareScalar, compareMask.template ReinterpretCast<uint8_t>(), e_outLocal0.template ReinterpretCast<half>(),
                      (mask7Local.template ReinterpretCast<half>())(0), CMPMODE::GT, tileNum);

        AIV_WITH_BARRIER(DataCopy, compareMask[64], compareMask, 64);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 1], compareMask, 64 << 1);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 2], compareMask, 64 << 2);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 3], compareMask, 64 << 3);
        
        AIV_WITH_BARRIER(GatherMask, e_outLocal1.template ReinterpretCast<half>(), e_inLocal.template ReinterpretCast<half>(),
                   compareMask.template ReinterpretCast<uint16_t>(), true, computeNum, {1, 1, 1, 0}, outerNum);

        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0, e_inLocal, (uint16_t)13, computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_outLocal0, e_outLocal0, (uint16_t)13, computeNum);

        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0[computeNum / 2], e_outLocal0[computeNum / 2], (uint16_t)3, computeNum / 2);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0[computeNum / 4], e_outLocal0[computeNum / 4], (uint16_t)6, computeNum / 4);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[computeNum / 4], computeNum / 4);

        AIV_WITH_BARRIER(ShiftRight, e_outLocal0[computeNum / 4], e_outLocal0, (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0, e_outLocal0, (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftRight, e_outLocal0, e_outLocal0, (uint16_t)8, computeNum / 4);

        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0[computeNum / 4  + computeNum / 8], e_outLocal0[computeNum / 4 + computeNum / 8], (uint16_t)4, computeNum / 8);
        AIV_WITH_BARRIER(Or, e_outLocal0[computeNum / 4], e_outLocal0[computeNum / 4], e_outLocal0[computeNum / 4  + computeNum / 8], computeNum / 8);

        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0[(computeNum * 3 / 16)], e_outLocal0[(computeNum * 3 / 16)], (uint16_t)8, computeNum * 3 / 16);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[(computeNum * 3 / 16)], computeNum * 3 / 16);

        if(cumulated_amount + outerNum >= computeNum){
            low_write_num = computeNum - cumulated_amount;
            high_unwrite_num = outerNum - low_write_num;

            AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount], e_outLocal1, (uint16_t)3, low_write_num);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 2], mergeLocal[computeNum / 2], (uint16_t)4, computeNum / 2);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 2], computeNum / 2);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[(computeNum * 4 / 16)], mergeLocal[(computeNum * 4 / 16)], (uint16_t)8, computeNum * 4 / 16);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[(computeNum * 4 / 16)], computeNum * 4 / 16);

            AIV_WITH_BARRIER(DataCopy, e_output1[write_offset], mergeLocal, (computeNum * 4 / 16));
            write_offset = write_offset + computeNum * 4 / 16;

            AIV_WITH_BARRIER(ShiftRight, mergeLocal, e_outLocal1[low_write_num], (uint16_t)3, high_unwrite_num);
            cumulated_amount = high_unwrite_num;
        }
        else {
            AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount], e_outLocal1, (uint16_t)3, outerNum);

            cumulated_amount = cumulated_amount + outerNum;
        }

        e_outQueue0.EnQue(e_outLocal0);
        ms_outQueue.EnQue(ms_outLocal);
        mbl_outQueue.EnQue(compareMask);
        inQueue.FreeTensor(e_inLocal);
    }

    __aicore__ inline void CopyOut(uint32_t i)
    {
        LocalTensor<T> e_outLocal0 = e_outQueue0.DeQue<T>();
        LocalTensor<T> ms_outLocal = ms_outQueue.DeQue<T>();
        LocalTensor<T> compareMask = mbl_outQueue.DeQue<T>();

        AIV_WITH_BARRIER(DataCopy, e_output0[i * (computeNum * 3 / 16)], e_outLocal0, computeNum * 3 / 16);
        AIV_WITH_BARRIER(DataCopy, ms_output[i * (computeNum / sizeof(T))], ms_outLocal, (computeNum / sizeof(T)));
        AIV_WITH_BARRIER(DataCopy, mbl_output[i * (tileNum / 8 / sizeof(T))], compareMask, tileNum / 8 / sizeof(T));
        // if(i == 0){
        //     DumpTensor(compareMask, 1, tileNum / 8 / sizeof(T));
        // }

        e_outQueue0.FreeTensor(e_outLocal0);
        ms_outQueue.FreeTensor(ms_outLocal);
        mbl_outQueue.FreeTensor(compareMask);
    }

private:
    TPipe *pipe;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> e_outQueue0;
    // TQue<QuePosition::VECOUT, 1> e_outQueue1;
    TQue<QuePosition::VECOUT, 1> ms_outQueue;
    TQue<QuePosition::VECOUT, 1> mbl_outQueue;

    TBuf<TPosition::VECCALC> temp0;
    TBuf<TPosition::VECCALC> e_out1;
    TBuf<TPosition::VECCALC> table;
    TBuf<TPosition::VECCALC> merge;
    TBuf<TPosition::VECCALC> cmbl;
    TBuf<TPosition::VECCALC> mask7;
    // TBuf<TPosition::VECCALC> cmp;

    GlobalTensor<T> input;
    GlobalTensor<T> table_input;
    GlobalTensor<T> mbl_output;
    GlobalTensor<T> e_output0;
    GlobalTensor<T> e_output1;
    GlobalTensor<T> ms_output;
    GlobalTensor<T> blockCompSizeOutput;

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

__global__ __aicore__ void compBF16(uint32_t datablockNum,
                                    uint32_t datablockSize,
                                    uint32_t elementNum,
                                    uint32_t tileLength,
                                    __gm__ uint8_t* srcDevice,       // e_input
                                    __gm__ uint8_t* msGlobal,        // ms_output
                                    __gm__ uint8_t* e0Global,         // e0_output
                                    __gm__ uint8_t* mblGlobal,       // mbl_output
                                    __gm__ uint8_t* e1Global,         // e1_output
                                    __gm__ uint8_t* histogramDevice, // table_input
                                    __gm__ uint8_t* blockCompSize)
{
    TPipe pipe;
    CompressKernelBF16<uint16_t> op;
    op.Init(&pipe, datablockNum, datablockSize, elementNum, tileLength, 
            srcDevice, 
            msGlobal, 
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
        uint32_t elementNum = cphd->dataBlockSize / sizeof(uint16_t);
        compBF16<<<BLOCK_NUM, nullptr, stream>>>(
            cphd->dataBlockNum, cphd->dataBlockSize, elementNum, cphd->tileLength, 
            srcDevice, 
            getMsdata(cphd, compressedFinal), 
            getEdata(cphd, compressedFinal),
            getMbl(cphd, compressedFinal), 
            getCompressed_exp(cphd, compressedDevice), 
            histogramDevice, 
            blockCompSize);
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
        return;
    }
}