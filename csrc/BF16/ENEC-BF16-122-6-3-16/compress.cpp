/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_host.h"

extern "C" void enec_table(uint32_t totalUncompressedSize, void *stream, uint8_t *srcDevice, uint8_t *histogramDevice, uint32_t dataType);
extern "C" void enec_compress(Header *cphd, void *stream, uint8_t *srcDevice, uint8_t *compressedDevice, uint8_t *compressedFinal, uint8_t *histogramDevice, uint8_t *blockCompSizeDevice);
extern "C" void enec_merge(Header *cphd, void *stream, uint8_t *compressedDevice, uint8_t *compressedFinal, uint8_t* histogramDevice, uint8_t* blockCompSizeDevice, uint32_t bufferSize);

int main(int32_t argc, char *argv[])
{
    std::string inputFile;
    std::string outputFile;
    size_t inputByteSize = 0;
    int tileLength = 16;
    int dataType = 0;

    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.file> <output.file> <inputByteSize>"
                  << " [tileLength=16] [dataTypes=0] [compLevel=0] [isStatistics=1]\n";
        std::cerr << "\nPositional arguments:\n"
                  << "  1. input.file      : Input file path\n"
                  << "  2. output.file     : Output file path\n"
                  << "  3. inputByteSize   : Size of input data in bytes\n"
                  << "  4. tileLength      : Tile size (default: 16)\n"
                  << "  5. dataTypes       : Data format (0=BF16, 1=FP16, 2=FP32) (default: 0)\n";
        return 1;
    }

    inputFile = argv[1];
    outputFile = argv[2];
    size_t inputByteSize_Origin = std::stoul(argv[3]);
    inputByteSize = (inputByteSize_Origin + 32768 - 1) / 32768 * 32768;
    tileLength = std::stoi(argv[4]);
    dataType = std::stoi(argv[5]);

    ifstream file(inputFile, ios::binary);
    if (!file)
    {
        cerr << "Unable to open the file: " << inputFile << endl;
        return EXIT_FAILURE;
    }
    streamsize fileSize = file.tellg();

    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint16_t *host = (uint16_t *)malloc(inputByteSize);
    file.read(reinterpret_cast<char *>(host), inputByteSize);
    file.close();

    uint32_t DATA_BLOCK_BYTE_NUM_C = DATA_BLOCK_ELEMENT_NUM_C * sizeof(uint16_t);
    uint32_t tileNum = DATA_BLOCK_ELEMENT_NUM_C / tileLength;

    uint8_t *compressedHost;
    CHECK_ACL(aclrtMallocHost((void **)(&compressedHost), getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C)));

    Header *cphd = (Header *)compressedHost;
    cphd->dataBlockSize = DATA_BLOCK_BYTE_NUM_C;
    cphd->dataBlockNum = (inputByteSize + DATA_BLOCK_BYTE_NUM_C - 1) / DATA_BLOCK_BYTE_NUM_C;
    cphd->threadBlockNum = BLOCK_NUM;
    cphd->compLevel = 0;
    cphd->totalUncompressedBytes_Origin = inputByteSize_Origin;
    cphd->totalUncompressedBytes = inputByteSize;
    cphd->totalCompressedBytes = 0;
    cphd->tileLength = tileLength;
    cphd->dataType = dataType;
    cphd->mblLength = 4;
    cphd->options = 3;

    uint8_t *srcDevice, *compressedDevice, *compressedFinal, *histogramDevice, *blockCompSizeDevice;
    CHECK_ACL(aclrtMalloc((void **)&srcDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&compressedDevice, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&compressedFinal, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&histogramDevice, BLOCK_NUM * HISTOGRAM_BINS * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&blockCompSizeDevice, BLOCK_NUM * 32 * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(srcDevice, inputByteSize, host, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("Compression begin!\n");
    for (int i = 0; i < 4; i++)
    {
        enec_compress(cphd, stream, srcDevice, compressedDevice, compressedFinal, histogramDevice, blockCompSizeDevice);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }
    // uint8_t *compressedHostMerged;
    // CHECK_ACL(aclrtMallocHost((void **)(&compressedHostMerged), getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C)));
    // CHECK_ACL(aclrtMemcpy(compressedHostMerged, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), compressedFinal, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), ACL_MEMCPY_DEVICE_TO_HOST));

    int datablockNum = (inputByteSize + DATA_BLOCK_BYTE_NUM_C - 1) / DATA_BLOCK_BYTE_NUM_C;
    // printf("datablockNum: %d\n", datablockNum);
    int datablockNumPerBLOCK = (datablockNum + BLOCK_NUM - 1) / BLOCK_NUM;
    // printf("datablockNumPerBLOCK: %d\n", datablockNumPerBLOCK);
    uint32_t bufferSize = (DATA_BLOCK_BYTE_NUM_C * datablockNumPerBLOCK);
    enec_merge(cphd, stream, compressedDevice, compressedFinal, histogramDevice, blockCompSizeDevice, bufferSize);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    uint8_t *compressedHostMerged;
    CHECK_ACL(aclrtMallocHost((void **)(&compressedHostMerged), getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C)));
    CHECK_ACL(aclrtMemcpy(compressedHostMerged, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), compressedFinal, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), ACL_MEMCPY_DEVICE_TO_HOST));
    Header *cphdM = (Header *)compressedHostMerged;
    uint32_t totalCompressedSize0 = cphdM->totalCompressedBytes;
    
    printf("Size before compression: %d\n", cphdM->totalUncompressedBytes_Origin);
    printf("Compressed size: %d\n", totalCompressedSize0);
    printf("cr: %f\n", computeCr(inputByteSize_Origin, totalCompressedSize0));

    std::ofstream ofile;
    ofile.open(outputFile, std::ios::binary);
    std::filebuf *obuf = ofile.rdbuf();
    ofile.write(reinterpret_cast<char *>(compressedHostMerged), totalCompressedSize0);
    ofile.close();


    // --- 内存释放与环境销毁 ---
    printf("Cleaning up resources...\n");

    // 1. 释放文件读取 buffer
    if (host) free(host);

    // 2. 释放 Device 内存
    if (srcDevice) CHECK_ACL(aclrtFree(srcDevice));
    if (compressedDevice) CHECK_ACL(aclrtFree(compressedDevice));
    if (compressedFinal) CHECK_ACL(aclrtFree(compressedFinal)); // 补上缺失的释放
    if (histogramDevice) CHECK_ACL(aclrtFree(histogramDevice));
    if (blockCompSizeDevice) CHECK_ACL(aclrtFree(blockCompSizeDevice));

    // 3. 释放 Host 内存
    if (compressedHost) CHECK_ACL(aclrtFreeHost(compressedHost)); // 补上最初的 Header buffer
    if (compressedHostMerged) CHECK_ACL(aclrtFreeHost(compressedHostMerged));

    // 4. 销毁 ACL 环境（严格遵守顺序：Stream -> Context -> Device -> Finalize）
    if (stream) CHECK_ACL(aclrtDestroyStream(stream));
    if (context) CHECK_ACL(aclrtDestroyContext(context)); // 补上销毁 Context
    
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    printf("Exited cleanly.\n");
    return 0;
}