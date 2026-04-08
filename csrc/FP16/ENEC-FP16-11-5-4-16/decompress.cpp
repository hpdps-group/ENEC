/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_host.h"

extern "C" void enec_decompress(Header *cphd, void *stream, uint8_t *compressed, uint8_t *decompressed);
extern "C" void enec_verify(Header *cphd, void *stream, uint8_t *compressed, uint8_t *source, uint8_t *out);

int main(int32_t argc, char *argv[])
{
    std::string inputFile;
    std::string outputFile;
    std::string sourceFile;

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.file> <output.file> ";
        std::cerr << "\nPositional arguments:\n"
                  << "  1. input.file      : Input file path\n"
                  << "  2. output.file     : Output file path\n"
                  << "  3. source.file     : Source file path\n";
        return 1;
    }

    inputFile = argv[1];
    outputFile = argv[2];
    sourceFile = argv[3];

    ifstream file(inputFile, ios::binary);
    if (!file)
    {
        cerr << "Unable to open the file: " << inputFile << endl;
        return EXIT_FAILURE;
    }
    file.seekg(0, ios::end);
    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *compressed = (uint8_t *)malloc(fileSize);
    file.read(reinterpret_cast<char *>(compressed), fileSize);
    file.close();

    Header *cphd = reinterpret_cast<Header *>(compressed);

    uint32_t *prefix = (uint32_t *)getCompSizePrefix(cphd, compressed);

    uint8_t *compressedDevice, *decompressed;
    CHECK_ACL(aclrtMalloc((void **)&compressedDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&decompressed, cphd->totalUncompressedBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(compressedDevice, fileSize, compressed, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("Decompression begin!\n");
    for (int i = 0; i < 4; i++)
    {
        enec_decompress(cphd, stream, compressedDevice, decompressed);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    uint8_t *decompressedHost;
    CHECK_ACL(aclrtMallocHost((void **)(&decompressedHost), cphd->totalUncompressedBytes));
    CHECK_ACL(aclrtMemcpy(decompressedHost, cphd->totalUncompressedBytes, decompressed, cphd->totalUncompressedBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    std::ofstream ofile;
    ofile.open(outputFile, std::ios::binary);

    aclrtStream stream0 = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream0));
    std::filebuf *obuf = ofile.rdbuf();
    ofile.write(reinterpret_cast<char *>(decompressedHost), cphd->totalUncompressedBytes_Origin);
    ofile.close();

    // ifstream file0(sourceFile, ios::binary);
    // if (!file0)
    // {
    //     cerr << "Unable to open the file: " << sourceFile << endl;
    //     return EXIT_FAILURE;
    // }
    // uint8_t *source = (uint8_t *)malloc(cphd->totalUncompressedBytes_Origin);
    // file0.read(reinterpret_cast<char *>(source), cphd->totalUncompressedBytes_Origin);
    // file0.close();

    // printf("Compressed size: %d\n", cphd->totalCompressedBytes);
    // printf("Size after decompression: %d\n", cphd->totalUncompressedBytes_Origin);

    // uint8_t *srcDevice, *outDevice;
    // CHECK_ACL(aclrtMalloc((void **)&srcDevice, cphd->totalUncompressedBytes_Origin, ACL_MEM_MALLOC_HUGE_FIRST));
    // CHECK_ACL(aclrtMalloc((void **)&outDevice, cphd->dataBlockNum * 32, ACL_MEM_MALLOC_HUGE_FIRST));
    // CHECK_ACL(aclrtMemcpy(srcDevice, cphd->totalUncompressedBytes_Origin, source, cphd->totalUncompressedBytes_Origin, ACL_MEMCPY_HOST_TO_DEVICE));

    // printf("自动验证解压正确性...\n");
    // bool check = true;
    // uint16_t* source2 = (uint16_t*)source;
    // uint16_t* decompressedHost2 = (uint16_t*)decompressedHost;
    // for (int i = 0; i < cphd->totalUncompressedBytes_Origin / 2; i++)
    // {
    //     // if(i / (cphd->dataBlockSize / 2) == 7967 && i % (cphd->dataBlockSize / 2) == 16286){
    //     //     printf("%d, %d\n", decompressedHost2[i], source2[i]);
    //     // }
    //     if (source2[i] != decompressedHost2[i])
    //     {
    //         check = false;
    //         int blockid = i / (cphd->dataBlockSize / 2);
    //         printf("fatal block id: %d, num: %d, decompressed: %d, source: %d\n", blockid, i % (cphd->dataBlockSize / 2), decompressedHost2[i], source2[i]);
    //         if(blockid >= 0){
    //             break;
    //         }
    //     }
    // }
    // if(check) printf("正确性无误！\n");
    // else printf("正确性有误！\n");

    printf("Cleaning up resources...\n");
    // 1. 释放 Device 端内存
    if (compressedDevice != nullptr) {
        CHECK_ACL(aclrtFree(compressedDevice));
    }
    if (decompressed != nullptr) {
        CHECK_ACL(aclrtFree(decompressed));
    }
    // if (srcDevice != nullptr) {
    //     CHECK_ACL(aclrtFree(srcDevice));
    // }
    // if (outDevice != nullptr) {
    //     CHECK_ACL(aclrtFree(outDevice));
    // }

    // 2. 释放 Host 端内存 (包含 malloc 分配的和 aclrtMallocHost 分配的)
    if (compressed != nullptr) {
        free(compressed);
        compressed = nullptr;
    }
    // if (source != nullptr) {
    //     free(source);
    //     source = nullptr;
    // }
    if (decompressedHost != nullptr) {
        CHECK_ACL(aclrtFreeHost(decompressedHost));
        decompressedHost = nullptr;
    }

    // 3. 销毁所有 Stream
    if (stream != nullptr) {
        CHECK_ACL(aclrtDestroyStream(stream));
    }
    if (stream0 != nullptr) {
        CHECK_ACL(aclrtDestroyStream(stream0));
    }

    // 4. 销毁 Context 并重置 Device
    if (context != nullptr) {
        CHECK_ACL(aclrtDestroyContext(context));
    }
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    printf("Resources cleaned up successfully.\n");
    return 0;
}