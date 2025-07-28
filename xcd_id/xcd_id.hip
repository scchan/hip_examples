#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <unistd.h> // for getopt


#if (defined(__gfx942__) || defined(__gfx950__))
__device__
unsigned int get_xcd_id() {
    return __builtin_amdgcn_s_getreg(GETREG_IMMED(XCC_ID_XCC_ID_SIZE - 1, 
                                     XCC_ID_XCC_ID_OFFSET, XCC_ID));
}

__device__
unsigned int get_cu_id() {
    return __builtin_amdgcn_s_getreg(GETREG_IMMED(HW_ID_CU_ID_SIZE - 1,
                                     HW_ID_CU_ID_OFFSET, HW_ID));
}
#else


__device__
unsigned int get_xcd_id() {
    assert(false);
    return 0;
}

__device__
unsigned int get_cu_id() {
    assert(false);
    return 0;
}

#endif


struct block_info {
    long long int start_time;
    long long int end_time;
    unsigned int block_id;
    unsigned int xcd_id;
    unsigned int cu_id;
    unsigned int padding;  /* pad to 64bit aligned */
};

__global__ void storeBlockIds(block_info* info, unsigned long long sleep,
                              unsigned int a, unsigned int b, unsigned int* c, unsigned char* give_up) {
    auto start = wall_clock64();

    unsigned int r;
    // change the vgr # to force the compiler to reserve a certain number of registers
    // as an indirect way to control occupancy
    __asm__("v_add_u32 %0, %1, %2" : "={v255}" (r) : "{v62}" (a), "{v61}" (b) : );

    constexpr auto sleep_interval = 500;
    if (threadIdx.x == 0) {
        auto i = blockIdx.x;

        // Block i will have to wait for the completion of block i-1 before making progress
        if (i != 0) {
            constexpr unsigned long long wait_timer = 60 * 1000;
            #pragma unroll 1
            while (__hip_atomic_load(c + i - 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != r) {
                __builtin_amdgcn_s_sleep(sleep_interval);
                if (give_up != nullptr && wall_clock64() > start + wait_timer) {
                    give_up[i] = 1;
                    break;
                }
            }
        }

        auto end = wall_clock64();

        if (give_up == nullptr || 
            give_up[i] == 0) {
            __hip_atomic_store(c + i, r, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }

        block_info bi = { start, end, i, get_xcd_id(), get_cu_id()};
        info[i] = bi;
    }
}


int main(int argc, char* argv[]) {
    unsigned int blocks = 1024 * 4; // allow override with -b
    const int threadsPerBlock = 256;
    bool g = false;

    constexpr unsigned a = 1;
    constexpr unsigned b = 2;
    constexpr unsigned long long sleep = 1000000;

    // Parse -s <num_streams>, -b <blocks>, and -g from command line
    unsigned int num_streams = 4;
    int opt;
    while ((opt = getopt(argc, argv, "s:b:g")) != -1) {
        switch (opt) {
            case 's':
                num_streams = std::stoi(optarg);
                if (num_streams == 0) {
                    std::cerr << "num_streams must be > 0" << std::endl;
                    return 1;
                }
                break;
            case 'b':
                blocks = static_cast<unsigned int>(std::stoul(optarg));
                if (blocks == 0) {
                    std::cerr << "blocks must be > 0" << std::endl;
                    return 1;
                }
                break;
            case 'g':
                g = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-s num_streams] [-b blocks] [-g]" << std::endl;
                return 1;
        }
    }

    int p_high = 0;
    int p_low = 0;
    hipDeviceGetStreamPriorityRange(&p_low, &p_high);
    auto p = p_high;
    std::vector<hipStream_t> streams(num_streams);
    for (auto& s : streams) {
        hipStreamCreateWithPriority(&s, hipStreamDefault, ++p);
        p > p_low ? p_high : p;
    }

    const int iters = 4;
    std::vector<block_info*> d_info(iters, nullptr);
    for (auto& i : d_info) {
        hipMalloc(&i, blocks * sizeof(block_info));
    }
    std::vector<unsigned int*> c(iters, nullptr);
    for (auto& i : c) {
        hipMalloc(&i, blocks * sizeof(unsigned int));
        hipMemset(i, 0, blocks * sizeof(unsigned int));
    }
    std::vector<unsigned char*> give_up(iters, nullptr);
    if (g) {
        for (auto& i : c) {
            hipMalloc(&i, blocks * sizeof(unsigned char));
            hipMemset(i, 0, blocks * sizeof(unsigned char));
        }
    }
    

    auto s = streams.cbegin();
    for (auto i = 0; i < iters; ++i) {
        storeBlockIds<<<dim3(blocks), dim3(threadsPerBlock), 0, *s>>>(d_info[i], sleep, a, b, c[i], give_up[i]);
        ++s;
        if (s == streams.cend())
            s = streams.cbegin();
    } 

    for (auto s : streams) {
        hipStreamSynchronize(s);
    }   

    for (auto j = 0; j < iters; ++j) {
        std::vector<block_info> info(blocks);
        hipMemcpy(info.data(), d_info[j], blocks * sizeof(block_info), hipMemcpyDeviceToHost);

        std::vector<unsigned int> r(blocks);
        hipMemcpy(r.data(), c[j], blocks * sizeof(unsigned int), hipMemcpyDeviceToHost);

        std::vector<unsigned char> gu(blocks);
        if (g) {
            hipMemcpy(gu.data(), give_up[j], blocks * sizeof(unsigned char), hipMemcpyDeviceToHost);
        }
#if 0
        for (auto rr : r) {
            if (rr != a + b) {
                std::cout << "Data failed to verify: Iter " << j << std::endl;
                exit(1);
            }
        }
#endif

        std::cout << "Iter " << j << std::endl;

        // header
        std::cout << "HIP_Block_ID,XCD_ID,CU_ID,START_TIME,END_TIME,HANG" << std::endl;
        for (auto i = 0; i < blocks; ++i) {
            std::cout << info[i].block_id << "," << info[i].xcd_id << ","
                      << info[i].cu_id << "," << info[i].start_time << ","
                      << info[i].end_time << "," << (unsigned)(g?gu[i]:0) << std::endl;
        }
        std::cout << std::endl;
    }

    for (auto p: d_info) {
        hipFree(p);
    }
    for (auto p: c) {
        hipFree(p);
    }
    if (g) {
        for (auto p: give_up) {
            hipFree(p);
        }
    }
 
    for (auto s : streams) {
        hipStreamDestroy(s);
    }

    return 0;
}



