// [expert-pin 2026-09-01] ดู expert-pin.cuh — สร้างครั้งเดียวต่อ tensor แล้ว cache ไว้ตลอดอายุโปรเซส
// [expert-pin v2 2026-09-01] โหมด dynamic (GGML_EXPERT_PIN_DYN=1): cache ปรับตัวเองตามความถี่จริง
//   พอร์ตแนวคิดจาก FreeToken (slot cache + capped fetch) แต่ fill เป็น async ล้วนบน side stream
//   เพราะเส้นทาง miss ของเราอ่าน host ผ่าน UVA/DMA ได้อยู่แล้ว ไม่ต้อง stall รอ copy
//   ข้อจำกัดสำคัญ: decode ถูก capture เป็น CUDA graph — โค้ด host ใน getter รันแค่ตอน capture
//   ครั้งแรกเท่านั้น เคอร์เนลนับความถี่ถูก capture ไว้ในกราฟ (นับทุกโทเคน) ส่วน refresh
//   ต้องทำจากเธรดพื้นหลัง ห้ามพึ่งว่า getter จะถูกเรียกซ้ำ
#include "expert-pin.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct expert_pin_table {
    void *  table_dev = nullptr;   // const void* [ne02]
    void *  blob_dev  = nullptr;   // ช่อง VRAM เรียงต่อกัน
    bool    ready     = false;

    // ---- โหมด dynamic ----
    bool     dyn         = false;
    int64_t  ne02        = 0;
    size_t   expert_size = 0;
    int      cap_slots   = 0;
    int      device      = 0;
    const char * host_base = nullptr; // ฐาน host ของ tensor (UVA) — ใช้จากเธรด refresh
    std::string  name;
    uint32_t * cnt_dev   = nullptr;   // ตัวนับความถี่บน device [ne02] (อัปเดตจากในกราф)
    uint32_t * cnt_host  = nullptr;   // pinned staging
    std::vector<float> score;
    std::vector<int>   slot_expert;   // slot -> expert (-1 = ว่าง)
    std::vector<int>   expert_slot;   // expert -> slot (-1 = ไม่ได้ pin)
    std::vector<char>  slot_cooldown; // 1 = เพิ่งไล่ ห้ามใช้จนรอบถัดไป (กัน race กับกราฟที่กำลังอ่าน)
};

static std::unordered_map<std::string, std::vector<int>> g_profile;
static bool     g_profile_loaded = false;
static size_t   g_budget_bytes   = 0;
static size_t   g_spent_bytes    = 0;
static int      g_pinned_total   = 0;

static bool     g_dyn          = false;
static int      g_refresh_ms   = 300; // คาบเธรด refresh
static int      g_max_swap     = 2;   // จำกัดสลับต่อ tensor ต่อรอบ (คุมแบนด์วิดท์ fill)
static int      g_tensors_hint = 93;
static cudaStream_t g_side_stream = nullptr;
static std::atomic<bool> g_thread_started{false};

static std::unordered_map<const void *, expert_pin_table> g_tables;
static std::mutex g_mutex;

static void expert_pin_load_profile() {
    const char * path = getenv("GGML_EXPERT_PIN");
    if (path) {
        FILE * f = fopen(path, "r");
        if (!f) {
            GGML_LOG_WARN("expert-pin: เปิดโปรไฟล์ %s ไม่ได้\n", path);
        } else {
            char name[256];
            int  expert;
            while (fscanf(f, "%255s %d", name, &expert) == 2) {
                g_profile[name].push_back(expert);
            }
            fclose(f);
        }
    }
    const char * mb = getenv("GGML_EXPERT_PIN_MB");
    g_budget_bytes = (size_t)(mb ? atoll(mb) : 2800) * 1024 * 1024;
    const char * dyn = getenv("GGML_EXPERT_PIN_DYN");
    g_dyn = dyn && atoi(dyn) != 0;
    const char * rf = getenv("GGML_EXPERT_PIN_REFRESH_MS");
    if (rf) { g_refresh_ms = std::max(50, atoi(rf)); }
    const char * sw = getenv("GGML_EXPERT_PIN_SWAP");
    if (sw) { g_max_swap = std::max(1, atoi(sw)); }
    const char * th = getenv("GGML_EXPERT_PIN_TENSORS");
    if (th) { g_tensors_hint = std::max(1, atoi(th)); }
    else if (!g_profile.empty()) { g_tensors_hint = (int) g_profile.size(); }
    fprintf(stderr, "expert-pin: โปรไฟล์ %zu tensors · งบ %zu MB · dyn=%d refresh=%dms swap=%d\n",
            g_profile.size(), g_budget_bytes / (1024 * 1024), (int) g_dyn,
            g_refresh_ms, g_max_swap);
}

// นับความถี่ expert ที่ถูก route จริง (ids = I32 บน device) — ถูก capture ในกราฟ decode
static __global__ void expert_pin_count_kernel(const int32_t * ids, int n, uint32_t * cnt, int ne02) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int32_t e = ids[i];
        if (e >= 0 && e < ne02) {
            atomicAdd(&cnt[e], 1u);
        }
    }
}

// เขียนพอยน์เตอร์ 1 ช่องในตาราง (store 64-bit aligned เดียว = atomic ต่อผู้อ่านในกราฟ)
static __global__ void expert_pin_set_entry_kernel(const void ** table, int idx, const void * ptr) {
    if (threadIdx.x == 0) {
        table[idx] = ptr;
    }
}

// สลับ pin ของ tensor เดียวไม่เกิน g_max_swap ช่อง — เรียกจากเธรด refresh ใต้ g_mutex
static void expert_pin_dyn_refresh(expert_pin_table & t) {
    uint64_t total = 0, hit = 0;
    for (int64_t e = 0; e < t.ne02; ++e) {
        t.score[e] = t.score[e] * 0.5f + (float) t.cnt_host[e];
        total += t.cnt_host[e];
        if (t.expert_slot[e] >= 0) { hit += t.cnt_host[e]; }
    }
    static int dbg_n = 0;   // ดีบักชั่วคราว: hit-rate เป็นระยะ
    if (total > 0 && dbg_n++ % 400 == 0) {
        fprintf(stderr, "[expert-pin dyn] %s hit %.0f%% (%llu/%llu) pinned=%zu/%d\n",
                t.name.c_str(), 100.0 * hit / total,
                (unsigned long long) hit, (unsigned long long) total,
                (size_t) std::count_if(t.expert_slot.begin(), t.expert_slot.end(),
                                       [](int s) { return s >= 0; }), t.cap_slots);
    }
    if (total == 0) {
        return; // ไม่มีการใช้งานรอบนี้ (เช่น อยู่ช่วง PP หรือ idle)
    }
    for (int swaps = 0; swaps < g_max_swap; ++swaps) {
        int   best_in   = -1; float best_in_s   = 0.0f;
        int   worst_out = -1; float worst_out_s = 0.0f;
        for (int64_t e = 0; e < t.ne02; ++e) {
            if (t.expert_slot[e] < 0) {
                if (t.score[e] > best_in_s) { best_in_s = t.score[e]; best_in = (int) e; }
            } else {
                if (worst_out < 0 || t.score[e] < worst_out_s) {
                    worst_out_s = t.score[e]; worst_out = (int) e;
                }
            }
        }
        if (best_in < 0) {
            break;
        }
        int free_slot = -1;
        for (int s = 0; s < t.cap_slots; ++s) {
            if (t.slot_expert[s] < 0 && !t.slot_cooldown[s]) { free_slot = s; break; }
        }
        if (free_slot < 0) {
            // ต้องไล่: คุ้มเมื่อผู้มาใหม่ดีกว่าผู้แย่สุดชัดเจน (กัน thrash)
            if (worst_out < 0 || best_in_s <= worst_out_s * 1.25f + 1.0f) {
                break;
            }
            const int s = t.expert_slot[worst_out];
            expert_pin_set_entry_kernel<<<1, 32, 0, g_side_stream>>>(
                (const void **) t.table_dev, worst_out, t.host_base + (size_t) worst_out * t.expert_size);
            t.expert_slot[worst_out] = -1;
            t.slot_expert[s] = -1;
            t.slot_cooldown[s] = 1; // ช่องนี้ใช้ได้รอบหน้า (กราฟที่กำลังอ่านอยู่จบไปก่อนแน่)
            continue;
        }
        char * dstp = (char *) t.blob_dev + (size_t) free_slot * t.expert_size;
        CUDA_CHECK(cudaMemcpyAsync(dstp, t.host_base + (size_t) best_in * t.expert_size,
                                   t.expert_size, cudaMemcpyHostToDevice, g_side_stream));
        expert_pin_set_entry_kernel<<<1, 32, 0, g_side_stream>>>(
            (const void **) t.table_dev, best_in, dstp);
        t.expert_slot[best_in]   = free_slot;
        t.slot_expert[free_slot] = best_in;
    }
    for (int s = 0; s < t.cap_slots; ++s) {
        if (t.slot_expert[s] < 0 && t.slot_cooldown[s]) { t.slot_cooldown[s] = 0; }
    }
}

// เธรดพื้นหลัง: ดูดตัวนับ → ปรับ pin — อิสระจาก getter (ซึ่งไม่ถูกเรียกตอนกราฟ replay)
static void expert_pin_thread_main(int device) {
    CUDA_CHECK(cudaSetDevice(device));
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(g_refresh_ms));
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto & kv : g_tables) {
            expert_pin_table & t = kv.second;
            if (!t.ready || !t.dyn || t.cap_slots <= 0) {
                continue;
            }
            CUDA_CHECK(cudaMemcpyAsync(t.cnt_host, t.cnt_dev, t.ne02 * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, g_side_stream));
            CUDA_CHECK(cudaMemsetAsync(t.cnt_dev, 0, t.ne02 * sizeof(uint32_t), g_side_stream));
            CUDA_CHECK(cudaStreamSynchronize(g_side_stream));
            expert_pin_dyn_refresh(t);
        }
        CUDA_CHECK(cudaStreamSynchronize(g_side_stream));
    }
}

const void * const * ggml_cuda_expert_pin_table(const ggml_tensor * src0, const ggml_tensor * ids,
                                                cudaStream_t stream) {
    if (!getenv("GGML_EXPERT_PIN") && !getenv("GGML_EXPERT_PIN_DYN")) {
        return nullptr;
    }
    // สนใจเฉพาะ weight ที่อยู่ host buffer (ถ้าอยู่ VRAM อยู่แล้วไม่ต้องทำอะไร)
    if (!src0->buffer || !ggml_backend_buffer_is_host(src0->buffer)) {
        return nullptr;
    }
    if (ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_profile_loaded) {
        expert_pin_load_profile();
        g_profile_loaded = true;
    }

    auto it = g_tables.find(src0->data);
    if (it != g_tables.end()) {
        expert_pin_table & t = it->second;
        if (t.ready && t.dyn && ids && ids->data) {
            // เคอร์เนลนับนี้จะถูก capture ในกราฟ — นับได้ทุกโทเคนแม้ getter ไม่ถูกเรียกอีก
            expert_pin_count_kernel<<<1, 256, 0, stream>>>(
                (const int32_t *) ids->data, (int) ggml_nelements(ids), t.cnt_dev, (int) t.ne02);
        }
        return t.ready ? (const void * const *) t.table_dev : nullptr;
    }

    expert_pin_table t;
    const int64_t ne02        = src0->ne[2];
    const size_t  expert_size = src0->nb[2];

    // ตารางตั้งต้น: ทุกช่องชี้ host (UVA) — ทำงานถูกต้องแม้ไม่มีอะไรถูก pin
    std::vector<const void *> host_table(ne02);
    for (int64_t i = 0; i < ne02; ++i) {
        host_table[i] = (const char *) src0->data + i * expert_size;
    }

    std::vector<int> pins;
    auto pit = g_profile.find(src0->name);
    if (pit != g_profile.end()) {
        pins = pit->second;
    } else if (g_profile.count("*")) {
        for (int64_t i = 0; i < ne02; ++i) {
            pins.push_back((int) i);
        }
    }

    // ทุก alloc/copy ของ creation ใช้ side stream — capture-safe แม้ถูกเรียกกลางการ capture กราฟ
    if (!g_side_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_side_stream, cudaStreamNonBlocking));
    }

    if (g_dyn) {
        t.dyn         = true;
        t.ne02        = ne02;
        t.expert_size = expert_size;
        t.host_base   = (const char *) src0->data;
        t.name        = src0->name;
        CUDA_CHECK(cudaGetDevice(&t.device));
        const size_t per_tensor = g_budget_bytes / (size_t) g_tensors_hint;
        t.cap_slots = (int) std::min<int64_t>((int64_t) (per_tensor / expert_size), ne02);
        t.score.assign(ne02, 0.0f);
        t.slot_expert.assign(std::max(t.cap_slots, 1), -1);
        t.expert_slot.assign(ne02, -1);
        t.slot_cooldown.assign(std::max(t.cap_slots, 1), 0);
        std::vector<int> take;
        for (int e : pins) {
            if (e >= 0 && e < ne02 && (int) take.size() < t.cap_slots) {
                take.push_back(e);
            }
        }
        for (size_t k = 0; k < take.size(); ++k) {
            t.score[take[k]] = (float) (take.size() - k);
        }
        cudaError_t err = cudaSuccess;
        if (t.cap_slots > 0) {
            err = cudaMalloc(&t.blob_dev, (size_t) t.cap_slots * expert_size);
            if (err != cudaSuccess) {
                t.blob_dev = nullptr;
                t.cap_slots = 0;
                take.clear();
            }
        }
        for (size_t k = 0; k < take.size(); ++k) {
            char * dstp = (char *) t.blob_dev + k * expert_size;
            CUDA_CHECK(cudaMemcpyAsync(dstp, host_table[take[k]], expert_size,
                                       cudaMemcpyHostToDevice, g_side_stream));
            host_table[take[k]] = dstp;
            t.slot_expert[k]       = take[k];
            t.expert_slot[take[k]] = (int) k;
        }
        g_spent_bytes  += (size_t) t.cap_slots * expert_size;
        g_pinned_total += (int) take.size();
        CUDA_CHECK(cudaMalloc((void **) &t.cnt_dev, ne02 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemsetAsync(t.cnt_dev, 0, ne02 * sizeof(uint32_t), g_side_stream));
        CUDA_CHECK(cudaMallocHost((void **) &t.cnt_host, ne02 * sizeof(uint32_t)));
    } else {
        // ---- static เดิม ----
        std::vector<int> take;
        for (int e : pins) {
            if (e < 0 || e >= ne02) {
                continue;
            }
            if (g_spent_bytes + expert_size > g_budget_bytes) {
                break;
            }
            g_spent_bytes += expert_size;
            take.push_back(e);
        }
        cudaError_t err = cudaSuccess;
        if (!take.empty()) {
            err = cudaMalloc(&t.blob_dev, take.size() * expert_size);
            if (err == cudaSuccess) {
                for (size_t k = 0; k < take.size(); ++k) {
                    char * dstp = (char *) t.blob_dev + k * expert_size;
                    CUDA_CHECK(cudaMemcpyAsync(dstp, host_table[take[k]], expert_size,
                                               cudaMemcpyHostToDevice, g_side_stream));
                    host_table[take[k]] = dstp;
                }
                g_pinned_total += (int) take.size();
            } else {
                g_spent_bytes -= take.size() * expert_size;
                t.blob_dev = nullptr;
            }
        }
    }

    cudaError_t err = cudaMalloc(&t.table_dev, ne02 * sizeof(void *));
    if (err != cudaSuccess) {
        g_tables[src0->data] = t;   // ready=false — ปิดถาวรสำหรับ tensor นี้
        return nullptr;
    }
    CUDA_CHECK(cudaMemcpyAsync(t.table_dev, host_table.data(), ne02 * sizeof(void *),
                               cudaMemcpyHostToDevice, g_side_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_side_stream));
    t.ready = true;

    static int dbg_create = 0;   // ดีบักชั่วคราว
    if (dbg_create++ < 6) {
        fprintf(stderr, "[expert-pin] %s %s cap=%d/%lld esize=%zuKB\n", src0->name,
                t.dyn ? "dyn" : "pin", t.dyn ? t.cap_slots : g_pinned_total,
                (long long) ne02, expert_size / 1024);
    }

    if (t.dyn && !g_thread_started.exchange(true)) {
        std::thread(expert_pin_thread_main, t.device).detach();
    }
    const void * const * ret = (const void * const *) t.table_dev;
    g_tables[src0->data] = std::move(t);
    return ret;
}
