// [expert-pin 2026-09-01] static hot-expert pinning สำหรับ MoE ที่ expert อยู่ pinned host
// ใช้คู่กับ -ot '...exps...=ROCm_Host' + GGML_OP_OFFLOAD_MIN_BATCH=1
// เปิดด้วย GGML_EXPERT_PIN=<profile> (บรรทัด: "<tensor name> <expert id>" เรียงตามความถี่)
// งบ VRAM: GGML_EXPERT_PIN_MB (ค่าเริ่มต้น 2800)
// คืนตารางพอยน์เตอร์บน device ขนาด ne02 ช่อง: ช่องที่ pin ชี้ VRAM, ที่เหลือชี้ host (UVA)
// คืน nullptr เมื่อปิดใช้งาน/ไม่เข้าเงื่อนไข (weight ไม่ได้อยู่ host ฯลฯ)
#pragma once
#include "common.cuh"

// v2: โหมด dynamic (GGML_EXPERT_PIN_DYN=1) — cache ปรับตัวเองจากความถี่จริง (พอร์ตแนวคิด FreeToken)
// ids = tensor เส้นทาง routing (I32 บน device) ใช้นับความถี่ · ปรับด้วย GGML_EXPERT_PIN_REFRESH/SWAP/TENSORS
const void * const * ggml_cuda_expert_pin_table(const ggml_tensor * src0, const ggml_tensor * ids,
                                                cudaStream_t stream);
