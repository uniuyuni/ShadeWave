// numpy 配列と Metal 間のゼロコピーバインディング。
// ページ境界に丸めて newBufferWithBytesNoCopy で包み、失敗時は従来のコピーに
// フォールバックする。呼び出し側は GPU 完了(waitUntilCompleted)まで元配列を
// 生かしておくこと(各バックエンドは同期実行なので自然に満たされる)。
#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <unistd.h>

inline NSUInteger metal_page_size_bytes() {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        page_size = 4096;
    }
    return static_cast<NSUInteger>(page_size);
}

struct BufferBinding {
    id<MTLBuffer> buffer;
    NSUInteger offset;
    bool no_copy;
};

inline bool make_no_copy_binding(id<MTLDevice> device, void* ptr, size_t bytes, BufferBinding* binding) {
    const NSUInteger page_size = metal_page_size_bytes();
    std::uintptr_t address = reinterpret_cast<std::uintptr_t>(ptr);
    std::uintptr_t base_address = address & ~(static_cast<std::uintptr_t>(page_size) - 1);
    NSUInteger offset = static_cast<NSUInteger>(address - base_address);
    NSUInteger wrapped_length = static_cast<NSUInteger>(bytes) + offset;
    NSUInteger rounded_length = ((wrapped_length + page_size - 1) / page_size) * page_size;

    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(base_address)
                                                     length:rounded_length
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    if (!buffer) {
        return false;
    }
    binding->buffer = buffer;
    binding->offset = offset;
    binding->no_copy = true;
    return true;
}

inline BufferBinding make_buffer_for_input(id<MTLDevice> device, const void* ptr, size_t bytes) {
    BufferBinding binding{};
    if (make_no_copy_binding(device, const_cast<void*>(ptr), bytes, &binding)) {
        return binding;
    }
    binding.buffer = [device newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
    binding.offset = 0;
    binding.no_copy = false;
    return binding;
}

inline BufferBinding make_buffer_for_output(id<MTLDevice> device, void* ptr, size_t bytes) {
    BufferBinding binding{};
    if (make_no_copy_binding(device, ptr, bytes, &binding)) {
        return binding;
    }
    binding.buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    binding.offset = 0;
    binding.no_copy = false;
    return binding;
}

// no_copy でなかった場合のみ、GPU 出力バッファから宛先へコピーする。
inline void finish_output_binding(const BufferBinding& binding, void* dst, size_t bytes) {
    if (!binding.no_copy) {
        memcpy(dst, [binding.buffer contents], bytes);
    }
}
