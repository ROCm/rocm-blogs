#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <map>
#include <hip/hip_runtime.h>

#include <rocdecode/rocdecode_version.h>
#include <rocdecode/rocdecode.h>
#include <rocdecode/rocparser.h>

#include <migraphx/migraphx.hpp>

// ============================================================================
// Macros
// ============================================================================

#define CHECK(callable, ...)                                                             \
    do                                                                                   \
    {                                                                                    \
        auto status__ = callable;                                                        \
        if (status__ != ROCDEC_SUCCESS)                                                  \
        {                                                                                \
            std::cerr << "ERROR: " << rocDecGetErrorName(status__) << "; "               \
                      << __FUNCTION__ << "; "                                            \
                      << __FILE__ << ":" << __LINE__ << std::endl;                       \
            std::abort();                                                                \
        }                                                                                \
    } while (false)

#define HIP_CHECK(call)                                                                  \
    do {                                                                                 \
        hipError_t hip_status = call;                                                    \
        if (hip_status != hipSuccess) {                                                  \
            std::cout << __FUNCTION__ << " -> " << #call;                                \
            std::cout << " ERROR " << hipGetErrorName(hip_status) << std::endl;          \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

// ============================================================================
// Global state
// ============================================================================

// HIP and MIGraphX libraries require explicit initialization.
// These globals hold the HIP runtime state used throughout the application.
int              num_devices_   = 0;
hipStream_t      hip_stream_    = nullptr;
hipDeviceProp_t  hip_dev_prop_  = {};

// Define USE_BS to use rocDecode's built-in Bitstream Reader instead of
// manual HEVC NAL-unit parsing.  The reader can auto-detect codec type
// and bit depth from the container.
//#define USE_BS
#ifdef USE_BS
#include <rocdecode/roc_bitstream_reader.h>
RocdecBitstreamReader bs_reader = nullptr;
#endif

rocDecVideoCodec    rocdec_codec_id;
int                 bit_depth;
std::shared_ptr<uint8_t> stream_data = nullptr;
size_t              stream_size = 0;

RocdecVideoParser   parser  = nullptr;
rocDecDecoderHandle decoder = nullptr;
size_t frame_width  = 0;
size_t frame_height = 0;

// Tracks frames currently inside the decoder pipeline.
// Incremented on decode callback, decremented on display callback.
// The value depends on the source stream format and the number of P/B frames.
uint64_t decoding_frames = 0;

std::shared_ptr<migraphx::program> onnx_program;

// ============================================================================
// GPU kernels
// ============================================================================
//
// The source HEVC stream produces frames in NV12 format (YUV colour space).
// To feed the model we need three planar RGB channels, normalized to 32-bit
// float in the range [0..1], and scaled to the expected 224x224 resolution.
//
// The best approach (absent dedicated CSC hardware blocks) is a GPU kernel
// that performs colour-space conversion + resize without ever touching system
// memory.
//
// This is also why a customized LLVM toolchain is required: it allows us to
// build HIP device kernels as part of the application source code.
//
// Three functions are involved:
//   - Clamp - GPU-only helper, called by the kernel.
//   - Nv12ToResizedRgbKernel - kernel, launched from CPU,
//                              executed on GPU.
//   - Nv12ToResizedRGB - CPU-side wrapper that launches the kernel.

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

// Kernel has three possible ways for resizing, which is chosen in compile time:
//   - CSC_RESIZE_STRETCH – stretching original image without keeping aspect
//                          ratio;
//   - CSC_RESIZE_CROP – cropping original image leaving only center of frame with
//                       aspect ratio 1:1;
//   - CSC_RESIZE_KEEP_ASPECT – resizing with keeping aspect ratio of original
//                              image, adding, centering image.
//
// You can choose most applicable resizing approach for better classification
// results, depending on original frame source.

#define CSC_RESIZE_CROP

__global__ static void Nv12ToResizedRgbKernel(uint8_t *y_plane, int y_pitch, uint8_t *uv_plane, int uv_pitch, float* rgb_plane, int src_width, int src_height, int dst_width, int dst_height) {
    int xc = threadIdx.x + blockIdx.x * blockDim.x;
    int yc = threadIdx.y + blockIdx.y * blockDim.y;
    if (xc + 1 >= dst_width || yc + 1 >= dst_height) {
        return;
    }

#if defined(CSC_RESIZE_STRETCH)
    const float x_scale = static_cast<float>(src_width) / dst_width;
    const float y_scale = static_cast<float>(src_height) / dst_height;
    const float xs_f = (xc + 0.5f) * x_scale - 0.5f;
    const float ys_f = (yc + 0.5f) * y_scale - 0.5f;
#elif defined(CSC_RESIZE_CROP)
    const float scale = fminf(static_cast<float>(src_width) / dst_width,
                              static_cast<float>(src_height) / dst_height);
    const float x_offset = (src_width  - dst_width  * scale) * 0.5f;
    const float y_offset = (src_height - dst_height * scale) * 0.5f;
    const float xs_f = (xc + 0.5f) * scale + x_offset - 0.5f;
    const float ys_f = (yc + 0.5f) * scale + y_offset - 0.5f;
#elif defined(CSC_RESIZE_KEEP_ASPECT)
    const float scale = fmaxf(static_cast<float>(src_width) / dst_width,
                              static_cast<float>(src_height) / dst_height);
    const float x_offset = (src_width  - dst_width  * scale) * 0.5f;
    const float y_offset = (src_height - dst_height * scale) * 0.5f;
    const float xs_f = (xc + 0.5f) * scale + x_offset - 0.5f;
    const float ys_f = (yc + 0.5f) * scale + y_offset - 0.5f;
    if (xs_f < 0 || xs_f >= src_width || ys_f < 0 || ys_f >= src_height) {
        // Keep pixel as is
        return;
    }
#endif

    int xs = max(0, min((int)floorf(xs_f), src_width - 1));
    int ys = max(0, min((int)floorf(ys_f), src_height - 1));

    const float maxf = (1 << 8) - 1.0f;

    uint8_t *y_pixel = y_plane + ys * y_pitch + xs;
    uint8_t *u_pixel = uv_plane + (ys / 2) * uv_pitch + (xs / 2) * 2;
    uint8_t *v_pixel = uv_plane + (ys / 2) * uv_pitch + (xs / 2) * 2 + 1;

    int16_t y = *y_pixel - 16, u = *u_pixel - 128, v = *v_pixel - 128;

    int dst_idx = xc + yc * dst_width;
    *(rgb_plane + (dst_width * dst_height * 0) + dst_idx) = (uint8_t)Clamp(1.164383f * y + 1.596027f * v, 0.0f, maxf) / 255.0f;
    *(rgb_plane + (dst_width * dst_height * 1) + dst_idx) = (uint8_t)Clamp(1.164383f * y - 0.391762f * u - 0.812968f * v, 0.0f, maxf) / 255.0f;
    *(rgb_plane + (dst_width * dst_height * 2) + dst_idx) = (uint8_t)Clamp(1.164383f * y + 2.017232f * u, 0.0f, maxf) / 255.0f;
}

void Nv12ToResizedRGB(
    uint8_t *y_plane, int y_pitch,
    uint8_t *uv_plane, int uv_pitch,
    float *rgb_plane,
    int src_width, int src_height,
    int dst_width, int dst_height,
    hipStream_t hip_stream)
{
    Nv12ToResizedRgbKernel
        <<<dim3((dst_width + 16 + 1) / 16, (dst_height + 16 + 1) / 16),
           dim3(16, 16), 0, hip_stream>>>
        (y_plane, y_pitch, uv_plane, uv_pitch, rgb_plane,
         src_width, src_height, dst_width, dst_height);
}

// ============================================================================
// HIP / MIGraphX initialization
// ============================================================================

// Simple HIP initialization: verify GPU availability, select a device, and
// create a HIP stream for the application.
// NOTE: ROCm libraries may also create internal streams for their own purposes.
// This does not affect functionality but can significantly affect performance.
// Also note that "software streams" map onto underlying "hardware streams"
// in a non-obvious way.
bool initHIP(int device_id) {
    HIP_CHECK(hipGetDeviceCount(&num_devices_));
    if (num_devices_ < 1) {
        std::cerr << "ERROR: didn't find any GPU!" << std::endl;
        return false;
    }
    HIP_CHECK(hipSetDevice(device_id));
    HIP_CHECK(hipGetDeviceProperties(&hip_dev_prop_, device_id));
    HIP_CHECK(hipStreamCreateWithFlags(&hip_stream_, 0));
    return true;
}

// MIGraphX initialization: parse the ONNX model, configure compilation
// options, and compile the model for GPU execution.
// set_offload_copy(false) tells MIGraphX that input/output buffers already
// reside in GPU memory and should NOT be copied.  This is required for
// correct pipelining and significantly reduces memory consumption and copying.
// NOTE: MIGraphX compilation is a complex process and may take considerable
// time.  Consider using a save/load approach for the compiled model to
// implement a cache and avoid recompilation on every startup.
void initMIGraphX() {
    migraphx::onnx_options options;
    onnx_program = std::make_shared<migraphx::program>(parse_onnx("./model.onnx", options));
    migraphx::compile_options c_opts;
    c_opts.set_offload_copy(false);
    onnx_program->compile(migraphx::target("gpu"), c_opts);
}

// ============================================================================
// Reading the source stream
// ============================================================================
//
// This tutorial uses a simple HEVC Annex-B stream where NAL units are
// delimited by the byte sequence 0x00 0x00 0x00 0x01.
// If you need a full demuxer, consider integrating FFMPEG.  For this
// tutorial the origin of the stream is not important.  You can extract a
// raw HEVC stream from any video container with:
//   ffmpeg -i input.mkv -c:v copy -bsf:v hevc_mp4toannexb -f hevc output.h265
//
// rocDecode provides a built-in Bitstream Reader (enabled via USE_BS) that
// can also be used instead of manual parsing below.

struct stream_chunk {
    uint8_t *data;
    size_t   size;
    int64_t  pts;
};

// Read the entire raw HEVC file into memory (plain fstream path) or
// initialize the rocDecode Bitstream Reader (USE_BS path).
void read_stream() {
    const char *file_name = "./video.h265";
    std::ifstream fs(file_name, std::ios::binary | std::ios::in);
    if (!fs.is_open()) {
        std::cerr << "Cannot open input file\n";
        std::abort();
    }
    fs.seekg(0, std::ios::end);
    stream_size = static_cast<size_t>(fs.tellg());
#ifndef USE_BS
    fs.seekg(0, std::ios::beg);
    std::shared_ptr<uint8_t> data(new uint8_t[stream_size], [](uint8_t *p) { delete[] p; });
    fs.read(reinterpret_cast<char *>(data.get()), stream_size);
    stream_data = data;
    fs.close();
    rocdec_codec_id = rocDecVideoCodec_HEVC;
    bit_depth = 8;
#else
    if (rocDecCreateBitstreamReader(&bs_reader, file_name) != ROCDEC_SUCCESS) {
        std::cerr << "Failed to create the bitstream reader." << std::endl;
        std::abort();
    }
    if (rocDecGetBitstreamCodecType(bs_reader, &rocdec_codec_id) != ROCDEC_SUCCESS) {
        std::cerr << "Failed to get stream codec type." << std::endl;
        std::abort();
    }
    if (rocDecGetBitstreamBitDepth(bs_reader, &bit_depth) != ROCDEC_SUCCESS) {
        std::cerr << "Failed to get stream bit depth." << std::endl;
        std::abort();
    }
#endif
}

// Parse the raw bitstream into chunks split on the 0x00000001 start-code
// delimiter for further decoding.
// When USE_BS is defined this function does nothing because the
// RocdecBitstreamReader will deliver picture data during decoding.
std::vector<stream_chunk> read_chunks() {
    std::vector<stream_chunk> chunks;
#ifndef USE_BS
    uint8_t *data = stream_data.get(), *prev_data = data;
    uint8_t chunk_type = 0;
    for (size_t i = 0; i < stream_size; ++i) {
        if (*(data + i) == 0 && *(data + i + 1) == 0 &&
            *(data + i + 2) == 0 && *(data + i + 3) == 1) {
            chunk_type = (*(data + i + 4) >> 1) & 0x3f;
            if (chunk_type <= 1) {
                chunks.push_back({prev_data, static_cast<size_t>((data + i) - prev_data), 0});
                prev_data = data + i;
            }
        }
    }
    // Adding last packet
    if((prev_data + 4) <= (data + stream_size)) {
        chunk_type = (*(prev_data + 4) >> 1) & 0x3f;
        if(chunk_type <= 1) {
            chunks.push_back({prev_data, static_cast<size_t>(((data + stream_size) - prev_data)), 0});
        }
    }
#endif
    return chunks;
}

// ============================================================================
// Decoder creation
// ============================================================================

// Initialize the hardware decoder using the format description received from
// the Video Parser's SPS/PPS callback.  Many fields are filled with "known"
// information that depends on what the application already knows about the
// source stream (some of it may come from the container, for example).
// The function also queries Decoder Caps to verify that the resolution and
// codec are supported by the hardware.
void create_decoder(RocdecVideoFormat *format) {
    RocDecoderCreateInfo create_info = {};
    create_info.codec_type           = format->codec;
    create_info.device_id            = 0;
    create_info.chroma_format        = rocDecVideoChromaFormat_420;
    create_info.output_format        = rocDecVideoSurfaceFormat_NV12;
    create_info.bit_depth_minus_8    = format->bit_depth_luma_minus8;
    create_info.width                = format->coded_width;
    create_info.height               = format->coded_height;
    create_info.max_width            = format->coded_width;
    create_info.max_height           = format->coded_height;
    create_info.display_rect.left    = format->display_area.left;
    create_info.display_rect.right   = format->display_area.right;
    create_info.display_rect.top     = format->display_area.top;
    create_info.display_rect.bottom  = format->display_area.bottom;
    create_info.target_width         = (format->display_area.right - format->display_area.left + 1) & ~1;
    create_info.target_height        = (format->display_area.bottom - format->display_area.top + 1) & ~1;
    create_info.num_decode_surfaces  = format->min_num_decode_surfaces;

    frame_width  = create_info.target_width;
    frame_height = create_info.target_height;

    RocdecDecodeCaps decode_caps;
    memset(&decode_caps, 0, sizeof(decode_caps));
    decode_caps.codec_type       = create_info.codec_type;
    decode_caps.chroma_format    = create_info.chroma_format;
    decode_caps.bit_depth_minus_8 = create_info.bit_depth_minus_8;

    rocDecGetDecoderCaps(&decode_caps);
    if (!decode_caps.is_supported) {
        std::cerr << "rocDecode:: Codec not supported on this GPU" << std::endl;
        std::abort();
    }
    if (create_info.max_width > decode_caps.max_width ||
        create_info.max_height > decode_caps.max_height) {
        std::cerr << std::endl
                  << "Resolution          : " << create_info.max_width << "x" << create_info.max_height << std::endl
                  << "Max Supported (wxh) : " << decode_caps.max_width << "x" << decode_caps.max_height << std::endl
                  << "Resolution not supported on this GPU" << std::endl;
        std::abort();
    }

    CHECK(rocDecCreateDecoder(&decoder, &create_info));
}

void destroy_decoder() {
    CHECK(rocDecDestroyDecoder(decoder));
}

// ============================================================================
// Parser callbacks
// ============================================================================

// Called when SPS/PPS NAL units are found in the stream.
// Receives a format descriptor used to (re-)initialize the decoder.
// user_data can be cast to an application object pointer if needed.
int handle_video_sequence(void *user_data, RocdecVideoFormat *format) {
    create_decoder(format);
    return 1;
}

// Called when a picture block is detected by the Video Parser.
// Simply forwards the parameters to the decoder initialized earlier.
// Incrementing decoding_frames signals that the decoder holds frames
// that have not yet been returned via the display callback.
int handle_picture_decode(void *user_data, RocdecPicParams *params) {
    decoding_frames++;
    CHECK(rocDecDecodeFrame(decoder, params));
    return 1;
}

// Called when a decoded frame is ready (in decode order).
// rocDecGetVideoFrame returns GPU memory pointers to the decoded NV12 data.
// This is a blocking call: the decoder guarantees that the data behind these
// pointers will NOT be modified by further decoding as long as we are inside
// this callback.  All processing of the unmodified frame must happen here.
// In a production application you would copy the frame to your own pre-
// allocated memory and process it on a parallel thread; multithreading is
// outside the scope of this tutorial.
int handle_picture_display(void *user_data, RocdecParserDispInfo *disp_info) {
    decoding_frames--;
    RocdecProcParams params = {};
    params.progressive_frame = disp_info->progressive_frame;
    params.top_field_first   = disp_info->top_field_first;

    void     *dev_mem_ptr[3] = {};
    uint32_t  pitch[3]       = {};
    CHECK(rocDecGetVideoFrame(decoder, disp_info->picture_index, dev_mem_ptr, pitch, &params));

    // --- NV12 -> resized RGB (on GPU) ---
    // Allocate GPU memory for three RGB planes (224x224 each, float32).
    // Shared pointers ensure correct deallocation if an exception is thrown.
    // NOTE: hipMalloc/hipFree are relatively expensive; real applications
    // should allocate once outside the callback or use memory pools.
    // hipMemsetD32 is used to initialize memory with 0.0f value.
    const size_t rgb_size = 224 * 224 * 3;
    float *p_dev_mem_ptr = nullptr;
    HIP_CHECK(hipMalloc(&p_dev_mem_ptr, rgb_size * sizeof(float)));
    std::shared_ptr<float> rgb_dev_mem(p_dev_mem_ptr, [](float *p) { HIP_CHECK(hipFree(p)); });
    p_dev_mem_ptr = nullptr;
    HIP_CHECK(hipMemsetD32(rgb_dev_mem.get(), 0.0, rgb_size));

    Nv12ToResizedRGB(
        (uint8_t *)dev_mem_ptr[0], pitch[0],
        (uint8_t *)dev_mem_ptr[1], pitch[1],
        rgb_dev_mem.get(),
        frame_width, frame_height,
        224, 224,
        hip_stream_);

    // --- MIGraphX inference ---
    // Allocate GPU memory for the model output (1000-class confidence vector).
    float *p_out_ptr = nullptr;
    const size_t OUTPUT_SIZE = 1000;
    HIP_CHECK(hipMalloc(&p_out_ptr, OUTPUT_SIZE * sizeof(float)));
    std::shared_ptr<float> output_dev_mem(p_out_ptr, [](float *p) { HIP_CHECK(hipFree(p)); });
    p_out_ptr = nullptr;

    // Bind GPU buffers to the model's named input/output parameters.
    // You can discover parameter names and shapes by iterating
    // param_shapes.names() and printing their lengths.
    migraphx::program_parameters prog_params;
    auto param_shapes = onnx_program->get_parameter_shapes();
    prog_params.add("input", migraphx::argument(param_shapes["input"], rgb_dev_mem.get()));
    prog_params.add("main:#output_0", migraphx::argument(param_shapes["main:#output_0"], output_dev_mem.get()));

    // Run inference asynchronously for correct synchronization between
    // different engines.  Synchronous execution may cause unexpected behavior.
    auto outputs = onnx_program->run_async(prog_params, hip_stream_);

    // --- Copy results back to host ---
    // Async copy followed by an explicit blocking hipStreamSynchronize()
    // ensures all GPU work (kernel + inference + copy) has completed.
    // Note that until this point ALL data manipulation was performed entirely
    // on the GPU, without system memory involvement.
    std::shared_ptr<float> p_output(new float[OUTPUT_SIZE], [](float *p) { delete[] p; });
    HIP_CHECK(hipMemcpyDtoHAsync((void *)p_output.get(), output_dev_mem.get(),
                                  OUTPUT_SIZE * sizeof(float), hip_stream_));
    HIP_CHECK(hipStreamSynchronize(hip_stream_));

    // --- Find the class with highest confidence ---
    // The output is a 1000-element array; the index of the maximum value
    // corresponds to the detected class.
    float  *result  = p_output.get();
    float   max_val = result[0];
    size_t  max_id  = 0;
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
        if (result[i] > max_val) {
            max_val = result[i];
            max_id  = i;
        }
    }

    std::cout << "Frame detected class: " << max_id
              << "  confidence: " << max_val << std::endl;

    return 1;
}

// ============================================================================
// Parser creation & decoding
// ============================================================================

// Create the rocDecode Video Parser.
// The parser is callback-driven and runs in the caller's thread: when you
// call rocDecParseVideoData it may invoke your callbacks synchronously.
// This design can affect performance and should be taken into account.
//
// Callback overview:
//   pfn_sequence_callback - invoked when SPS/PPS units are found;
//                           used to initialize or reinitialize the decoder.
//   pfn_decode_picture    - invoked when a picture block is found.
//   pfn_display_picture   - invoked when a frame is ready (decode order).
void create_parser() {
    RocdecParserParams params        = {};
    params.codec_type                = rocdec_codec_id;
    params.clock_rate                = 1000;
    params.max_num_decode_surfaces   = 10;
    params.max_display_delay         = 1;
    params.user_data                 = nullptr;
    params.pfn_sequence_callback     = handle_video_sequence;
    params.pfn_decode_picture        = handle_picture_decode;
    params.pfn_display_picture       = handle_picture_display;
    CHECK(rocDecCreateVideoParser(&parser, &params));
}

void destroy_parser() {
    CHECK(rocDecDestroyVideoParser(parser));
}

// Feed data chunks to the Video Parser; everything else happens inside the
// parser through the registered callbacks.
// IMPORTANT: the last chunk must carry ROCDEC_PKT_ENDOFSTREAM, otherwise
// the parser may not return the final frames.
void decode_frames(const std::vector<stream_chunk> &chunks) {
#ifdef USE_BS
    int n_video_bytes = 0;
    uint8_t *pvideo = nullptr;
    int64_t pts = 0;
    do {
        RocdecSourceDataPacket packet = {};
        if (rocDecGetBitstreamPicData(bs_reader, &pvideo, &n_video_bytes, &pts) != ROCDEC_SUCCESS) {
            std::cerr << "Failed to get picture data." << std::endl;
            std::abort();
        }
        if (n_video_bytes == 0)
            packet.flags = ROCDEC_PKT_ENDOFSTREAM;
        packet.payload_size = n_video_bytes;
        packet.payload      = pvideo;
        CHECK(rocDecParseVideoData(parser, &packet));
    } while (n_video_bytes);
#else
    for (int i = 0; i < static_cast<int>(chunks.size()); ++i) {
        RocdecSourceDataPacket packet = {};
        if (i == static_cast<int>(chunks.size() - 1))
            packet.flags = ROCDEC_PKT_ENDOFSTREAM;
        packet.payload_size = chunks[i].size;
        packet.payload      = chunks[i].data;
        CHECK(rocDecParseVideoData(parser, &packet));
    }
#endif
    // Flush remaining frames from the decoder after the stream has ended.
    RocdecSourceDataPacket packet = {};
    packet.flags = ROCDEC_PKT_ENDOFSTREAM;
    while (decoding_frames > 0) {
        CHECK(rocDecParseVideoData(parser, nullptr));
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    if (!initHIP(0))
        return 1;

    initMIGraphX();

    read_stream();
    auto chunks = read_chunks();

    create_parser();
    decode_frames(chunks);
    destroy_decoder();
    destroy_parser();
    if (hip_stream_) {
        hipError_t hip_status = hipSuccess;
        hip_status = hipStreamDestroy(hip_stream_);
        if (hip_status != hipSuccess) {
            std::cerr << "ERROR: hipStream_Destroy failed! (" << hip_status << ")" << std::endl;
        }
    }
    onnx_program.reset();

    return 0;
}
