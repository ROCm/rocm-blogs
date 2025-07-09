---
blogpost: true
blog_title: "HIP 7.0 Is Coming: What You Need to Know to Stay Ahead"
date: 28 May 2025
author: 'Christophe Paquot, Julia Jiang, Denny Iriawan, Saad Rahim'
thumbnail: 'hip-runtime-7.jpg'
tags: Compiler, Developers, HPC
category: Ecosystems and Partners
target_audience: Users coding applications with HIP C++
key_value_propositions: This blog assists users to transition code to HIP 7.0 and navigate the breaking changes in backward compatibility
language: English
myst:
    html_meta:
        "author": "Christophe Paquot, Julia Jiang, Denny Iriawan, Saad Rahim"
        "description lang=en": "Get ready for HIP 7.0—explore key API changes that boost CUDA compatibility and streamline portable GPU development, start preparing your code today."
        "keywords": "AMD, HIP, C++, runtime"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_topic_categories": "HPC & Scientific Computing, Software & Ecosystem"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_developer_type": "Software Developer"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_hardware_platforms": 'Instinct GPUs'
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "Design, Simulation & Modeling"
        "amd_blog_topic_categories": "HPC & Scientific Computing"
---
<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# HIP 7.0 Is Coming: What You Need to Know to Stay Ahead

At AMD, we understand that code portability between AMD and NVIDIA GPU programming models is top of mind
for our customers. We are committed to making GPU development more seamless and portable across vendors. With
the upcoming HIP 7.0 release in second half of 2025, we’re taking a bold step toward simplifying cross-platform programming by aligning HIP C++ even more closely with CUDA. AMD tightly integrates our automatic
HIPIFY conversion tool with our HIP runtime and compiler. Users can quickly port CUDA code into HIP C++ with HIPIFY to target AMD GPUs. However, small
differences between our implementation of the HIP C++ programming model and CUDA C++ often require manual intervention to adjust your code base. This causes
additional work for software developers targeting GPU families from both providers. We understand this and are making
changes to ROCm to reduce this friction based on customer requests. We also know adopting changes in our programming model requires
early notification. We don't take API breaking changes lightly and for your benefit, we are making an early prototype available to assist in porting to the new HIP 7.0 API.
The preview release is based on ROCm 6.4.1 release for functionality but contains 7.0 API previews. It is intended as a drop-in replacement
for 6.4.1 intended for non-production use, enabling users to write code with the new API and adopt HIP 7.0 more smoothly. In this blog, you will learn how HIP 7.0 aligns more closely with CUDA, what API and behavior changes to expect, and how to prepare your codebase to ensure compatibility and portability across GPU platforms. Let's delve into the details of the API changes.

## How will you be impacted?

HIP 7.0 will be available with our next major ROCm release in the second half of 2025. Code compiled for HIP version 6 series of releases may not
work with HIP version 7.0 without recompilation. In some cases, updates to the source code are required.
Planned API changes reduce behavior differences between CUDA C++ and HIP C++ to simplify writing portable GPU code. API calls in HIP now align more closely to their equivalent CUDA calls. In addition,
header files are cleaned up to remove namespace collisions and have a clear separation between hipRTC and the HIP runtime.

## How to get the HIP runtime preview?

The preview is available as source delta on top of the ROCm 6.4.1 release. Please see the source code on the [HIP](https://github.com/ROCm/hip/tree/rocm-6.4.x-with-7.0-preview) and [CLR](https://github.com/ROCm/clr/tree/rocm-6.4.x-with-7.0-preview) repositories on Github. The binary build is also available on [Github](https://github.com/ROCm/clr/releases/tag/rocm-6.4.1-with-7.0-preview). Please build it from source yourself or download the prebuilt package. This release is intended for development purposes only and not for production workloads. Only the HIP runtime build is provided with this tag.

## Changes in the HIP Runtime API

### hipGetLastError

Before ROCm 7.0, ```hipGetLastError``` was not fully compliant with CUDA’s behavior. The purpose of the change is to get the error code returned by ```hipGetLastError``` which should be the last actual error caught in the current thread during the application execution, neither ```hipSuccess``` nor ```hipErrorNotReady``` is considered an error.

Take the following codes as an example:

```C++
1: hipError_t err = hipMalloc(...); // returns hipOutOfMemory
2: err = hipSetDevice(0); // returns hipSuccess
3: err = hipGetLastError();
```

The current behavior prior to ROCm 7.0 is as follows.  At line 1, ```hipMalloc``` returns the ```hipOutOfMemory``` error code, and, at line 2, ```hipSetDevice``` returns ```hipSuccess```. Subsequently, calling ```hipGetLastError``` in line 3,  the value of ```err``` is ```hipSuccess```. Starting from the upcoming release ROCm 7.0, the value of ```err``` at line 3 is ```hipOutOfMemory``` to match CUDA.

To use the old functionality, we have a function called ```hipExtGetLastError```. Note that the function starts with ```hipExt```. This denotes a function call that is unique to AMD and extends the CUDA Runtime API. This function is available today (and was introduced with ROCm 6.0).

### Cooperative Groups

For ```hipLaunchCooperativeKernelMultiDevice``` function, AMD added additional input parameter validation checks.

* If the input launch stream is a NULLPTR or it is ```hipStreamLegacy```, the function now returns ```hipErrorInvalidResourceHandle```.
* If the stream capturing is active, the function returns the error code ```hipErrorStreamCaptureUnsupported```.
* If the stream capture status is invalidated, the function returns the error ```hipErrorStreamCaptureInvalidated```.

The ```hipLaunchCooperativeKernel``` function now checks the input stream handle. If it’s invalid, the returned error is changed to ```hipErrorInvalidHandle``` from ```hipErrorContextIsDestroyed```.

### ```hipPointerGetAttributes```

```hipPointerGetAttributes``` now matches the functionality of ```cudaPointerGetAttributes``` which changed with CUDA 11 and above. If a NULL host or attribute pointer is passed as input parameter, ```hipPointerGetAttributes``` now returns ```hipSuccess``` instead of the error code ```hipErrorInvalidValue```.

Any application which is expecting the API to return an error instead of success could be impacted and code change may need to handle the error properly.

### ```hipFree```

```hipFree``` currently has an implicit wait which is applicable for all memory allocations, for synchronization purpose. This wait will be disabled for allocations made with hipMallocAsync and hipMallocFromPoolAsync to match the behavior of CUDA API ```cudaFree```

### hipRTC

Runtime compilation for HIP is available through the hipRTC library. The library grew organically within the main HIP runtime code. However, segregation of the hipRTC code is now needed to ensure better compatibility and easier code portability.

#### Removal of ```hipRTC``` symbols from HIP Runtime Library

```hipRTC``` has been an independent library since ROCm 6.0 release, but the ```hipRTC``` symbols were still available in the HIP runtime library. Starting with ROCm 7.0, they will be removed.

Any application using ```hipRTC``` APIs should link explicitly with the ```hipRTC``` library.

This change makes the usage of ```hipRTC``` library on Linux the same as on Windows and matches the behavior of CUDA ```nvRTC```.

#### ```hipRTC``` compilation

The device code compilation via ```hipRTC``` now uses namespace ```__hip_internal```, instead of the standard headers ```std```, to avoid namespace collision. These changes are made in the HIP header files.

No code change is required in any application, but rebuilding is necessary.

#### Removal of datatypes from ```hipRTC```

In ```hipRTC```, datatype definitions such as ```int64_t```, ```uint64_t```, ```int32_t```, and ```uint32_t```, etc. could result in conflicts in some applications, as they use their own definitions for these types. ```nvRTC``` doesn't define these datatypes either.
These datatypes are removed and replaced by HIP internal datatypes prefixed with ```__hip```, for example, ```__hip_int64_t```.

Any application relying on HIP internal datatypes during “hipRTC” compilation can be impacted.
These changes have no impact on any application if compiles fine using “nvRTC”.

### HIP Header Clean Up

#### Usage of STD headers

The HIP header files included unnecessary STL headers. Starting with ROCm 7.0, they will only include the necessary ones.

Applications relying on HIP runtime header files may potentially have compilation issues, they need to update including STL headers for the fix.

#### Deprecated Structure

The deprecated structure ```HIP_MEMSET_NODE_PARAMS``` is removed.
Developers can use the definition ```hipMemsetParams``` instead, as input parameter, while using these two APIs:

* ```hipDrvGraphAddMemsetNode``` and
* ```hipDrvGraphExecMemsetNodeSetParams```.

### API Signature/Struct Changes

#### API Signature changes

Signatures are adjusted in some APIs to match corresponding CUDA APIs.

The RTC method definition is changed in the following ```hipRTC``` APIs:

* ```hiprtcCreateProgram```
* ```hiprtcCompileProgram```

In these APIs, the input parameter type changes from ```const char**``` to ```const char* const*```.

In addition, the following APIs have signature changes:

* ```hipMemcpyHtoD```, the type of the second argument pointer changes from ```const void*``` to ```void*```.
* ```hipCtxGetApiVersion``` the type of second argument is changed from ```int*``` to ```unsigned int*```.

These signature changes do not require code modifications but do require rebuilding the application.

#### HIP Struct Change

The struct ```hipMemsetParams``` is updated to be compatible with CUDA.
The change is from the old struct definition:

```C++
typedef struct hipMemsetParams {
  void* dst;
  unsigned int elementSize;
  size_t height;
  size_t pitch;
  unsigned int value;
  size_t width;
} hipMemsetParams;
```

To the new struct definition:

```C++
typedef struct hipMemsetParams {
  void* dst;
  size_t pitch;
  unsigned int value;
  unsigned int elementSize;
  size_t width;
  size_t height;
} hipMemsetParams;
```

No code change is required in any application using this structure, but rebuilding is necessary.

#### HIP Vector Constructor Change

The changes will be made in HIP vector constructors for ```hipComplex``` initialization, to generate correct values. The affected constructors will be small vector types such as ```float2```, ```int4```, etc.
If relying on a single value to initialize all components within a vector or complex type, you will need to update your code. Otherwise, no code change is required in any application using these constructors, but rebuilding is necessary.

### Stream Capture updates

#### Restricts Stream Capture Mode

Stream capture mode will be restricted in HIP APIs through the addition of the macro ```CHECK_STREAM_CAPTURE_SUPPORTED ()```.

In the current HIP enumeration ```hipStreamCaptureMode```, three capture modes are defined. When checking in the macro, the only supported stream capture mode is ```hipStreamCaptureModeRelaxed```. The rest are not supported, and the macro will return ```hipErrorStreamCaptureUnsupported```.

This change matches the behavior of CUDA. There will be no impact on any application if stream capture works fine on the CUDA platform. However, on the AMD platform, the API return code will be adjusted on unsupported stream capture modes.

This update involves the following APIs. They are allowed only in relaxed stream capture mode. Not all three capture modes.

* ```hipMallocManaged```,
* ```hipMemAdvise```.

#### Checks Stream Capture Mode

The following APIs will check the stream capture mode and return error codes to match the behavior of CUDA. No impact if stream capture is working fine on CUDA. Otherwise, the application would need to tweak the graph that is being captured.

* ```hipLaunchCooperativeKernelMultiDevice``` -

Returns error code while stream capture status is active.
The usage is restricted during stream capture.

* ```hipEventQuery``` -

Returns an error ```hipErrorStreamCaptureUnsupported``` in global capture mode.

* ```hipStreamAddCallback``` -

The stream capture behavior is updated. The function checks if any of the blocking streams is capturing, if so, returns an error and invalidates all capturing streams.
The usage of this API is restricted during stream capture to match CUDA.

#### Returns Error During Stream Capture

During stream capture, the following HIP APIs should return specific error ```hipErrorStreamCaptureUnsupported``` on the AMD platform, but not always ```hipSuccess```, to match behavior with CUDA.

* ```hipDeviceSetMemPool```
* ```hipMemPoolCreate```
* ```hipMemPoolDestroy```
* ```hipDeviceSetSharedMemConfig```
* ```hipDeviceSetCacheConfig```

The usage of these APIs is restricted during stream capture. No impact if stream capture is working fine on CUDA.

### Error code changes

Returned error/value codes are updated in the following HIP APIs to match the corresponding CUDA APIs.
The APIs have been updated to return new or additional error codes. Most applications just check if “hipSuccess” is returned,
and so no change is needed. However, if an application checks for a specific error code, the application code may need to be updated to match/handle the new error code accordingly.

Your application can check for both the existing and updated error codes so that it will run in both the current and preview releases.

This update involves the following APIs:

#### Module Management Related APIs

##### Kernel Launch APIs

The following APIs have updated implementations:

* ```hipModuleLaunchKernel```
* ```hipExtModuleLaunchKernel```
* ```hipExtLaunchKernel```
* ```hipDrvLaunchKernelEx```
* ```hipLaunchKernel```
* ```hipLaunchKernelExC```
More conditional checks are added in the API implementation, the return errors are added or changed in the following scenarios:

* If the input stream handle is invalid, the returned error is changed to ```hipErrorContextIsDestroyed``` from ```hipErrorInvalidValue```.
* Adds grid dimension check, if any of them is zero, any input global work size dimension is zero, returns ```hipErrorInvalidValue```.
* Adds extra shared memory size check, if exceeds the size limit, returns ```hipErrorInvalidValue```.
* If the total number of threads per block exceeds the maximum work group limit during a kernel launch, the return value is changed to```hipErrorInvalidConfiguration``` from ```hipErrorInvalidValue```.

##### ```hipModuleLaunchCooperativeKernel```

More conditions are added in the API implementation, the returned errors are added in the following scenarios:

* If the input stream is invalid, returns ```hipErrorContextIsDestroyed```, instead of ```hipErrorInvalidValue```.
* If any grid dimension or block dimension is zero, returns ```hipErrorInvalidValue```.
* If any grid dimension exceeds maximum dimension limit, or work group size exceeds the maximum size, returns ```hipErrorInvalidConfiguration``` , instead of ```hipErrorInvalidValue``` .
* If shared memory size in bytes exceeds the device local memory size per CU, returns ```hipErrorCooperativeLaunchTooLarge```.

##### ```hipModuleLoad```

The API updates the negative return, in case the file name exists but the file size is 0, returns ```hipErrorInvalidImage``` instead of ```hipErrorInvalidValue```, to match the behavior with CUDA.

#### Texture Management Related APIs

The following APIs update the return codes to match the behavior with CUDA:

* ```hipTexObjectCreate```, supports zero width and height for 2D image. If either is zero, will not return ```false```.
* ```hipBindTexture2D```, adds extra check, if pointer for texture reference or device is NULL, returns ```hipErrorNotFound```.
* ```hipBindTextureToArray```, if any NULL pointer is input for texture object, resource descriptor, or texture descriptor, returns error ```hipErrorInvalidChannelDescriptor```, instead of ```hipErrorInvalidValue```.
* ```hipGetTextureAlignmentOffset```, adds a return code ```hipErrorInvalidTexture``` when the texture reference pointer is NULL.

#### Cooperative Group Related APIs

##### ```hipLaunchCooperativeKernelMultiDevice```

More validations are added in the API implementation, as follows:

* If input launch stream is NULLPTR or it is ```hipStreamLegacy```, returns ```hipErrorInvalidResourceHandle```.
* If the stream capturing is active, returns the error ```hipErrorStreamCaptureUnsupported```.
* If the stream capture status is invalidated, returns the error ```hipErrorStreamCaptureInvalidated```
* If the total number of threads per block exceeds the maximum work group limit during a kernel launch, the return value is changed to ```hipErrorInvalidConfiguration```  from ```hipErrorInvalidValue```.

##### ```hipLaunchCooperativeKernel```

Additional validation are added to the API implementation of ```hipLaunchCooperativeKernel```. These are follows:

* If the input stream handle is invalid, the returned error is changed to hipErrorInvalidHandle from hipErrorContextIsDestroyed.
* If the total number of threads per block exceeds the maximum work group limit during a kernel launch, the return value is changed to ```hipErrorInvalidConfiguration``` from ```hipErrorInvalidValue``` .

### Invalid stream input parameter handling matches CUDA

In order to match the CUDA runtime behavior more closely, HIP APIs with streams passed as input parameters no longer check the stream validity. Currently, the HIP runtime returns an error code ```hipErrorContextIsDestroyed``` if the stream is invalid. In CUDA version 12 and later, the equivalent behavior is to raise a segmentation fault. With HIP 7.0, the HIP runtime matches the CUDA by causing a segmentation fault. The list of APIs impacted by this change are as follows:

* Stream Management Related APIs
  * ```hipStreamGetCaptureInfo```
  * ```hipStreamGetPriority```
  * ```hipStreamGetFlags```
  * ```hipStreamDestroy```
  * ```hipStreamAddCallback```
  * ```hipStreamQuery```
  * ```hipLaunchHostFunc```
* Graph Management Related APIs
  * ```hipGraphUpload```
  * ```hipGraphLaunch```
  * ```hipStreamBeginCaptureToGraph```
  * ```hipStreamBeginCapture```
  * ```hipStreamIsCapturing```
  * ```hipStreamGetCaptureInfo```
  * ```hipGraphInstantiateWithParams```
* Memory Management Related APIs
  * ```hipMemcpyPeerAsync```
  * ```hipMemcpy2DValidateParams```
  * ```hipMallocFromPoolAsync```
  * ```hipFreeAsync```
  * ```hipMallocAsync```
  * ```hipMemcpyAsync```
  * ```hipMemcpyToSymbolAsync```
  * ```hipStreamAttachMemAsync```
  * ```hipMemPrefetchAsync```
  * ```hipDrvMemcpy3D```
  * ```hipDrvMemcpy3DAsync```
  * ```hipDrvMemcpy2DUnaligned```
  * ```hipMemcpyParam2D```
  * ```hipMemcpyParam2DAsync```
  * ```hipMemcpy2DArrayToArray```
  * ```hipMemcpy2D```
  * ```hipMemcpy2DAsync```
  * ```hipDrvMemcpy2DUnaligned```
  * ```hipMemcpy3D```
* Event Management Related APIs
  * ```hipEventRecord```
  * ```hipEventRecordWithFlags```

Users porting CUDA code no longer needed to modify their error handling code. However, users that have come to expect the current functionality where the runtime returns the error code ```hipErrorContextIsDestroyed``` will have to adjust.

### ```warpSize``` Change

In order to match the CUDA specification, the `warpSize` variable is no longer `constexpr`. In general, this should be a transparent change; however, if an application was using `warpSize` as a compile-time constant, it will have to be updated to handle the new definition. For more information, see either the discussion of the `warpSize` change within the ROCm 6.4.1 [deprecation notice](https://rocm.docs.amd.com/en/latest/about/release-notes.html#amdgpu-wavefront-size-compiler-macro-deprecation) or the [HIP C++ language extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#warpsize).

## Summary

HIP 7.0 is designed to enhance GPU code portability and simplify cross-vendor GPU development. By aligning HIP more closely with CUDA semantics,
refining error handling, and streamlining header structures, the HIP 7.0 release reduces the effort needed to maintain portable codebases. This blog
 outlines the key updates and provides guidance to help developers prepare their code for HIP 7.0. You are encouraged to begin testing with the HIP
 7.0 preview to ensure a seamless transition and smooth upgrade path when we go live later during the second half of 2025.

```{update} 20 Jun 2025
The link to the preview version of the 7.0 HIP API is now updated for the ROCm 6.4.1 release.
```
