Short answer: **Yes. With ONNX Runtime + CUDA EP you _can_ run parallel inference by calling `InferenceSession::Run()` from multiple threads on the _same_ session**—that API is thread-safe. The big caveat: **if you enable CUDA Graphs, you must _not_ call `Run()` from multiple threads** (it’s explicitly unsupported). ([GitHub][1])

---

## What to use to max out one GPU

**Recommended default:** **Single session → many threads → one I/O binding per thread**

- ORT’s CUDA EP uses **per-thread CUDA streams by default**, so concurrent `Run()` calls from different threads can overlap on the GPU. (There’s an option `use_ep_level_unified_stream` that _forces_ a single stream; leave it **false** to keep per-thread streams.) ([ONNX Runtime][2])
- **Do not enable CUDA Graphs** if you need multithreaded concurrency; Graphs force single-thread usage for a session. ([ONNX Runtime][2])

**When to try multi-session:**

- If you _must_ pin the compute stream (`user_compute_stream`) or you want hard isolation of memory/streams, use **one session per worker** (cost: duplicate model state). You can mitigate some duplication with **PrepackedWeightsContainer** to share pre-packed weights across sessions. ([ONNX Runtime][3])

---

## C++ sketch (single session, multi-thread)

```cpp
// 1) Build session with CUDA EP
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "app"};
Ort::SessionOptions so;
OrtCUDAProviderOptionsV2* cuda_opts{};
Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_opts));

// keep default per-thread streams (use_ep_level_unified_stream=false)
Ort::GetApi().UpdateCUDAProviderOptionsWithValue(
    cuda_opts, "do_copy_in_default_stream", "1");  // safer default
Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(so, cuda_opts);

Ort::Session session{env, "model.onnx", so};

// 2) Worker thread pattern
void worker(...) {
  Ort::IoBinding bind{session};

  // Pre-allocate device/pinned buffers, then bind
  Ort::MemoryInfo mi_cuda{"Cuda", OrtArenaAllocator, /*device*/0, OrtMemTypeDefault};
  // e.g., create Ort::Value wrapping your device pointers:
  auto x = Ort::Value::CreateTensor(mi_cuda, d_input, bytes_in, in_shape.data(), in_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  bind.BindInput("input", x);

  // Bind output to device (let ORT allocate on GPU) or to your prealloc:
  bind.BindOutput("output", mi_cuda);

  session.Run(Ort::RunOptions{nullptr}, bind);  // synchronous by default
  // If your inputs/outputs were produced on other streams, you can explicitly sync:
  Ort::GetApi().SynchronizeBoundOutputs(bind);
}
```

Key points:

- **One `Ort::IoBinding` per thread**, and **distinct buffers** per thread (no sharing).
- Use **device tensors** and/or **CUDA-pinned host memory** to overlap copies and compute; ORT recognizes `"Cuda"` and `"CudaPinned"` memory types. ([ONNX Runtime][3])
- Keep `do_copy_in_default_stream=1` unless you fully manage copy/compute stream hazards (doc warns of races). ([ONNX Runtime][4])

---

## Footguns & useful knobs

- **CUDA Graphs:** great for latency, but **no multi-threaded Run()** on a session when graphs are enabled. Use single thread or multi-session if you need graphs. ([ONNX Runtime][2])
- **Forcing one stream:** `use_ep_level_unified_stream=true` (or supplying a `user_compute_stream`) will make _all_ threads share one stream → serializes GPU work. Avoid if you want concurrency. ([ONNX Runtime][2])
- **Threading knobs (`intra_op`/`inter_op`):** these are **CPU-side**; they matter less with CUDA EP but can help/prevent CPU contention. Also consider `ORT_PARALLEL` execution mode for branched graphs. ([ONNX Runtime][5])
- **I/O Binding & sync:** If you bind memory from other streams, call `SynchronizeBoundInputs/Outputs` appropriately. ([ONNX Runtime][6])

---

## TL;DR playbook for L20, single process

1. **Start with one session, N threads (2–8)**, each with its own `IoBinding` and buffers. Leave **per-thread streams** on (default). ([ONNX Runtime][2])
2. **Use device tensors and/or CUDA-pinned host memory** via I/O binding to minimize copies and overlap transfers. ([ONNX Runtime][3])
3. **Only consider multi-session** if you need custom stream pinning or graphs; if you do, use **PrepackedWeightsContainer** to limit duplication. ([ONNX Runtime][7])

If you want, I can drop in a ready-to-run C++ sample that wires up pinned buffers and `IoBinding` in a small thread-pool around your model’s exact input/output shapes.

[1]: https://github.com/microsoft/onnxruntime/discussions/10107?utm_source=chatgpt.com "ONNX Runtime Thread Safety #10107"
[2]: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html?utm_source=chatgpt.com "NVIDIA - CUDA"
[3]: https://onnxruntime.ai/docs/performance/device-tensor.html?utm_source=chatgpt.com "Device tensors"
[4]: https://onnxruntime.ai/docs/api/c/struct_ort_c_u_d_a_provider_options.html?utm_source=chatgpt.com "OrtCUDAProviderOptions Struct Reference"
[5]: https://onnxruntime.ai/docs/performance/tune-performance/threading.html?utm_source=chatgpt.com "Thread management"
[6]: https://onnxruntime.ai/docs/api/c/struct_ort_api.html?utm_source=chatgpt.com "OrtApi Struct Reference"
[7]: https://onnxruntime.ai/docs/api/c/struct_ort_1_1_prepacked_weights_container.html?utm_source=chatgpt.com "Ort::PrepackedWeightsContainer Struct Reference"
