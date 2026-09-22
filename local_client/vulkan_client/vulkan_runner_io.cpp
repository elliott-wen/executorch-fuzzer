/*
 * vulkan_runner_io.cpp — host x86 file-I/O runner for the ExecuTorch Vulkan delegate.
 *
 * Runs one ExecuTorch .pte whose graph was partly delegated to the Vulkan backend. The
 * delegate (vulkan_backend, linked whole-archive so it self-registers) builds its compute
 * pipelines from SPIR-V compiled into the binary at build time, and dispatches them on a
 * Vulkan device at method-load time; any non-delegated ops fall back to the portable CPU
 * kernels (portable_ops_lib). On this headless host the device is Mesa **lavapipe** (CPU
 * software Vulkan) — no GPU, and no emulation layer, because unlike the VGF path this
 * delegate targets core Vulkan compute rather than the ML extensions. vulkan_runner.sh
 * points the loader at lavapipe.
 *
 * Structurally identical to vgf_runner_io.cpp (same Program/Method file-I/O contract) — the
 * only difference is which delegate is LINKED. The delegate runs transparently inside
 * method->execute().
 *
 * Some ops are expected NOT to run here: lavapipe, like SwiftShader, lacks integer dot
 * product and 8-bit int support, which upstream's own test_vulkan_delegate.py marks with
 * @disable_test. Those surface as execute() failures, not as wrong answers.
 *
 * Contract (matches the nxp/cadence/vgf client expectations):
 *   --model model.pte  --inputs in0,in1,..  --output OUTDIR
 * Reads each input file as raw little-endian bytes into the method's input tensor; writes each
 * output tensor's raw bytes to OUTDIR/000i.bin (4-wide zero-padded), which the shell seam
 * normalizes to out_<i>.bin.
 *
 *   exit 0 = ran, outputs written
 *   exit 2 = graph not loadable/runnable (bad program, missing kernel, load failure) -> SKIP
 *   exit 3 = execute() failed (e.g. a Vulkan dispatch error) -> the seam classifies it
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include <sys/stat.h>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

using executorch::extension::FileDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

DEFINE_string(model, "", "Path to serialized .pte model.");
DEFINE_string(inputs, "", "Comma-separated raw input tensor files: in0,in1,...");
DEFINE_string(output, "results", "Directory to write output tensors into.");

// Fixed method/temp allocator pools. Fuzz graphs are small; 256MB each is generous.
static uint8_t method_allocator_pool[256 * 1024 * 1024U];
static uint8_t tmp_allocator_pool[256 * 1024 * 1024U];

static void split_csv(const std::string& s, std::vector<std::string>& out) {
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(tok);
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model.empty()) {
    fprintf(stderr, "--model is required\n");
    return 2;
  }
  std::vector<std::string> inputFiles;
  split_csv(FLAGS_inputs, inputFiles);

  executorch::runtime::runtime_init();

  Result<FileDataLoader> loader = FileDataLoader::from(FLAGS_model.c_str());
  if (!loader.ok()) {
    fprintf(stderr, "Model PTE loading failed\n");
    return 2;
  }
  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    fprintf(stderr, "Program loading failed\n");
    return 2;
  }
  const char* method_name = nullptr;
  {
    const auto r = program->get_method_name(0);
    if (!r.ok()) {
      fprintf(stderr, "Program has no methods\n");
      return 2;
    }
    method_name = *r;
  }
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  if (!method_meta.ok()) {
    fprintf(stderr, "Failed to get method_meta\n");
    return 2;
  }

  MemoryAllocator method_allocator(sizeof(method_allocator_pool), method_allocator_pool);
  MemoryAllocator tmp_allocator(sizeof(tmp_allocator_pool), tmp_allocator_pool);

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  size_t n_planned = method_meta->num_memory_planned_buffers();
  for (size_t id = 0; id < n_planned; ++id) {
    size_t sz = static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    planned_buffers.push_back(std::make_unique<uint8_t[]>(sz));
    planned_spans.push_back({planned_buffers.back().get(), sz});
  }
  HierarchicalAllocator planned_memory({planned_spans.data(), planned_spans.size()});
  MemoryManager memory_manager(&method_allocator, &planned_memory, &tmp_allocator);

  // Vulkan delegate init (instance/device/pipeline creation) happens inside load_method.
  Result<Method> method = program->load_method(method_name, &memory_manager);
  if (!method.ok()) {
    fprintf(stderr, "Loading of method failed with 0x%" PRIx32 "\n",
            (uint32_t)method.error());
    return 2;
  }

  // --- set inputs from raw files ---
  if (method->inputs_size() != inputFiles.size()) {
    fprintf(stderr, "Mismatch: method has %zu inputs, got %zu files\n",
            method->inputs_size(), inputFiles.size());
    return 2;
  }
  {
    std::vector<EValue> values(method->inputs_size());
    Error st = method->get_inputs(values.data(), values.size());
    if (st != Error::Ok) { fprintf(stderr, "get_inputs failed\n"); return 2; }
    for (size_t i = 0; i < values.size(); ++i) {
      FILE* f = fopen(inputFiles[i].c_str(), "rb");
      if (!f) { fprintf(stderr, "cannot open input %s\n", inputFiles[i].c_str()); return 2; }
      fseek(f, 0, SEEK_END);
      long sz = ftell(f);
      fseek(f, 0, SEEK_SET);
      auto t = values[i].toTensor();
      if ((size_t)sz != t.nbytes()) {
        fprintf(stderr, "input %zu size %ld != tensor nbytes %zu\n", i, sz, t.nbytes());
        fclose(f); return 2;
      }
      size_t rd = fread(t.mutable_data_ptr(), 1, (size_t)sz, f);
      fclose(f);
      if (rd != (size_t)sz) { fprintf(stderr, "short read on input %zu\n", i); return 2; }
    }
  }

  // --- execute (Vulkan-delegated subgraphs dispatch to the GPU/lavapipe queue here) ---
  Error status = method->execute();
  if (status != Error::Ok) {
    fprintf(stderr, "Execution failed with 0x%" PRIx32 "\n", (uint32_t)status);
    return 3;
  }

  // --- write outputs as OUTDIR/000i.bin (raw bytes) ---
  struct stat stt;
  if (stat(FLAGS_output.c_str(), &stt) == -1) mkdir(FLAGS_output.c_str(), 0700);
  std::vector<EValue> outs(method->outputs_size());
  status = method->get_outputs(outs.data(), outs.size());
  if (status != Error::Ok) { fprintf(stderr, "get_outputs failed\n"); return 3; }
  for (size_t i = 0; i < outs.size(); ++i) {
    if (!outs[i].isTensor()) continue;  // skip non-tensor outputs
    auto t = outs[i].toTensor();
    int pad = 4 - (int)std::to_string(i).size();
    std::string name = FLAGS_output + "/" + std::string(pad > 0 ? pad : 0, '0') +
        std::to_string(i) + ".bin";
    FILE* f = fopen(name.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", name.c_str()); return 3; }
    fwrite(t.const_data_ptr(), 1, t.nbytes(), f);
    fclose(f);
  }
  printf("OK: %zu outputs written to %s\n", outs.size(), FLAGS_output.c_str());
  return 0;
}
