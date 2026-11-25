# Individual Project Proposal

## Title
Real-time traffic-sign recognition on a low-power edge CPU

## 1. Problem
Driver-assistance systems on low-cost hardware (e.g. a Raspberry Pi 4-class CPU without a GPU) must recognise traffic
signs from a camera stream in real time. Large CNNs are accurate but too slow and too large for such devices. The goal
is a compact classifier that keeps high accuracy while meeting strict latency and size budgets.

## 2. Dataset
**GTSRB** (German Traffic Sign Recognition Benchmark): 43 classes, ~39 k training and ~12.6 k test images of varying
resolution, resized to 32×32 RGB.
- Split: the official test set for final evaluation; 10% of the training set (stratified by class) for validation.
- Known issues: class imbalance, motion blur, lighting variation → augmentation (colour jitter, small rotations,
  random crop). No horizontal flip, because flipping changes the meaning of some signs.
- Licence: free for research and educational use.

## 3. Success metric
- Primary: **top-1 accuracy ≥ 97%** on the GTSRB test set. Macro-F1 is also reported because of class imbalance.
- The optimized model may lose **at most 1 percentage point** of accuracy compared with the FP32 baseline.

## 4. Deployment constraints
| Constraint | Target | How measured |
|---|---|---|
| Latency | **≤ 50 ms** per image, batch 1 | median of 1000 runs on CPU, 1 thread, after warm-up |
| Model size | **≤ 5 MB** | size of the serialized model file (TorchScript / ONNX) on disk |
| Hardware | CPU only, no GPU | laptop CPU limited to 1 thread (Raspberry Pi 4 as a stretch goal) |

## 5. Approach and planned optimization
1. **Baseline:** a small CNN / MobileNetV2 (width 0.5), trained with the Lab 6 pipeline (YAML config, seed,
   checkpoints, MLflow tracking). Training on GPU (GTX 1650), evaluation on CPU.
2. **Teacher:** ResNet-18, accurate but too large and slow for deployment.
3. **Planned optimization: knowledge distillation (ResNet-18 → small CNN) + post-training INT8 quantization**
   (static quantization with a calibration set). If INT8 loses more than 1 pp, quantization-aware training is used.
   Pruning is the fallback if the model is still larger than 5 MB.
4. Export to TorchScript / ONNX, then benchmark latency and size on the CPU.

## 6. Evaluation plan
One table with accuracy, macro-F1, latency (ms) and size (MB) for: teacher, FP32 student, distilled student,
and INT8 distilled student. All runs are tracked in MLflow, and all configs are stored as YAML.

## 7. Risks and mitigation
- INT8 accuracy drop → quantization-aware training.
- Latency > 50 ms → smaller input (32×32 → 24×24) or a smaller width multiplier (0.35).
- Class imbalance → weighted loss or oversampling.

## 8. Timeline
| Week | Milestone |
|---|---|
| 1 | Data pipeline, baseline model, `data_card.md` |
| 2 | Teacher training + knowledge distillation |
| 3 | INT8 quantization, latency and size benchmarks |
| 4 | `model_card.md`, final report, demo |