# ONNX Benchmark Report: 2026-01-30 00:14:02

**Model ID:** `None`

## Configuration
- **Providers Tested:** `CPUExecutionProvider`
- **Runs per Config:** 1
- **Tokens generated:** 5

## Results Matrix

### Prompt: "The nature of artificial intelligence is"

| Arch | Provider | TTFT | TPS | Peak RAM | Quality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Monolith | CPUExecutionProvider | 0.1000s | 0.00/s | 255.7MB | N/A |
| Split | CPUExecutionProvider | 0.4874s | 2.09/s | 59.7MB | MATCH ✅ |

#### Comparisons (Efficiency vs Performance)
- **Performance King:** Fastest configuration detected for bulk generation.
- **Memory King:** Lowest transient footprint for resource-constrained use cases.

## Model Structure Details
- **Total Model Size:** 34.1 MB
- **Total Components:** 28
