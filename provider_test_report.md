# ONNX Benchmark Report: 2026-01-30 00:04:48

**Model ID:** `None`

## Configuration
- **Providers Tested:** `CPUExecutionProvider,VitisAIExecutionProvider`
- **Runs per Config:** 1
- **Tokens generated:** 5

## Results Matrix

| Arch | Provider | TTFT | TPS | Peak RAM | Max Layer RAM |
| :--- | :--- | :--- | :--- | :--- | :--- |

### Prompt: "The nature of artificial intelligence is"
| Monolith | CPUExecutionProvider | 0.1000s | 0.00/s | 260.7MB | N/A |
| Split | CPUExecutionProvider | 0.4699s | 2.10/s | 60.1MB | 125.2MB |
| Monolith | VitisAIExecutionProvider | 0.1000s | 0.00/s | 254.1MB | N/A |
| Split | VitisAIExecutionProvider | 0.6373s | 1.53/s | 53.9MB | 119.1MB |

#### Comparisons (Efficiency vs Performance)
- **Performance King:** Fastest configuration detected for bulk generation.
- **Memory King:** Lowest transient footprint for resource-constrained use cases.

## Model Structure Details
- **Total Model Size:** 34.1 MB
- **Total Components:** 28
