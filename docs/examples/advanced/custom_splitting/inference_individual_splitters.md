# Custom Splitting for Inference

This script demonstrates how to use individual splitters (`DataSplitterForecasting` and `DataSplitterEvents`) combined via the unified `DataSplitter` API for **inference** — i.e., when you only have input data and no target labels.

Key concepts:

- Configuring split events and forecasting categories
- Using `DataSplitter.get_splits_from_patient_inference()` to generate splits at inference time
- Converting splits to text prompts with `ConverterInstruction.forward_conversion_inference()`
- Overriding which variables/events to predict

!!! note "Run from project root"
    This script should be run from the root folder of the TwinWeaver repository.

```python title="inference_individual_splitters.py"
--8<-- "examples/advanced/custom_splitting/inference_individual_splitters.py"
```
