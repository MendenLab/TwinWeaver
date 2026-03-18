# Prepare Pretraining Data

This script demonstrates how to prepare pretraining data using the `ConverterPretrain` class. Unlike instruction-tuning, pretraining converts full patient timelines into continuous text without explicit question/answer formatting.

Key concepts:

- Setting up `Config` and `DataManager` for pretraining
- Using `ConverterPretrain.forward_conversion()` to convert patient data to text
- Verifying data integrity with `ConverterPretrain.reverse_conversion()`
- Checking round-trip consistency with `get_difference_in_event_dataframes()`

!!! note "Run from project root"
    This script should be run from the root folder of the TwinWeaver repository.

```python title="prepare_pretraining_data.py"
--8<-- "examples/advanced/pretraining/prepare_pretraining_data.py"
```
