# API Index

This page provides a complete reference to all classes, methods, and functions available in TwinWeaver. Click on any item to view its detailed documentation.

---

## Quick Import

All main components can be imported directly from `twinweaver`:

```python
from twinweaver import (
    # Core
    Config,
    DataManager,

    # Instruction Tuning
    ConverterInstruction,
    DataSplitter,
    DataSplitterForecasting,
    DataSplitterEvents,

    # Pretraining
    ConverterPretrain,

    # Utilities
    convert_meds_to_dtc,
    identify_constant_and_changing_columns,
    aggregate_events_to_weeks,
)
```

---

## Common Module

### Config

Configuration manager for TwinWeaver settings.

| Member | Type | Description |
|--------|------|-------------|
| [`Config`](reference/common/config.md) | Class | Centralized configuration repository |
| [`Config.set_delta_time_unit`](reference/common/config.md#twinweaver.common.config.Config.set_delta_time_unit) | Method | Set the time unit for delta calculations |
| [`Config.seed`](reference/common/config.md#twinweaver.common.config.Config.seed) | Property | Random seed for reproducibility |

---

### DataManager

Handles data loading and management.

| Member | Type | Description |
|--------|------|-------------|
| [`DataManager`](reference/common/data_manager.md) | Class | Manages data loading, processing, and splitting |
| [`DataManager.load_indication_data`](reference/common/data_manager.md#twinweaver.common.data_manager.DataManager.load_indication_data) | Method | Load data tables for a specific indication |
| [`DataManager.process_indication_data`](reference/common/data_manager.md#twinweaver.common.data_manager.DataManager.process_indication_data) | Method | Process loaded indication data |
| [`DataManager.setup_unique_mapping_of_events`](reference/common/data_manager.md#twinweaver.common.data_manager.DataManager.setup_unique_mapping_of_events) | Method | Create unique mapping for all events |
| [`DataManager.setup_dataset_splits`](reference/common/data_manager.md#twinweaver.common.data_manager.DataManager.setup_dataset_splits) | Method | Split data into train/val/test sets |
| [`DataManager.get_all_patientids_in_split`](reference/common/data_manager.md#twinweaver.common.data_manager.DataManager.get_all_patientids_in_split) | Method | Get all patient IDs in a specific split |
| [`DataManager.get_patient_split`](reference/common/data_manager.md#twinweaver.common.data_manager.DataManager.get_patient_split) | Method | Get the split assignment for a patient |
| [`DataManager.get_patient_data`](reference/common/data_manager.md#twinweaver.common.data_manager.DataManager.get_patient_data) | Method | Retrieve all data for a specific patient |
| [`DataManager.infer_var_types`](reference/common/data_manager.md#twinweaver.common.data_manager.DataManager.infer_var_types) | Method | Infer variable types (numeric/categorical) |

---

### ConverterBase

Base class for all converters.

| Member | Type | Description |
|--------|------|-------------|
| [`ConverterBase`](reference/common/converter_base.md) | Class | Base class for data-to-text conversion |
| [`round_and_strip`](reference/common/converter_base.md#twinweaver.common.converter_base.round_and_strip) | Function | Format numbers with precision control |
| [`ConverterBase.get_difference_in_event_dataframes`](reference/common/converter_base.md#twinweaver.common.converter_base.ConverterBase.get_difference_in_event_dataframes) | Method | Compare two event DataFrames |
| [`ConverterBase.forward_conversion_inference`](reference/common/converter_base.md#twinweaver.common.converter_base.ConverterBase.forward_conversion_inference) | Method | Abstract method for inference conversion |
| [`ConverterBase.generate_target_manual`](reference/common/converter_base.md#twinweaver.common.converter_base.ConverterBase.generate_target_manual) | Method | Abstract method to generate target text |
| [`ConverterBase.aggregate_multiple_responses`](reference/common/converter_base.md#twinweaver.common.converter_base.ConverterBase.aggregate_multiple_responses) | Method | Abstract method to aggregate responses |

---

## Instruction Module

### ConverterInstruction

Main converter for instruction-tuning data.

| Member | Type | Description |
|--------|------|-------------|
| [`ConverterInstruction`](reference/instruction/converter_instruction.md) | Class | Converter combining forecasting and events |
| [`ConverterInstruction.set_custom_summarized_row_fn`](reference/instruction/converter_instruction.md#twinweaver.instruction.converter_instruction.ConverterInstruction.set_custom_summarized_row_fn) | Method | Set custom function for row summarization |
| [`ConverterInstruction.get_nr_tokens`](reference/instruction/converter_instruction.md#twinweaver.instruction.converter_instruction.ConverterInstruction.get_nr_tokens) | Method | Count tokens in a string |
| [`ConverterInstruction.forward_conversion`](reference/instruction/converter_instruction.md#twinweaver.instruction.converter_instruction.ConverterInstruction.forward_conversion) | Method | Convert patient data to text (training) |
| [`ConverterInstruction.forward_conversion_inference`](reference/instruction/converter_instruction.md#twinweaver.instruction.converter_instruction.ConverterInstruction.forward_conversion_inference) | Method | Convert patient data to text (inference) |
| [`ConverterInstruction.generate_target_manual`](reference/instruction/converter_instruction.md#twinweaver.instruction.converter_instruction.ConverterInstruction.generate_target_manual) | Method | Generate target text from reverse conversion |
| [`ConverterInstruction.aggregate_multiple_responses`](reference/instruction/converter_instruction.md#twinweaver.instruction.converter_instruction.ConverterInstruction.aggregate_multiple_responses) | Method | Aggregate multiple LLM responses |
| [`ConverterInstruction.reverse_conversion`](reference/instruction/converter_instruction.md#twinweaver.instruction.converter_instruction.ConverterInstruction.reverse_conversion) | Method | Convert text back to structured data |
| [`ConverterInstruction.get_difference_in_event_dataframes`](reference/instruction/converter_instruction.md#twinweaver.instruction.converter_instruction.ConverterInstruction.get_difference_in_event_dataframes) | Method | Compare predicted vs actual events |

---

### ConverterForecasting

Converter for forecasting tasks.

| Member | Type | Description |
|--------|------|-------------|
| [`ConverterForecasting`](reference/instruction/converter_forecasting.md) | Class | Converter for time-series forecasting |
| [`ConverterForecasting.forward_conversion`](reference/instruction/converter_forecasting.md#twinweaver.instruction.converter_forecasting.ConverterForecasting.forward_conversion) | Method | Convert patient split to text (training) |
| [`ConverterForecasting.forward_conversion_inference`](reference/instruction/converter_forecasting.md#twinweaver.instruction.converter_forecasting.ConverterForecasting.forward_conversion_inference) | Method | Convert patient split to text (inference) |
| [`ConverterForecasting.generate_target_manual`](reference/instruction/converter_forecasting.md#twinweaver.instruction.converter_forecasting.ConverterForecasting.generate_target_manual) | Method | Generate forecasting target text |
| [`ConverterForecasting.aggregate_multiple_responses`](reference/instruction/converter_forecasting.md#twinweaver.instruction.converter_forecasting.ConverterForecasting.aggregate_multiple_responses) | Method | Aggregate multiple forecasting responses |
| [`ConverterForecasting.reverse_conversion`](reference/instruction/converter_forecasting.md#twinweaver.instruction.converter_forecasting.ConverterForecasting.reverse_conversion) | Method | Parse text back to structured forecasts |

---

### ConverterForecastingQA

Converter for forecasting with Q&A format.

| Member | Type | Description |
|--------|------|-------------|
| [`ConverterForecastingQA`](reference/instruction/converter_forecasting_qa.md) | Class | Q&A format forecasting converter |
| [`ConverterForecastingQA.forward_conversion`](reference/instruction/converter_forecasting_qa.md#twinweaver.instruction.converter_forecasting_qa.ConverterForecastingQA.forward_conversion) | Method | Convert to Q&A format text |

---

### ConverterEvents

Converter for event prediction tasks.

| Member | Type | Description |
|--------|------|-------------|
| [`ConverterEvents`](reference/instruction/converter_events.md) | Class | Converter for clinical event prediction |
| [`ConverterEvents.forward_conversion`](reference/instruction/converter_events.md#twinweaver.instruction.converter_events.ConverterEvents.forward_conversion) | Method | Convert patient split to text (training) |
| [`ConverterEvents.forward_conversion_inference`](reference/instruction/converter_events.md#twinweaver.instruction.converter_events.ConverterEvents.forward_conversion_inference) | Method | Convert patient split to text (inference) |
| [`ConverterEvents.generate_target_manual`](reference/instruction/converter_events.md#twinweaver.instruction.converter_events.ConverterEvents.generate_target_manual) | Method | Generate event prediction target |
| [`ConverterEvents.reverse_conversion`](reference/instruction/converter_events.md#twinweaver.instruction.converter_events.ConverterEvents.reverse_conversion) | Method | Parse text back to event predictions |
| [`ConverterEvents.get_difference_in_event_dataframes`](reference/instruction/converter_events.md#twinweaver.instruction.converter_events.ConverterEvents.get_difference_in_event_dataframes) | Method | Compare predicted vs actual events |
| [`ConverterEvents.aggregate_multiple_responses`](reference/instruction/converter_events.md#twinweaver.instruction.converter_events.ConverterEvents.aggregate_multiple_responses) | Method | Aggregate multiple event responses |

---

### DataSplitter

Main data splitter combining forecasting and events.

| Member | Type | Description |
|--------|------|-------------|
| [`DataSplitter`](reference/instruction/data_splitter.md) | Class | Combined data splitter |
| [`DataSplitter.get_splits_from_patient_with_target`](reference/instruction/data_splitter.md#twinweaver.instruction.data_splitter.DataSplitter.get_splits_from_patient_with_target) | Method | Get splits with target data (training) |
| [`DataSplitter.get_splits_from_patient_inference`](reference/instruction/data_splitter.md#twinweaver.instruction.data_splitter.DataSplitter.get_splits_from_patient_inference) | Method | Get splits for inference |

---

### DataSplitterForecasting

Splitter for forecasting data.

| Member | Type | Description |
|--------|------|-------------|
| [`DataSplitterForecasting`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_forecasting.DataSplitterForecasting) | Class | Forecasting-specific data splitter |
| [`DataSplitterForecasting.setup_statistics`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_forecasting.DataSplitterForecasting.setup_statistics) | Method | Compute statistics for normalization |
| [`DataSplitterForecasting.get_splits_from_patient`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_forecasting.DataSplitterForecasting.get_splits_from_patient) | Method | Generate splits for a patient |
| [`DataSplitterForecastingOption`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_forecasting.DataSplitterForecastingOption) | Class | Configuration for forecasting splits |
| [`DataSplitterForecastingGroup`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_forecasting.DataSplitterForecastingGroup) | Class | Grouping for forecasting options |
| [`DataSplitterForecastingGroup.append`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_forecasting.DataSplitterForecastingGroup.append) | Method | Add option to group |

---

### DataSplitterEvents

Splitter for event data.

| Member | Type | Description |
|--------|------|-------------|
| [`DataSplitterEvents`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_events.DataSplitterEvents) | Class | Event-specific data splitter |
| [`DataSplitterEvents.setup_variables`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_events.DataSplitterEvents.setup_variables) | Method | Setup event variables |
| [`DataSplitterEvents.get_splits_from_patient`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_events.DataSplitterEvents.get_splits_from_patient) | Method | Generate splits for a patient |
| [`DataSplitterEventsOption`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_events.DataSplitterEventsOption) | Class | Configuration for event splits |
| [`DataSplitterEventsGroup`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_events.DataSplitterEventsGroup) | Class | Grouping for event options |
| [`DataSplitterEventsGroup.append`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_events.DataSplitterEventsGroup.append) | Method | Add option to group |

---

### BaseDataSplitter

Base class for all data splitters.

| Member | Type | Description |
|--------|------|-------------|
| [`BaseDataSplitter`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_base.BaseDataSplitter) | Class | Abstract base for splitters |
| [`BaseDataSplitter.select_random_splits`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_base.BaseDataSplitter.select_random_splits) | Method | Randomly select split points |
| [`BaseDataSplitter.drop_duplicates_except_na_for_date_col`](reference/instruction/data_splitters.md#twinweaver.instruction.data_splitter_base.BaseDataSplitter.drop_duplicates_except_na_for_date_col) | Method | Remove duplicates preserving NA dates |

---

## Pretrain Module

### ConverterPretrain

Converter for pretraining data.

| Member | Type | Description |
|--------|------|-------------|
| [`ConverterPretrain`](reference/pretrain/converter_pretrain.md) | Class | Converter for pretraining data preparation |
| [`ConverterPretrain.forward_conversion`](reference/pretrain/converter_pretrain.md#twinweaver.pretrain.converter_pretrain.ConverterPretrain.forward_conversion) | Method | Convert patient data to pretraining text |
| [`ConverterPretrain.reverse_conversion`](reference/pretrain/converter_pretrain.md#twinweaver.pretrain.converter_pretrain.ConverterPretrain.reverse_conversion) | Method | Parse pretraining text back to data |

---

## Utils Module

Utility functions for data preprocessing and integration.

| Name | Type | Description |
|------|------|-------------|
| [`convert_meds_to_dtc`](reference/utils/meds_importer.md) | Function | Convert MEDS format data to TwinWeaver format |
| [`identify_constant_and_changing_columns`](reference/utils/preprocessing_helpers.md#twinweaver.utils.preprocessing_helpers.identify_constant_and_changing_columns) | Function | Identify constant vs. time-varying columns |
| [`aggregate_events_to_weeks`](reference/utils/preprocessing_helpers.md#twinweaver.utils.preprocessing_helpers.aggregate_events_to_weeks) | Function | Aggregate event data to weekly intervals |

---

## Module Reference

For complete module documentation with full parameter details, see the [API Reference](reference/common/config.md) section in the navigation.

| Module | Description |
|--------|-------------|
| [`twinweaver.common`](reference/common/config.md) | Core configuration and base classes |
| [`twinweaver.instruction`](reference/instruction/converter_instruction.md) | Instruction-tuning converters and splitters |
| [`twinweaver.pretrain`](reference/pretrain/converter_pretrain.md) | Pretraining data converters |
| [`twinweaver.utils`](reference/utils/meds_importer.md) | Utility functions and integrations |
