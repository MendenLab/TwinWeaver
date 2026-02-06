# Pro Tips

This page contains practical advice for debugging, scaling, and optimizing your TwinWeaver workflows.

---

## Debugging & Development

### Use a Subset of Patients

When debugging or developing new features, **use a subset of patients** rather than limiting the number of events. This approach:

- Maintains realistic patient trajectories with complete event sequences
- Reduces data loading and processing time significantly
- Helps identify issues with edge cases in patient histories

```python
# Example: Filter to a subset of patients for debugging
patient_subset = df["patientid"].unique()[:100]  # First 100 patients
df_debug = df[df["patientid"].isin(patient_subset)]
```

### Use Simpler Interface Libraries

For debugging and prototyping, use **Hugging Face Transformers** directly for both training and inference:

- Easier to debug with clear error messages
- More flexible for experimentation
- Well-documented with extensive community support

---

## Scaling Up

### Enable Flash Attention

For larger-scale training and inference, ensure **Flash Attention** is enabled for significant memory and speed improvements:

!!! note "Requirements"
    Flash Attention requires compatible hardware (Ampere GPUs or newer) and the `flash-attn` package:
    ```bash
    pip install flash-attn --no-build-isolation
    ```

### Use Specialized Deployment Libraries

For production-scale inference, consider using **vLLM** with prefix caching enabled:

- **vLLM**: High-throughput inference with optimized memory management
- **Prefix Caching**: Reuses computed KV cache for shared prompt prefixes, ideal for patient history prompts
- **Enable further parallelization via OpenAI server**: By launching as a separate server instance, this allows requests to run more parallel.

!!! tip "GDT Examples"
    For real-world examples of vLLM deployment with prefix caching, see the [GDT (Genie Digital Twin) repository](https://github.com/MendenLab/GDT).

### Experiment Tracking with Weights & Biases

Use **Weights & Biases (W&B)** for comprehensive experiment tracking at scale:

- Track hyperparameters, metrics, and model artifacts
- Compare runs and identify optimal configurations
- Collaborate with team members on experiment analysis

---
