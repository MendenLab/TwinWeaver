# Time-to-Event (TTE) Probability Inference

TwinWeaver supports **landmark event prediction** as one of its core clinical tasks. During training, the model learns to classify whether a clinical event (e.g., death, disease progression, therapy switching) was *censored*, *occurred*, or *did not occur* within a given observation window. At inference time, we potentially want **probabilities** for each outcome.

This page explains the TTE probability inference mechanism: how it works, why length-normalised log-probabilities are used, and where future research should focus to improve calibration.

!!! info "Reference"
    The TTE probability inference approach is described in the TwinWeaver preprint:

    > Makarov, N. et al. (2026). *TwinWeaver: An LLM-Based Foundation Model Framework for Pan-Cancer Digital Twins.* [arXiv:2601.20906](https://arxiv.org/abs/2601.20906)

---

## Overview

Instead of generating free-text answers, the TTE inference pipeline **scores three mutually exclusive completions** for each patient and derives probabilities from the model's own confidence in each completion.

| Outcome | Meaning |
|---|---|
| **Censored** | The patient's observation window ended before the event could be observed |
| **Occurred** | The event happened within the observation window |
| **Not occurred** | The event did not happen within the observation window |

This three-class formulation matches TwinWeaver's training-time target format, ensuring that inference is consistent with what the model learned during fine-tuning.

---

## Pipeline Architecture

```
Patient data ──► DataSplitter (events) ──► ConverterInstruction
                                                │
                                  instruction text per patient
                                                │
                     ┌──────────────────────────┘
                     ▼
              vLLM server (OpenAI-compatible API)
                     │
          log-probs for 3 completions
                     │
                     ▼
     compute_length_normalized_probabilities()
                     │
            calibrated probabilities
            + hard predictions (DataFrame)
```

### Step-by-step

1. **Prompt construction** — For each patient, `build_scored_prompt()` assembles the full prompt prefix (chat template + patient history + task instruction) and three completion suffixes (one per outcome). The suffixes are constructed from the same `Config` attributes used during training to ensure exact token-level consistency.

2. **Log-probability scoring** — Each `(prompt_prefix + completion_suffix)` is sent to the vLLM server with `max_tokens=0` and `echo=True`. This makes the server return per-token log-probabilities for the entire prompt *without generating any new tokens*. Only the log-probabilities of the completion tokens are retained.

3. **Length normalisation** — The mean log-probability across completion tokens is computed for each outcome. This prevents longer completions from being unfairly penalised.

4. **Softmax** — The three mean log-probabilities are passed through a softmax to yield probabilities that sum to 1.

5. **Hard prediction** — The outcome with the highest probability is selected as the hard prediction.

---

## Mathematical Formulation

Given an instruction prompt $\mathbf{x}$ and a completion string $\mathbf{c}_k$ for outcome $k \in \{\text{censored}, \text{occurred}, \text{not occurred}\}$, the model assigns a per-token log-probability to each token $c_{k,t}$ in the completion:

$$
\log p(c_{k,t} \mid \mathbf{x}, c_{k,<t})
$$

The **length-normalised log-probability** for outcome $k$ is:

$$
\bar{\ell}_k = \frac{1}{T_k} \sum_{t=1}^{T_k} \log p(c_{k,t} \mid \mathbf{x}, c_{k,<t})
$$

where $T_k$ is the number of tokens in completion $\mathbf{c}_k$.

The **softmax probabilities** are then:

$$
P(k \mid \mathbf{x}) = \frac{\exp(\bar{\ell}_k)}{\sum_{j} \exp(\bar{\ell}_j)}
$$

These three probabilities sum to 1 and represent the model's relative confidence in each outcome.

---

## Key API Functions

TwinWeaver provides three main functions for TTE inference in `twinweaver.utils.tte_inference`:

| Function | Purpose |
|---|---|
| `build_scored_prompt()` | Constructs the prompt prefix and three completion suffixes from a patient instruction and config. Useful for debugging and inspecting what the model sees. |
| `run_tte_probability_estimation()` | Synchronous entry-point that scores all patients against an OpenAI-compatible API. Returns raw per-token log-probabilities. |
| `run_tte_probability_estimation_notebook()` | Asynchronous variant for use inside Jupyter notebooks (returns a coroutine that can be `await`ed). |
| `compute_length_normalized_probabilities()` | Post-processes raw log-probs into length-normalised softmax probabilities and hard predictions. Returns a `pandas.DataFrame`. |

For full API documentation, see the [TTE Inference API Reference](reference/utils/tte_inference.md).

---

## Usage Requirements

- A **fine-tuned model** — An off-the-shelf instruction model will produce random probabilities because it was not trained on TwinWeaver's target format. See [`03_end_to_end_llm_finetuning.ipynb`](examples/03_end_to_end_llm_finetuning.ipynb) for training.
- A **vLLM server** (or any OpenAI-compatible API) that supports `echo=True` and `logprobs` in the completions endpoint.
- A GPU with enough memory to serve the model (≥ 16 GB for a 4-bit quantised 8B model).

```bash
pip install twinweaver[fine-tuning-example] vllm openai
```

!!! tip "Tutorial"
    For a complete worked example, see the [TTE Probability Inference notebook](examples/advanced/tte_inference/tte_probability_inference.ipynb).

---

## Prompt Consistency

A critical design principle is that the completion suffixes scored at inference time must **exactly match** the target strings produced during training. TwinWeaver achieves this by constructing both from the same `Config` attributes:

- `config.target_prompt_censor_true` — The censored completion text
- `config.target_prompt_censor_false` — The "not censored" prefix
- `config.target_prompt_before_occur` — Text bridging censoring status and occurrence
- `config.target_prompt_occur` / `config.target_prompt_not_occur` — The occurrence markers

If you customised any of these during training, the same `Config` must be used at inference time.

---

## Limitations and Future Research

While the TTE probability inference pipeline provides a practical mechanism for extracting event probabilities from a fine-tuned LLM, there are important limitations to be aware of.

### Calibration

As discussed in the [TwinWeaver preprint](https://arxiv.org/abs/2601.20906), the softmax probabilities derived from length-normalised log-probs are **not well calibrated** in the statistical sense. That is, a predicted probability of 0.7 does not necessarily mean that the event occurs 70% of the time across a population of patients with similar predictions. The probabilities are useful for **ranking** patients by risk and for deriving hard predictions, but should not be interpreted as true event probabilities without further calibration.

### Directions for Future Research

Several avenues could improve the quality and calibration of TTE probability estimates:

1. **Post-hoc calibration** — Applying calibration methods such as Platt scaling, isotonic regression, or temperature scaling on a held-out calibration set could improve the reliability of the predicted probabilities.

2. **Alternative scoring functions** — Instead of length-normalised mean log-probabilities, one could explore:
    - Geometric mean of token probabilities
    - Weighted scoring that accounts for token position or importance
    - Mutual information–based scoring

3. **Verbalised confidence** — Rather than extracting probabilities from log-probs, the model could be trained to directly output a numerical probability or confidence score as part of its generation. This approach has shown promise in other domains and could be more naturally calibrated.

4. **Ensemble methods** — Combining predictions from multiple fine-tuned models or using Monte Carlo dropout at inference time could provide better uncertainty estimates.

5. **Continuous survival modelling** — Extending the framework to predict full survival curves rather than discrete event/no-event classifications at fixed time horizons. This could be achieved by evaluating multiple time horizons jointly (as demonstrated in the [tutorial notebook](examples/advanced/tte_inference/tte_probability_inference.ipynb)) or by training the model to output parametric survival functions.

6. **Training objective alignment** — Modifying the training loss to explicitly optimise for calibrated probability estimation rather than standard next-token prediction, for example through proper scoring rules.

!!! note "Contributing"
    If you explore any of these directions, we welcome contributions and discussions via [GitHub Issues](https://github.com/MendenLab/TwinWeaver/issues).

---

## Citation

If you use TwinWeaver's TTE probability inference in your research, please cite:

```bibtex
@misc{makarov2026twinweaver,
      title={TwinWeaver: An LLM-Based Foundation Model Framework for Pan-Cancer Digital Twins},
      author={Nikita Makarov and Maria Bordukova and Lena Voith von Voithenberg and Estrella Pivel-Villanueva and Sabrina Mielke and Jonathan Wickes and Hanchen Wang and Mingyu Derek Ma and Keunwoo Choi and Kyunghyun Cho and Stephen Ra and Raul Rodriguez-Esteban and Fabian Schmich and Michael Menden},
      year={2026},
      eprint={2601.20906},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.20906},
}
```
