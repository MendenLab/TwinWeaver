# TwinWeaver

<p align="center">
  <img src="https://raw.githubusercontent.com/MendenLab/TwinWeaver/refs/heads/main/docs/images/candidate_jpg.jpg#only-light" width="25%" alt="TwinWeaver Logo" title="Title">
  <img src="https://raw.githubusercontent.com/MendenLab/TwinWeaver/refs/heads/main/docs/images/candidate_dark_bg_jpg.jpg#only-dark" width="25%" alt="TwinWeaver Logo" title="Title">
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black.svg?logo=github)](https://github.com/MendenLab/TwinWeaver)

TwinWeaver is a longitudinal framework for LLM-based Patient Digital Twins. It serializes longitudinal patient histories into text, enabling unified event prediction as well as forecasting with large language models (LLMs). This framework transforms structured patient history—including demographics, labs, treatments, and genetics—into a single, human-readable text prompt, enabling LLMs to jointly forecast continuous biomarkers and predict discrete clinical events.

---

## Get Started

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Installation**

    ---

    Install TwinWeaver with pip in seconds

    [:octicons-arrow-right-24: Installation Guide](installation.md)

-   :material-book-open-variant:{ .lg .middle } **Tutorials**

    ---

    Step-by-step notebooks to learn TwinWeaver

    [:octicons-arrow-right-24: View Tutorials](tutorials.md)

-   :material-code-braces:{ .lg .middle } **Quick Start**

    ---

    Minimal code example for experienced users

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-file-document:{ .lg .middle } **Dataset Format**

    ---

    Understand the expected data structure

    [:octicons-arrow-right-24: Dataset Format](dataset-format.md)

</div>

---

## Why TwinWeaver?

TwinWeaver addresses the challenge of modeling sparse, multi-modal clinical time series by leveraging the generative capabilities of LLMs:

- **Text Serialization**: Transforms multi-modal inputs into structured textual representations
- **Unified Tasks**: Supports both time-series forecasting and landmark event prediction
- **Flexible Horizons**: Avoids overfitting to specific canonical time points
- **MEDS Integration**: Easily integrate existing MEDS datasets

[:octicons-arrow-right-24: Learn more about the framework](framework.md)

---

## Featured: Genie Digital Twin (GDT)

GDT is a pan-cancer model instantiated using TwinWeaver, trained on over 93,000 patients across 20 cancer types. It achieves a median MASE of **0.87** for forecasting and an average C-index of **0.703** for risk stratification.

[:octicons-arrow-right-24: GDT Repository](https://github.com/MendenLab/GDT)

---

## Citation

If you use TwinWeaver in your research, please cite our paper:
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
