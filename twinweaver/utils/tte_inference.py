"""
Time-to-event (TTE) inference helpers for probability estimation.

Provides functions to estimate the length-normalized probabilities of three
mutually exclusive TTE outcomes (censored, occurred, not occurred) by scoring
each completion against an OpenAI-compatible API (e.g. a local vLLM server).

The prompt construction is driven entirely by a :class:`~twinweaver.common.config.Config`
instance, so the same code works for any dataset / prompt template.

Typical usage
-------------
>>> import asyncio
>>> from transformers import AutoTokenizer
>>> from twinweaver.common.config import Config
>>> from twinweaver.utils.tte_inference import (
...     run_tte_probability_estimation,
...     compute_length_normalized_probabilities,
... )
>>> config = Config()
>>> tokenizer = AutoTokenizer.from_pretrained("my-model")
>>> data = [("patient_1", "instruction text ...")]
>>> raw = asyncio.run(run_tte_probability_estimation(
...     data, tokenizer, config, prediction_url="http://localhost:8000/v1/",
...     prediction_model="my-model"))
>>> df = compute_length_normalized_probabilities(raw)
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import numpy as np
import pandas as pd
import scipy.special
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from twinweaver.common.config import Config

# ---------------------------------------------------------------------------
# Label constants  (re-exported for downstream convenience)
# ---------------------------------------------------------------------------
LABEL_OCCURRED: str = "occurred"
LABEL_NOT_OCCURRED: str = "not_occurred"
LABEL_CENSORED: str = "censored"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_tte_completion_strings(config: Config) -> list[tuple[str, str]]:
    """Build the three (label, completion_suffix) pairs from *config*.

    The completion suffixes are the text fragments that follow
    ``config.target_prompt_start`` for each of the three TTE states.
    They are constructed from the same config attributes used during
    training-data generation, ensuring consistency.

    Returns
    -------
    list[tuple[str, str]]
        ``[(LABEL_CENSORED, "censored."), (LABEL_OCCURRED, "not censored and occurred."), ...]``
        (exact strings depend on the config).
    """
    # Censored case – the target string ends immediately after the
    # censoring marker (no occurrence info).
    censored_suffix = config.target_prompt_censor_true

    # Occurred (not censored) case
    occurred_suffix = config.target_prompt_censor_false + config.target_prompt_before_occur + config.target_prompt_occur

    # Not occurred (not censored) case
    not_occurred_suffix = (
        config.target_prompt_censor_false + config.target_prompt_before_occur + config.target_prompt_not_occur
    )

    return [
        (LABEL_CENSORED, censored_suffix),
        (LABEL_OCCURRED, occurred_suffix),
        (LABEL_NOT_OCCURRED, not_occurred_suffix),
    ]


def _extract_event_name_from_instruction(instruction: str, config: Config) -> str:
    """Extract the event name embedded in a TTE instruction string.

    The instruction is expected to follow the pattern produced by
    :class:`~twinweaver.instruction.converter_events.ConverterEvents` using
    ``config.forecasting_tte_prompt_mid`` and ``config.forecasting_tte_prompt_end``.

    For the default config the instruction contains a substring like::

        ... weeks from the last clinical visit and whether the event occurred or not: <event_name>.
        Please provide your prediction ...

    Parameters
    ----------
    instruction : str
        The full instruction text for a single patient / time-point.
    config : Config
        The configuration object (used to locate delimiter strings).

    Returns
    -------
    str
        The extracted event name (stripped of leading / trailing whitespace
        and trailing periods).

    Raises
    ------
    ValueError
        If the event name cannot be located in *instruction*.
    """
    # The event name sits between forecasting_tte_prompt_mid and
    # forecasting_tte_prompt_end.  Both are defined in Config and may be
    # customised by the user, so we split on the actual config values.
    mid = config.forecasting_tte_prompt_mid
    end = config.forecasting_tte_prompt_end

    if mid in instruction:
        after_mid = instruction.split(mid, 1)[1]
        event_name = after_mid.split(end, 1)[0] if end and end in after_mid else after_mid
    else:
        # Fallback: try to pull the name from the target_prompt_start
        # pattern which contains ``({event_name})``.
        m = re.search(r"\(([^)]+)\)\s*was\s", instruction)
        if m:
            event_name = m.group(1)
        else:
            raise ValueError(
                f"Could not extract event name from instruction. Looked for '{mid}' as delimiter but it was not found."
            )

    return event_name.strip().rstrip(".")


def build_scored_prompt(
    instruction: str,
    tokenizer: Any,
    config: Config,
    *,
    system_prompt: str | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Construct the full prompt prefix and the three scored completions.

    This is the **model-agnostic** equivalent of the prompt assembly that was
    previously hard-coded for Llama-3.1 chat tokens.  It uses the tokenizer's
    ``apply_chat_template`` when available, falling back to a simple
    concatenation otherwise.

    Parameters
    ----------
    instruction : str
        The user-facing instruction text (one patient / time-point).
    tokenizer
        A HuggingFace-compatible tokenizer (must support ``encode``; ideally
        also ``apply_chat_template``).
    config : Config
        The configuration object.
    system_prompt : str or None, optional
        An optional system prompt.  When *None* the chat template is built
        without a system message.

    Returns
    -------
    prompt_prefix : str
        The fully assembled prompt up to (and including) the
        ``config.target_prompt_start`` fragment.
    completions : list[tuple[str, str]]
        The three ``(label, suffix_text)`` pairs to score.

    Notes
    -----
    The slicing index (to separate prefix tokens from completion tokens)
    is **not** computed here because BPE tokenizers can merge tokens across
    the prefix/suffix boundary.  The caller should compute the slicing
    index from the full concatenated prompt instead.
    """

    # TODO: use config.task_prompt_each_task, task_prompt_events etc to start the correct build
    # raise NotImplementedError("This function is not yet implemented. It needs to be updated.")
    print("TODO: adjust to use config initial prompt building blocks.")

    event_name = _extract_event_name_from_instruction(instruction, config)
    target_start = config.target_prompt_start.format(event_name=event_name)
    completions = _build_tte_completion_strings(config)

    instruction_clean = instruction.strip()

    # --- assemble the full text that precedes each scored suffix ----------
    has_chat_template = hasattr(tokenizer, "apply_chat_template")

    if has_chat_template:
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instruction_clean})

        # Tokenize *without* generation prompt so we can append the
        # assistant preamble (target_prompt_start) ourselves.
        chat_prefix = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_prefix = chat_prefix + target_start
    else:
        # Simple fallback – just concatenate.
        parts = []
        if system_prompt is not None:
            parts.append(system_prompt)
        parts.append(instruction_clean)
        prompt_prefix = "\n\n".join(parts) + target_start

    return prompt_prefix, completions


# ---------------------------------------------------------------------------
# Async LLM call (single patient)
# ---------------------------------------------------------------------------


async def _call_llm_for_prediction_async(
    client: AsyncOpenAI,
    model_to_use: str,
    patientid: str,
    instruction: str,
    semaphore: asyncio.Semaphore,
    tokenizer: Any,
    config: Config,
    *,
    system_prompt: str | None = None,
) -> dict | None:
    """Score the three TTE completions for a single patient via the API.

    This is the generalised replacement for the reference GDT implementation.
    It uses :func:`build_scored_prompt` (which in turn uses *config*) so no
    prompt strings are hard-coded.

    Parameters
    ----------
    client : AsyncOpenAI
        An ``openai.AsyncOpenAI`` client pointing at the inference server.
    model_to_use : str
        Model identifier / path accepted by the server.
    patientid : str
        Unique identifier for the patient (passed through to the result).
    instruction : str
        The full instruction text for this patient / time-point.
    semaphore : asyncio.Semaphore
        Concurrency limiter.
    tokenizer
        HuggingFace-compatible tokenizer.
    config : Config
        Configuration object.
    system_prompt : str or None, optional
        Optional system prompt prepended to every request.

    Returns
    -------
    dict or None
        ``{"patientid": ..., "occurred": [...], "not_occurred": [...],
        "censored": [...]}`` where each value list contains per-token
        log-probabilities for the corresponding completion.  Returns
        *None* on unrecoverable API errors.
    """
    prompt_prefix, completions = build_scored_prompt(instruction, tokenizer, config, system_prompt=system_prompt)

    async with semaphore:
        try:
            return_dic: dict[str, Any] = {"patientid": patientid}

            for label, suffix in completions:
                combined_prompt = prompt_prefix + suffix

                # Compute the slicing index from the *full* combined prompt.
                # We cannot pre-compute this from prompt_prefix alone because
                # BPE tokenizers may merge tokens across the prefix/suffix
                # boundary, shifting the split point.
                #
                # Strategy: tokenize the full combined string and the suffix
                # independently, then find the split by looking at the token
                # count of the suffix portion.  Even this can be off by a
                # token, so after the API call we also verify using the
                # server's own tokenization and fall back to searching for
                # the suffix in the returned token list.
                suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
                n_suffix_tokens = len(suffix_tokens)

                response = await client.completions.create(
                    model=model_to_use,
                    prompt=combined_prompt,
                    max_tokens=0,  # scoring only – no generation
                    logprobs=1,
                    echo=True,
                )

                logprobs = response.choices[0].logprobs.token_logprobs
                tokens = response.choices[0].logprobs.tokens

                # Use the server's own tokenization to find the split.
                # The total token count minus the number of suffix tokens
                # (as counted by the local tokenizer) gives us a good
                # initial estimate.  We then verify with a sanity check.
                total_tokens = len(tokens)
                slicing_index = total_tokens - n_suffix_tokens

                # Verify: the joined suffix tokens should contain the suffix text.
                # If not, search nearby indices (±2) to handle boundary merging.
                output_text = "".join(tokens[slicing_index:]).strip()
                if suffix.strip() not in output_text:
                    # Try shifting the slicing index by ±1 or ±2
                    found = False
                    for offset in [-1, 1, -2, 2]:
                        candidate = slicing_index + offset
                        if 0 <= candidate <= total_tokens:
                            candidate_text = "".join(tokens[candidate:]).strip()
                            if suffix.strip() in candidate_text:
                                slicing_index = candidate
                                output_text = candidate_text
                                found = True
                                break
                    if not found:
                        raise ValueError(
                            f"Completion mismatch for label '{label}': "
                            f"expected '{suffix}' in output tokens, got "
                            f"'{output_text}'. Could not find a valid "
                            f"slicing index (tried offsets ±2)."
                        )

                # Slice to keep only the completion part
                completion_logprobs = logprobs[slicing_index:]

                return_dic[label] = completion_logprobs

            return return_dic

        except AuthenticationError as exc:
            print(f"Authentication error: {exc}")
        except RateLimitError as exc:
            print(f"Rate limit exceeded: {exc}")
        except APIConnectionError as exc:
            print(f"Network error: {exc}")
            raise
        except OpenAIError as exc:
            print(f"An OpenAI error occurred: {exc}")

    return None


# ---------------------------------------------------------------------------
# Async orchestrator (all patients)
# ---------------------------------------------------------------------------


async def _run_tte_probability_estimation_async(
    instructions_with_ids: list[tuple[str, str]],
    tokenizer: Any,
    config: Config,
    *,
    prediction_url: str = "http://0.0.0.0:8000/v1/",
    prediction_model: str = "default-model",
    max_concurrent_requests: int = 40,
    system_prompt: str | None = None,
    api_key: str = "EMPTY",
    timeout: float = 600.0,
) -> list[dict | None]:
    """Async implementation of :func:`run_tte_probability_estimation`."""
    client = AsyncOpenAI(
        base_url=prediction_url,
        api_key=api_key,
        timeout=timeout,
    )

    semaphore = asyncio.Semaphore(max_concurrent_requests)

    tasks = [
        _call_llm_for_prediction_async(
            client,
            prediction_model,
            patientid,
            instruction,
            semaphore,
            tokenizer,
            config,
            system_prompt=system_prompt,
        )
        for patientid, instruction in instructions_with_ids
    ]

    return await asyncio.gather(*tasks)


def run_tte_probability_estimation(
    instructions_with_ids: list[tuple[str, str]],
    tokenizer: Any,
    config: Config,
    *,
    prediction_url: str = "http://0.0.0.0:8000/v1/",
    prediction_model: str = "default-model",
    max_concurrent_requests: int = 40,
    system_prompt: str | None = None,
    api_key: str = "EMPTY",
    timeout: float = 600.0,
) -> list[dict | None]:
    """Score all patients against an OpenAI-compatible API and return raw log-probs.

    This is the main entry-point for running TTE probability estimation.
    It is synchronous (calls ``asyncio.run`` internally) so it can be used
    from plain scripts or notebooks.

    Parameters
    ----------
    instructions_with_ids : list[tuple[str, str]]
        Each element is ``(patientid, instruction_text)``.
    tokenizer
        HuggingFace-compatible tokenizer (used for token counting).
    config : Config
        TwinWeaver configuration object.
    prediction_url : str
        Base URL of the OpenAI-compatible inference server.
    prediction_model : str
        Model name / path served by the inference server.
    max_concurrent_requests : int
        Maximum number of concurrent API requests.
    system_prompt : str or None
        Optional system prompt.
    api_key : str
        API key (``"EMPTY"`` for local vLLM servers).
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    list[dict or None]
        One dict per patient with keys ``"patientid"``,
        ``"occurred"``, ``"not_occurred"``, ``"censored"``
        whose values are lists of per-token log-probabilities.
        ``None`` entries indicate API failures.
    """
    return asyncio.run(
        _run_tte_probability_estimation_async(
            instructions_with_ids,
            tokenizer,
            config,
            prediction_url=prediction_url,
            prediction_model=prediction_model,
            max_concurrent_requests=max_concurrent_requests,
            system_prompt=system_prompt,
            api_key=api_key,
            timeout=timeout,
        )
    )


def run_tte_probability_estimation_notebook(
    instructions_with_ids: list[tuple[str, str]],
    tokenizer: Any,
    config: Config,
    *,
    prediction_url: str = "http://0.0.0.0:8000/v1/",
    prediction_model: str = "default-model",
    max_concurrent_requests: int = 40,
    system_prompt: str | None = None,
    api_key: str = "EMPTY",
    timeout: float = 600.0,
) -> list[dict | None]:
    """Score all patients against an OpenAI-compatible API and return raw log-probs.

    This is the main entry-point for running TTE probability estimation, for use in Jupyter notebooks.
    It is asynchronous and returns a coroutine, so it can be awaited directly in notebooks.

    Parameters
    ----------
    instructions_with_ids : list[tuple[str, str]]
        Each element is ``(patientid, instruction_text)``.
    tokenizer
        HuggingFace-compatible tokenizer (used for token counting).
    config : Config
        TwinWeaver configuration object.
    prediction_url : str
        Base URL of the OpenAI-compatible inference server.
    prediction_model : str
        Model name / path served by the inference server.
    max_concurrent_requests : int
        Maximum number of concurrent API requests.
    system_prompt : str or None
        Optional system prompt.
    api_key : str
        API key (``"EMPTY"`` for local vLLM servers).
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    list[dict or None]
        One dict per patient with keys ``"patientid"``,
        ``"occurred"``, ``"not_occurred"``, ``"censored"``
        whose values are lists of per-token log-probabilities.
        ``None`` entries indicate API failures.
    """

    # No need for a separate function in the notebook – just call the async version directly.

    return _run_tte_probability_estimation_async(
        instructions_with_ids,
        tokenizer,
        config,
        prediction_url=prediction_url,
        prediction_model=prediction_model,
        max_concurrent_requests=max_concurrent_requests,
        system_prompt=system_prompt,
        api_key=api_key,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Post-processing: length-normalized probabilities
# ---------------------------------------------------------------------------


def compute_length_normalized_probabilities(
    raw_results: list[dict | None],
    *,
    drop_failures: bool = True,
) -> pd.DataFrame:
    """Convert raw per-token log-probs into length-normalized softmax probabilities.

    For each patient the function:

    1. Computes the **mean log-probability** (length-normalised) of each of the
       three completion strings.
    2. Applies a **softmax** over the three mean log-probs to obtain calibrated
       probabilities that sum to one.
    3. Derives a hard **prediction** (``censored``, ``occurred`` columns) by
       picking the class with the highest softmax score.

    Parameters
    ----------
    raw_results : list[dict or None]
        Output of :func:`run_tte_probability_estimation`.
    drop_failures : bool
        If *True* (default), silently drop ``None`` entries (API failures).
        If *False*, raise a ``ValueError`` when any entry is ``None``.

    Returns
    -------
    pd.DataFrame
        Columns: ``patientid``, ``censored`` (bool), ``occurred`` (bool),
        ``probability_occurrence`` (float), ``probability_no_occurrence`` (float),
        ``probability_censored`` (float), plus the intermediate
        ``avg_logprob_*`` and ``softmax_*`` columns.

    Raises
    ------
    ValueError
        If *drop_failures* is *False* and any result is ``None``.
    """
    if drop_failures:
        valid = [r for r in raw_results if r is not None]
    else:
        if any(r is None for r in raw_results):
            raise ValueError("Some results are None (API failures). Set drop_failures=True to silently ignore them.")
        valid = raw_results  # type: ignore[assignment]

    if not valid:
        raise ValueError("No valid results to process.")

    df = pd.DataFrame(valid)
    df = df[["patientid", LABEL_OCCURRED, LABEL_NOT_OCCURRED, LABEL_CENSORED]]

    # --- 1. Length-normalised mean log-probability ---------------------------
    for label in (LABEL_OCCURRED, LABEL_NOT_OCCURRED, LABEL_CENSORED):
        df[f"avg_logprob_{label}"] = df[label].apply(np.mean)

    # --- 2. Softmax across the three states ----------------------------------
    logprob_cols = [
        f"avg_logprob_{LABEL_OCCURRED}",
        f"avg_logprob_{LABEL_NOT_OCCURRED}",
        f"avg_logprob_{LABEL_CENSORED}",
    ]
    softmax_cols = [
        f"softmax_{LABEL_OCCURRED}",
        f"softmax_{LABEL_NOT_OCCURRED}",
        f"softmax_{LABEL_CENSORED}",
    ]

    df[softmax_cols] = df[logprob_cols].apply(
        lambda row: pd.Series(scipy.special.softmax(row.values)),
        axis=1,
    )

    # --- 3. Hard prediction ---------------------------------------------------
    selection = [LABEL_OCCURRED, LABEL_NOT_OCCURRED, LABEL_CENSORED]

    def _hard_prediction(row):
        probs = [
            row[f"softmax_{LABEL_OCCURRED}"],
            row[f"softmax_{LABEL_NOT_OCCURRED}"],
            row[f"softmax_{LABEL_CENSORED}"],
        ]
        best = selection[int(np.argmax(probs))]
        return pd.Series(
            {
                "censored": best == LABEL_CENSORED,
                "occurred": best == LABEL_OCCURRED,
            }
        )

    df[["censored", "occurred"]] = df.apply(_hard_prediction, axis=1)

    # --- 4. Friendly probability columns --------------------------------------
    df["probability_occurrence"] = df[f"softmax_{LABEL_OCCURRED}"]
    df["probability_no_occurrence"] = df[f"softmax_{LABEL_NOT_OCCURRED}"]
    df["probability_censored"] = df[f"softmax_{LABEL_CENSORED}"]

    return df
