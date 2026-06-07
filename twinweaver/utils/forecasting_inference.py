"""
Forecasting inference helpers for vLLM-based text generation.

Provides functions to generate forecasting predictions (future lab values,
vitals, etc.) by sending instruction prompts to an OpenAI-compatible API
(e.g. a local vLLM server) and parsing the model's text output back into
structured DataFrames via :class:`~twinweaver.instruction.converter_instruction.ConverterInstruction`.

The prompt construction is driven by
:class:`~twinweaver.instruction.converter_instruction.ConverterInstruction`
(specifically its ``forward_conversion_inference`` method), so the same
code works for any dataset / prompt template.

Typical usage
-------------
>>> import asyncio
>>> from twinweaver.common.config import Config
>>> from twinweaver.utils.forecasting_inference import (
...     run_forecasting_inference,
...     parse_forecasting_results,
... )
>>> config = Config()
>>> # prompts_with_meta is a list of dicts with keys:
>>> #   "patientid", "instruction", "split_date"
>>> results = asyncio.run(run_forecasting_inference(
...     prompts_with_meta,
...     prediction_url="http://localhost:8000/v1/",
...     prediction_model="my-model",
... ))
>>> df = parse_forecasting_results(results, converter, dm)
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)


# ---------------------------------------------------------------------------
# Type alias for a single prompt payload
# ---------------------------------------------------------------------------
# Each element is a dict with at least:
#   "patientid"   : str   – unique patient identifier
#   "instruction" : str   – full instruction text (from converter)
#   "split_date"  : datetime – the reference date used for reverse conversion
# Additional keys are preserved and passed through to the results.
PromptPayload = dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_chat_messages(
    instruction: str,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build the list of chat messages for the API call.

    Parameters
    ----------
    instruction : str
        The user-facing instruction text.
    system_prompt : str or None
        Optional system prompt.

    Returns
    -------
    list[dict[str, str]]
    """
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    return messages


# ---------------------------------------------------------------------------
# Async LLM call (single patient)
# ---------------------------------------------------------------------------


async def _call_llm_for_generation_async(
    client: AsyncOpenAI,
    model_to_use: str,
    payload: PromptPayload,
    semaphore: asyncio.Semaphore,
    *,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    n_samples: int = 1,
) -> dict | None:
    """Generate forecasting text for a single patient via the API.

    Parameters
    ----------
    client : AsyncOpenAI
        An ``openai.AsyncOpenAI`` client pointing at the inference server.
    model_to_use : str
        Model identifier / path accepted by the server.
    payload : PromptPayload
        Dict with ``"patientid"``, ``"instruction"``, ``"split_date"`` and
        any other metadata the caller wants to pass through.
    semaphore : asyncio.Semaphore
        Concurrency limiter.
    system_prompt : str or None, optional
        Optional system prompt prepended to every request.
    max_new_tokens : int
        Maximum number of tokens to generate per completion.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus-sampling probability mass.
    n_samples : int
        Number of independent completions to generate per prompt.  When > 1
        the caller can later aggregate multiple trajectories.

    Returns
    -------
    dict or None
        A copy of the input *payload* augmented with:

        * ``"generated_texts"`` – list[str] of generated completion texts
          (length = *n_samples*).

        Returns *None* on unrecoverable API errors.
    """
    messages = _build_chat_messages(payload["instruction"], system_prompt=system_prompt)

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n_samples,
            )

            generated_texts = [choice.message.content for choice in response.choices]

            result = {k: v for k, v in payload.items() if k != "instruction"}
            result["generated_texts"] = generated_texts
            return result

        except AuthenticationError as exc:
            print(f"Authentication error for patient {payload.get('patientid')}: {exc}")
        except RateLimitError as exc:
            print(f"Rate limit exceeded for patient {payload.get('patientid')}: {exc}")
        except APIConnectionError as exc:
            print(f"Network error for patient {payload.get('patientid')}: {exc}")
            raise
        except OpenAIError as exc:
            print(f"An OpenAI error occurred for patient {payload.get('patientid')}: {exc}")

    return None


# ---------------------------------------------------------------------------
# Async orchestrator (all patients)
# ---------------------------------------------------------------------------


async def _run_forecasting_inference_async(
    prompts_with_meta: list[PromptPayload],
    *,
    prediction_url: str = "http://0.0.0.0:8000/v1/",
    prediction_model: str = "default-model",
    max_concurrent_requests: int = 40,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    n_samples: int = 1,
    api_key: str = "EMPTY",
    timeout: float = 600.0,
) -> list[dict | None]:
    """Async implementation of :func:`run_forecasting_inference`."""
    client = AsyncOpenAI(
        base_url=prediction_url,
        api_key=api_key,
        timeout=timeout,
    )

    semaphore = asyncio.Semaphore(max_concurrent_requests)

    tasks = [
        _call_llm_for_generation_async(
            client,
            prediction_model,
            payload,
            semaphore,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n_samples=n_samples,
        )
        for payload in prompts_with_meta
    ]

    return await asyncio.gather(*tasks)


def run_forecasting_inference(
    prompts_with_meta: list[PromptPayload],
    *,
    prediction_url: str = "http://0.0.0.0:8000/v1/",
    prediction_model: str = "default-model",
    max_concurrent_requests: int = 40,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    n_samples: int = 1,
    api_key: str = "EMPTY",
    timeout: float = 600.0,
) -> list[dict | None]:
    """Generate forecasting predictions for all patients via an OpenAI-compatible API.

    This is the main synchronous entry-point.  It calls ``asyncio.run``
    internally so it can be used from plain scripts.

    Parameters
    ----------
    prompts_with_meta : list[PromptPayload]
        Each element is a dict with **at least** the following keys:

        * ``"patientid"``   – unique identifier (str)
        * ``"instruction"`` – full instruction text produced by
          ``ConverterInstruction.forward_conversion_inference``
        * ``"split_date"``  – the reference date (datetime) used when
          building the split; needed later for ``reverse_conversion``

        Any extra keys are passed through unchanged to the results.

    prediction_url : str
        Base URL of the OpenAI-compatible inference server.
    prediction_model : str
        Model name / path served by the inference server.
    max_concurrent_requests : int
        Maximum number of concurrent API requests.
    system_prompt : str or None
        Optional system prompt.
    max_new_tokens : int
        Maximum number of tokens to generate per completion.
    temperature : float
        Sampling temperature (0 = greedy).
    top_p : float
        Nucleus-sampling probability mass.
    n_samples : int
        Number of independent completions per prompt.  Useful for
        trajectory aggregation (see
        :meth:`ConverterInstruction.aggregate_multiple_responses`).
    api_key : str
        API key (``"EMPTY"`` for local vLLM servers).
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    list[dict or None]
        One dict per patient.  Each dict contains all keys from the input
        payload (except ``"instruction"``) plus ``"generated_texts"`` – a
        list of *n_samples* generated completion strings.
        ``None`` entries indicate API failures.
    """
    return asyncio.run(
        _run_forecasting_inference_async(
            prompts_with_meta,
            prediction_url=prediction_url,
            prediction_model=prediction_model,
            max_concurrent_requests=max_concurrent_requests,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n_samples=n_samples,
            api_key=api_key,
            timeout=timeout,
        )
    )


def run_forecasting_inference_notebook(
    prompts_with_meta: list[PromptPayload],
    *,
    prediction_url: str = "http://0.0.0.0:8000/v1/",
    prediction_model: str = "default-model",
    max_concurrent_requests: int = 40,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    n_samples: int = 1,
    api_key: str = "EMPTY",
    timeout: float = 600.0,
) -> list[dict | None]:
    """Generate forecasting predictions – async variant for Jupyter notebooks.

    Identical to :func:`run_forecasting_inference` but returns a *coroutine*
    that can be ``await``-ed directly in a notebook cell (which already has
    a running event loop).

    Returns
    -------
    Coroutine[..., list[dict or None]]
    """
    return _run_forecasting_inference_async(
        prompts_with_meta,
        prediction_url=prediction_url,
        prediction_model=prediction_model,
        max_concurrent_requests=max_concurrent_requests,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        n_samples=n_samples,
        api_key=api_key,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Post-processing: parse & reverse-convert generated text
# ---------------------------------------------------------------------------


def parse_forecasting_results(
    raw_results: list[dict | None],
    converter: Any,
    data_manager: Any,
    *,
    drop_failures: bool = False,
    aggregate_samples: bool = True,
) -> pd.DataFrame:
    """Parse raw generated texts into structured DataFrames via reverse conversion.

    For each patient the function:

    1. Calls ``converter.reverse_conversion`` on every generated text to obtain
       structured forecasting DataFrames.
    2. When *n_samples > 1* and ``aggregate_samples`` is ``True``, aggregates the
       multiple trajectories using ``converter.aggregate_multiple_responses``.
    3. Returns a single long-format DataFrame with all patients' predictions.

    Parameters
    ----------
    raw_results : list[dict or None]
        Output of :func:`run_forecasting_inference`.
    converter : ConverterInstruction
        The same converter instance used to generate the instruction prompts.
        Must expose ``reverse_conversion`` and ``aggregate_multiple_responses``.
    data_manager : DataManager
        The data manager instance (passed to ``reverse_conversion``).
    drop_failures : bool
        If *True*, silently drop ``None`` entries (API failures).
        If *False*, raise a ``ValueError`` when any entry is ``None``.
    aggregate_samples : bool
        If *True* (default) and multiple samples were generated per patient,
        aggregate them via ``converter.aggregate_multiple_responses``.
        If *False*, each sample is returned as a separate row block with
        a ``"sample_idx"`` column.

    Returns
    -------
    pd.DataFrame
        A long-format DataFrame with columns from the reverse-converted
        forecasting data plus ``"patientid"`` and optionally ``"sample_idx"``.

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

    all_rows: list[pd.DataFrame] = []

    for result in valid:
        patientid = result["patientid"]
        split_date = result["split_date"]
        generated_texts = result["generated_texts"]

        # Reverse-convert each sample
        sample_dfs: list[pd.DataFrame] = []
        for sample_idx, text in enumerate(generated_texts):
            try:
                parsed_tasks = converter.reverse_conversion(
                    text,
                    data_manager,
                    split_date,
                    patientid=patientid,
                    inference_override=True,
                )
            except Exception as exc:
                print(f"Warning: reverse_conversion failed for patient {patientid} sample {sample_idx}: {exc}")
                continue

            # Collect forecasting task results
            for task_result in parsed_tasks:
                task_type = task_result.get("task_type", "")
                result_data = task_result.get("result")

                if isinstance(result_data, pd.DataFrame) and not result_data.empty:
                    df_task = result_data.copy()
                    df_task["patientid"] = patientid
                    df_task["sample_idx"] = sample_idx
                    df_task["task_type"] = task_type
                    sample_dfs.append(df_task)

        if not sample_dfs:
            continue

        if aggregate_samples and len(generated_texts) > 1 and len(sample_dfs) > 1:
            # Use the converter's built-in aggregation (groups by task type)
            # Separate by task type, aggregate each, then combine
            combined = pd.concat(sample_dfs, ignore_index=True)
            task_types = combined["task_type"].unique()
            agg_parts = []
            for tt in task_types:
                task_subset = combined[combined["task_type"] == tt]
                # Group into per-sample DataFrames for the aggregator
                per_sample = [
                    task_subset[task_subset["sample_idx"] == si].drop(
                        columns=["sample_idx", "task_type"], errors="ignore"
                    )
                    for si in task_subset["sample_idx"].unique()
                ]
                try:
                    # ``per_sample`` is a list[pd.DataFrame] (one reverse-converted
                    # forecasting frame per sample), which matches the signature of the
                    # forecasting sub-converter — not ``ConverterInstruction`` (which expects
                    # list[list[dict]]). Route to the forecasting sub-converter when present,
                    # and fall back to the converter itself for a bare ``ConverterForecasting``.
                    agg_converter = getattr(converter, "converter_forecasting", converter)
                    agg_df, _meta = agg_converter.aggregate_multiple_responses(per_sample)
                    agg_df["task_type"] = tt
                    agg_df["patientid"] = patientid
                    agg_parts.append(agg_df)
                except Exception as exc:
                    print(f"Warning: aggregation failed for patient {patientid}: {exc}")
                    # Fallback: just keep first sample
                    fallback = per_sample[0].copy()
                    fallback["task_type"] = tt
                    fallback["patientid"] = patientid
                    agg_parts.append(fallback)

            if agg_parts:
                all_rows.append(pd.concat(agg_parts, ignore_index=True))
        else:
            all_rows.extend(sample_dfs)

    if not all_rows:
        return pd.DataFrame()

    df_out = pd.concat(all_rows, ignore_index=True)
    return df_out
