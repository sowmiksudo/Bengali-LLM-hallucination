Nice project. I read through the notebook cell-by-cell (papermill logs it timing for every cell) and cross-referenced it against the category breakdown in your sheet. Here's what's actually happening, and where the real waste is.

## Where the 9 hours goes

This particular run took 6.32h total (22,755s of 32,400s budget). Breakdown by section:

| Section | Time | Share |
|---|---|---|
| Load judge model | 327s | 1.4% |
| **§8 Evaluate on dev** | **3,410s** | **15.0%** |
| **§9 Predict test → N3.csv** | **19,013s** | **83.6%** |
| Everything else (setup, grounding, few-shot, fallback def) | ~5s | ~0% |

So essentially all your time is spent in two LLM-judging passes. Everything else is noise. That's where to focus.

## 1. You're running dev and test at different batch sizes — free fix

`FAST_BATCH` is set to `2` when `fast_judge(dev)` runs in §8, then redefined to `8` in §9 right before `fast_judge(test)` runs. Dev is paying a 4x batching penalty for no reason — just set `FAST_BATCH = 8` (or higher, see below) once, before §8, and delete the redefinition in §9. This alone should cut a meaningful chunk off your 3,410s dev stage.

## 2. Batch size is almost certainly still too conservative

A 4-bit-quantized ≤14B model's *weights* take ~7-9GB. Batch=8 with only 64 generated tokens leaves a lot of GPU headroom unused on a T4 (16GB) or T4x2. Push `FAST_BATCH` up in steps (16 → 24 → 32) and watch `torch.cuda.mem_get_info()` — batched decoding throughput scales close to linearly until you hit a memory or compute wall, so this is usually the single biggest lever available without touching architecture.

## 3. Check whether you're accidentally splitting the model across both GPUs

Your model-loading cell uses `device_map="auto"`. If Kaggle gave you 2 GPUs and the model comfortably fits on *one* (a 4-bit ≤14B model will), `device_map="auto"` may still shard it across both — which forces sequential cross-GPU handoffs on every layer during autoregressive decoding and can be dramatically slower per-token than fitting on one GPU. Check the printed `GPU: ... | #GPU: ... | per-GPU: ... GB` banner from cell 10. If `#GPU: 2` and the model fits in one GPU's memory:
- Pin it explicitly: `device_map={"": 0}`
- Then, better, load a **second replica on `cuda:1`** and split the test rows in half across two processes/threads — true data parallelism instead of one slow pipeline-split model. This can nearly double throughput on a dual-T4 box.

## 4. Batch-level early stopping has a straggler problem

`StopWhenAllVerdicts` only stops a batch once **every** sequence in it has emitted a verdict. One slow row in a batch of 8 forces all 8 to keep generating up to `FAST_MAX_NEW=64`. This is a real architectural ceiling — vanilla HF `generate()` can't evict a finished sequence from a batch and backfill a new one. Two options:
- **Cheap:** check your dev run's distribution of "tokens until verdict" and set `FAST_MAX_NEW` just above the ~90th percentile instead of a round 64 — caps the worst-case waste.
- **Bigger lever:** if internet access is allowed during your *scored* submission run (double-check — many competitions disable it for the final run, and your notebook already has offline-fallback logic suggesting you've hit this before), swapping the generate loop for **vLLM** gives continuous batching where each sequence exits the moment it's done, independent of its batch-mates. This is usually the highest-ceiling fix but is more engineering risk if you have to package it as an offline wheel dataset.

## 5. You already built a routing shortcut — it's just never called

`fast_route()` auto-labels rows with a very strong grounding mismatch (`ground <= -0.55`) without touching the LLM. It's defined in cell 12 but never invoked in either §8 or §9 — dead code. Wiring it in before `fast_judge` skips the LLM entirely for whatever fraction of rows already have a confident signal:

```python
auto_labels, auto_conf, hard_mask = fast_route(test)
tl, tc = auto_labels.copy(), auto_conf.copy()
sub = test[hard_mask].reset_index(drop=True)
sub_labels, sub_conf = fast_judge(sub)
tl[hard_mask], tc[hard_mask] = sub_labels, sub_conf
```

## 6. `selective_self_consistency` re-runs failures one row at a time

`SC_BATCH = 1` with `SC_SAMPLES = 3` and up to 192 tokens — if more than a handful of rows fail to parse a verdict, this is a sequential tail-latency trap. Batch it (`SC_BATCH=8` or so, memory permitting). Also worth checking: if a non-trivial % of dev rows come back with `labels < 0` (unparsed), that's a correctness problem with your verdict format/token budget, not just a speed one.

## 7. Your grounding signal skips ~46% of rows entirely — matching your EDA

`grounding_signal()` returns `0.0` immediately if `has_context` is false. Your sheet shows **1,155 of ~2,516 rows (46%) are zero-shot** — so nearly half the dataset gets zero benefit from the one heuristic that could bypass the LLM. Two category-specific fixes that pull double duty (speed *and* accuracy):

- **Mathematical (354 zero-shot rows):** add an independent arithmetic parser on the *prompt* text — extract the expression, compute it in Python, compare to the number in the response. A symbolic check is more reliable than an LLM's CoT for arithmetic anyway, and it's nearly free.
- **Idiom Meaning (150 rows, 100% zero-shot):** you already load `IDIOM_DICT` with the correct figurative meaning — right now it's only injected as a prompt hint. A direct similarity check between the response and the dictionary's figurative meaning (reusing the TF-IDF machinery you already have) could auto-resolve a chunk of these 150 rows without the LLM at all.

## 8. Add a time-budget circuit breaker

Right now if you fall behind pace, Kaggle kills the kernel at 9h and you get **nothing** — no partial submission. Cheap insurance:

```python
import time
START, BUDGET_S = time.time(), 8.5 * 3600  # 30-min safety margin
def time_left(): return BUDGET_S - (time.time() - START)
```

Check `time_left()` periodically in the §9 loop; if you're behind pace, route remaining rows to the TF-IDF+grounding fallback instead of the LLM. Also write `submission.to_csv` incrementally (every N batches) so you always have a valid partial file on disk.

---

One thing I can't tell from the stripped notebook: how many rows were in *this run's* `test.csv`. If it was already ~2,500, your 6.32h total leaves a real but not huge (~30%) safety margin. If it was smaller, scaling to 2,500 could blow the 9h limit outright — worth checking the `test: {len(test)} rows` print from your Kaggle run log before you rely on the current timing. Combining fixes #1, #2, and #5-7 alone (all low-risk, no new dependencies) should meaningfully cut the §9 judge stage regardless.

Want me to go through the notebook and actually apply these changes (batch size, `fast_route` wiring, the math/idiom shortcuts, checkpointing) so you have an updated `.ipynb` to run?