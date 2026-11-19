# Reproducing our results

Everything needed to rerun Team A4's final grant screener is in this folder. Open **`reproduce.ipynb`** and run it top to bottom; it works on Google Colab, SCU's WAVE HPC, or any Linux machine with an NVIDIA GPU.

## What gets reproduced

The final system rates each grant for *Ardook Data* (`docs/ardook-data-profile.md`) with **Qwen3-32B** ([Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)):

1. **Eligibility checklist** (`src/run_pipeline.py`): the model answers fixed questions about the listing and the rules in `src/ardook_grants/rules.py` decide whether Ardook Data can apply.
2. **Retrieval-augmented Fit judge** (`src/run_judge.py --k 16`): the model rates High / Moderate / Low, shown the 16 most similar already-rated grants (found with `sentence-transformers/all-MiniLM-L6-v2`).

Final Fit = Low if the checklist rules the grant out, otherwise the judge's Fit. *Relevant* = Fit is not Low.

Expected results (`bundle/expected_results.csv`; the notebook checks these):

| Dataset | n | System | 3-tier accuracy | 3-tier macro F1 | Relevant accuracy | Relevant F1 |
|---|---|---|---|---|---|---|
| Synthetic, held-out wording (`synthetic_alt_test`) | 200 | final system | 0.755 | 0.725 | 0.865 | 0.866 |
| | | judge alone | 0.765 | 0.738 | 0.885 | 0.890 |
| | | checklist alone | 0.515 | 0.518 | 0.695 | 0.772 |
| Synthetic test split (`synthetic_test`) | 250 | final system | 0.940 | 0.934 | 0.952 | 0.957 |
| Real grants rated by the sponsor (`real`) | 28 | final system | 0.429 | 0.325 | 0.750 | 0.857 |

The held-out wording set is the fairest estimate: the judge's example pool shares fact phrasings with the regular synthetic test split but not with it. The real set is small and its labels are almost all Moderate/High.

## Two modes

- **Replay** (`MODE = "replay"`, default; CPU only, about 2 minutes): re-scores the answers Qwen3-32B gave in our runs (`bundle/cache/`) through the current code, and must match the table above exactly. Checks the code, rules, retrieval and scoring.
- **Live** (`MODE = "live"`; GPU): serves the model yourself with vLLM 0.18.0 (the version our server ran), asks every question again, then compares your scores and per-grant answers with ours on the same grants. `SUBSET = "quick"` uses the first 40 grants of each set; `"full"` uses all 478 (about 2,900 model calls).

The model is picked from GPU memory: **≥ 70 GB** serves `Qwen/Qwen3-32B` in bf16 (exactly our setup); **38–70 GB** serves `Qwen/Qwen3-32B-AWQ` (same model, 4-bit, so a few grants may differ); **14–38 GB** serves `Qwen/Qwen3-4B`, which runs end to end but scores lower. Set `MODEL` to override.

## Google Colab

1. Upload `reproduce.ipynb` to Colab (*File → Upload notebook*), or open it from GitHub (*File → Open notebook → GitHub*, tick *Include private repos*).
2. For live mode pick *Runtime → Change runtime type → A100 or H100 GPU* (Colab Pro or pay-as-you-go) and set `MODE = "live"` in the first cell. Replay mode works on any runtime.
3. *Runtime → Run all*. The repository is private, so the second cell asks for a GitHub token with read access to it (or leave it blank and upload a `.zip` of the repository).

Colab deletes everything when the runtime ends, so each new session downloads the weights again (66 GB for bf16, 20 GB for AWQ: roughly 5–15 minutes).

## WAVE HPC

From the repository root on the **login node** (it has internet):

```bash
git clone https://github.com/SCU-CSEN166/ArdookGrantProject.git && cd ArdookGrantProject
export HF_HOME=/path/with/100GB/free/hf-cache        # your scratch or project space, not $HOME
bash reproduce/wave/setup_env.sh                      # one time: .repro-env/ and .vllm-env/ (about 10 GB)
REPRO_PREFETCH_MODEL=Qwen/Qwen3-32B bash reproduce/wave/setup_env.sh   # optional: download weights now, in case GPU nodes have no internet
sinfo -o "%P %G %l"                                   # check partition and GPU names, then edit the #SBATCH lines
```

Then either

```bash
sbatch reproduce/wave/run.sbatch                                  # batch: live mode, quick subset
sbatch --export=ALL,REPRO_SUBSET=full reproduce/wave/run.sbatch   # batch: all 478 grants
```

which writes the executed notebook, with every table, to `outputs/reproduce/reproduce-<jobid>.ipynb`; or

```bash
sbatch reproduce/wave/jupyter.sbatch     # interactive: Jupyter Lab on a GPU node
cat outputs/reproduce/jupyter-<jobid>.log  # the SSH tunnel command and the login URL
```

If the vLLM wheel won't run on the cluster (it needs glibc 2.31 or newer), add `REPRO_LAUNCHER=apptainer` to use the `vllm/vllm-openai:v0.18.0` container (the same image our Akash server used; set `APPTAINER_CACHEDIR` to a big disk, it is about 20 GB).

## Any other machine

Replay mode runs anywhere with Python 3.10+: `pip install -r requirements.txt jupyter`, then open the notebook. Live mode needs Linux and an NVIDIA GPU, or set `BASE_URL` to any OpenAI-compatible server that serves the model (e.g. the team's Akash server from `deploy/akash/akash_llm.py`) and put its key in the `REPRO_LLM_KEY` environment variable.

## Peer testing your own grants

Set `OWN_GRANTS` in the first cell to a CSV/XLSX in the Grant Tracker's format (columns `Program`, `Type`, `Amount / Benefit`, `Deadline`, `Core Requirements`) and run in live mode; step 6 prints each grant's final Fit and the reason. The TA's hidden test runs the same way if its files are in `hiddenTest/` (they aren't shared in the repository).

## Files

| Path | What |
|---|---|
| `reproduce.ipynb` | the notebook |
| `repro.py` | helpers it uses: model choice, vLLM server, running and scoring the two steps |
| `bundle/data/` | the labeled sets (`real_grants.csv` without the sponsor's "Applied?" column, which the pipeline never reads; the synthetic sets use fictional funders) |
| `bundle/cache/` | Qwen3-32B's answers for those sets only (no hidden-test answers) |
| `bundle/expected_results.csv` | the scores replay must give |
| `build_bundle.py` | maintainers: rebuilds `bundle/` from the team's full caches after new runs |
| `wave/` | WAVE setup script and Slurm jobs |

## Troubleshooting

- **`vLLM exited`**: the cell prints the end of `outputs/reproduce/vllm.log`. Out of memory: use a bigger GPU or set `MODEL = "Qwen/Qwen3-32B-AWQ"`. On a T4 or V100, vLLM support is best effort.
- **Not enough disk**: set `HF_HOME` to a disk with room for the weights before running the first cell.
- **Disconnected mid-run**: rerun the cells. Live answers are cached in `outputs/reproduce/live/<model>/cache/`, so only unanswered questions are asked again.
- **Live scores differ a little from ours**: expected. Greedy decoding can still change on a different GPU, batch size or quantization. The agreement table shows which grants changed.
