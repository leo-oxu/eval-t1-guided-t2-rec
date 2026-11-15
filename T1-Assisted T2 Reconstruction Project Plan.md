# T1-Assisted T2 Reconstruction: Project Plan

Nov 13, 2026 · @Leo

We test whether, and when, a fully sampled T1 scan helps reconstruct a fast, noisy, undersampled T2 scan on low-field (0.3 T) M4Raw data, by training the same network with and without T1.

## 1. What we are doing

This is an analysis project, not a new-architecture project. We use a standard network on purpose, so that every difference we measure comes from T1.

**The situation we simulate.** A low-field T2 scan is made faster in two ways: fewer k-space lines (undersampling by a factor R) and one repetition instead of three. The result is noisy and aliased. A good, fully sampled T1 of the same slice is available as a guide.

**The core comparison.** We train the same MoDL network twice, once without T1 and once with T1, with identical size, data and training budget.

**The three questions:**

1. How much does T1 improve the T2 image?
2. How does that benefit change as the scan gets faster (R = 4, 6, 8, …)?
3. When does T1 stop helping or start hurting (misalignment, wrong or blank T1, features that only T2 shows)?

**What counts as success (proposed, to confirm together).** Hypothesis: T1 gives a statistically significant per-subject gain over the T2-only model, and the gain grows with R. Either outcome is reportable; a failure case we can explain is also a result.

**How we differ from prior work.** Two papers already do guided reconstruction on M4Raw: TGVN (FLAIR guided by T2) and CoSMo-RecNet (T2 guided by T1, focused on small training sets). Neither systematically measures how the T1 benefit changes with R, guide quality and misalignment at low field. That analysis, plus the lessons it gives for method design, is our contribution.

## 2. Setup already decided

We frame the problem the way the credible raw-data papers do, mainly [TGVN](https://arxiv.org/abs/2501.03021) and [CoSMo-RecNet](https://arxiv.org/abs/2609.01959), which both use M4Raw. The full evidence for each point is in the literature review doc.

| Decision | What it means in practice | Follows |
| --- | --- | --- |
| Target and guide | Reconstruct T2; T1 is the guide. | Our proposal; CoSMo-RecNet |
| Guided (conditional) reconstruction | Only T2 is undersampled; T1 is fully sampled. | TGVN, CoSMo-RecNet |
| 4-coil forward model | y = mask · FFT · coil maps · x, using all 4 M4Raw coils (not a virtual single coil). | TGVN, CoSMo-RecNet, MoDL |
| Undersampling | 1D Cartesian along phase encoding, with a fully sampled centre (ACS). No 2D masks. | M4Raw paper, TGVN, CoSMo-RecNet |
| Backbone | MoDL: shared CNN denoiser + conjugate-gradient data consistency. T1 enters only the denoiser. | MoDL (Aggarwal et al.), CoSMo-RecNet |
| Alignment | Align T1 to T2 before use; misregistration is tested as a robustness experiment. | CoSMo-RecNet, TGVN, Xuan et al. |
| Input and target | Input: one undersampled T2 repetition. Reference: multi-repetition averaged RSS image. | TGVN, M4Raw paper |
| Core comparison | Size-matched MoDL with vs without T1, plus zero-filled, classical CS and one guided baseline. | TGVN, CoSMo-RecNet, jVN |
| Evaluation | SSIM, PSNR, NRMSE; per-subject paired Wilcoxon tests; robustness to misaligned or degraded T1. | TGVN, CoSMo-RecNet |

Two facts about M4Raw that shape the setup:

- **There is a systematic T1–T2 shift** of about 1.5 pixels along frequency encoding, caused by B0 drift between the first (T1) and last (T2) scan.
- **There is no clean reference.** The 3-repetition T2 average has SNR of only about 20 (single repetition: about 12).

## 3. Methods we compare

The minimum set has only two trained networks. Each other method is a reference point that answers one question a reader will ask; none is a separate study.

| Method | Question it answers | Needs training? | Priority |
| --- | --- | --- | --- |
| Zero-filled (inverse FFT + RSS) | How bad is the undersampled input? | No | Must |
| Classical CS (FISTA-TV or L1-wavelet, with coil maps) | Does learning beat a standard method? Also a debugging check: if MoDL does not beat it, the pipeline has a bug. | No (tune λ on validation) | Must |
| MoDL without T1 | What can be recovered from T2 data alone? This is the control. | Yes | Must |
| MoDL with T1 | What does T1 add? This is the main model. | Yes | Must |
| U-Net with T1 as an extra input channel | Is our way of using T1 reasonable compared with a simple published idea (Xiang et al.)? | Yes (small) | Nice to have |
| Small TGVN (public code, retrained for T2 guided by T1) | How do we compare with a recent published guided method? Could replace the U-Net baseline. | Yes (heavy) | Only if GPU time allows |
| Directional TV (classical, uses T1 edges) | Does T1 help even without learning? | No | Optional |

Classical methods are not there to "prove they fail". They anchor the numbers and catch bugs; being worse at high R is an expected side result.

## 4. Experiments and what they produce

Each experiment maps to one figure or table in the report, so we run only what the report needs.

| # | Experiment | What we vary | Question | Output in report | Priority |
| --- | --- | --- | --- | --- | --- |
| E1 | Main comparison | All methods at each R | How good is each method? | Main table: SSIM, PSNR, NRMSE, runtime per slice, parameters | Must |
| E2 | Benefit vs acceleration | R = 4, 6, 8 (10 if time) | Does the T1 gain grow as the scan gets faster? | Curve: (MoDL + T1) minus (MoDL) per R, with per-subject spread | Must |
| E3 | Misalignment sweep | Shift T1 by 0–4 px, rotate 0–4°, use adjacent slice; models trained with and without shift augmentation | When does misalignment cancel the benefit? | Curve: T1 gain vs size of error | Must |
| E4 | Wrong or missing guide | T1 from another subject; blank T1 | Does the model fall back to T2-only quality, or copy the wrong T1? | Rows in a robustness table + example images | Must |
| E5 | Guide quality | Single-repetition T1 vs 3-repetition averaged T1 | Does a noisy guide limit the benefit? | Small table | Should |
| E6 | Synthetic lesions | Lesion inserted in T2 only (must survive) or in T1 only (must not appear); several sizes and R | Does T1 erase T2-only features or create false ones? | Image panel + detectability numbers | Nice to have |
| E7 | Generality checks | T2 → T1 direction; E2E-VarNet backbone | Do the trends hold in another setting? | Short paragraph or appendix | Only if time |

Also planned: one example figure showing all methods on the same slice, with error maps and a zoomed region.

## 5. Week-by-week checklist

Tick items as they are done and add your initials to the ones you own. Each week ends with a checkpoint; if it fails, fix it before moving on.

### Week 1: data

- [ ] Download M4Raw; write the loader for k-space, coils and repetitions
- [ ] Check for all-zero (padded) phase-encode lines
- [ ] Measure the T1–T2 shift per subject; visualise sample slices
- [ ] Fix the train / validation / test split by subject
- [ ] Precompute and save coil maps (ESPIRiT), masks and averaged references

Checkpoint: loader returns correct k-space, coil maps look smooth, alignment numbers are recorded.

### Week 2: classical baselines

- [ ] Zero-filled reconstruction at each R
- [ ] FISTA-TV (or L1-wavelet) with the 4-coil model; tune λ on validation
- [ ] Metric code: SSIM / PSNR / NRMSE inside a brain mask, fixed settings

Checkpoint: classical results improve clearly over zero-filled at R = 4.

### Week 3: T2-only MoDL (control)

- [ ] Implement MoDL with CG data consistency
- [ ] Train at each chosen R (or one model across R)

Checkpoint: MoDL clearly beats FISTA-TV. If not, debug before adding T1.

### Week 4: T1-assisted MoDL (main model)

- [ ] Add T1 to the denoiser input; match parameter count with the control
- [ ] Train with identical settings; run E1 and E2
- [ ] Optional: U-Net + T1 baseline

Checkpoint: main table and benefit-vs-R curve exist.

### Week 5: failure cases

- [ ] E3 misalignment sweep (with and without shift augmentation)
- [ ] E4 wrong-subject and blank T1
- [ ] E5 single vs averaged T1
- [ ] Optional: E6 synthetic lesions

### Week 6: final results

- [ ] Final test-set runs; per-subject Wilcoxon tests
- [ ] All figures and tables

### Week 7: report and delivery

- [ ] Report (including "implications for method design"), presentation, poster
- [ ] Code clean-up, README, configs, fixed seeds

## 6. Decisions still open

Compute, data split and model size come first, because they limit everything else. All suggested defaults are agreed on my side (13 Nov); Jackie to confirm or flag any row to change.

| Decision | Options | Suggested default | Status |
| --- | --- | --- | --- |
| Compute (we have no GPU of our own) | Course or U of T cluster; Colab Pro; Kaggle | Google Colab (no school GPU available); Kaggle as backup. Save checkpoints to Drive every epoch | Agreed |
| Data split | TGVN's released M4Raw split; our own split of the validation set | Use TGVN's public split, so numbers are comparable | Agreed |
| Motion-corrupted subjects | Include; exclude; robustness test only | Exclude from main results | Agreed |
| Acceleration factors and ACS | Which R values; centre width | R = 4, 6, 8 (+10), where R is the true net acceleration (fastMRI-style: total sampled lines = 195 acquired / R, ACS included); fixed ACS of 12 lines. Decided 14 Nov: with 16 ACS lines on top of every R-th line, net R only reached 3.2–5.7 | Decided |
| Mask type | Equispaced; random; both | Equispaced as main; random as second, with fixed seeds | Agreed |
| Which repetitions form the target | Average of all 3; average of the other two | Input = repetition k, target = mean of the other two; rotate k for 3× data | Agreed |
| Averaging domain | Magnitude RSS (M4Raw default); complex | Complex if repetitions are phase-stable, otherwise magnitude (stated in report) | Agreed |
| T1 guide quality | Single repetition; 3-repetition average | Averaged T1 for main results; single as E5 | Agreed |
| Alignment method | Register with the full T2; with the T2 ACS image only | Training pairs: full registration. Test time: ACS-only alignment (realistic) | Agreed |
| MoDL size | Unrolls K, CG steps, λ, channels | K = 5–8, 5–10 CG steps, learned λ, under about 2 M parameters | Agreed |
| How T1 enters | Concatenate T1; small T1 encoder; learned trust map | Concatenate T1 (or encoder features) at every unroll | Agreed |
| Models per R | One model per R; one shared model across R | One per R if compute allows | Agreed |
| Loss | L1; L1 + SSIM | L1 inside a brain mask (+ SSIM optional) | Agreed |
| Training budget and seeds | Epochs, schedule, number of runs | Identical for all learned models; 2 seeds if possible | Agreed |
| Guided baseline | U-Net + T1; small TGVN; none | U-Net + T1; TGVN only if GPU time allows | Agreed |
| Success criteria | As proposed in Section 1, or stricter | Section 1 version | Agreed |
| Who does what | Split by pipeline stage | e.g. one person: data, masks, classical, metrics; other: MoDL, training, robustness | Agreed |

Also check the course rubric (report length, poster and presentation format, deadlines) so the experiments match what is graded.

## 7. Risks, and lessons for future work

| Risk | Why it matters | What we do |
| --- | --- | --- |
| No GPU of our own | Training several networks may not fit the one-month window. | Colab: small MoDL, mixed precision, compact precomputed data, checkpoints on Drive so a disconnect loses little; drop optional items first. |
| Unfair comparison | A bigger or longer-trained T1 model could win for the wrong reason. | Match parameters, data and training budget; also test with a blank T1. |
| Test-time leakage | Using the fully sampled T2 to align, normalise or tune at test time inflates results. | Use only input or ACS data at test time; tune on validation subjects only. |
| Noisy reference | Averaged T2 SNR is only about 20, so PSNR penalises extra denoising. | Metrics inside a brain mask, error maps, and target built from the other repetitions. |
| Healthy-only data | No real lesions in M4Raw. | Synthetic lesions (optional E6); state the limitation clearly. |
| Motivation wording | One M4Raw T2 repetition (72 s) is shorter than one T1 repetition (98 s), because T2 uses fast spin echo. | Reword: T2 has the lowest SNR, needs about 3 repetitions, and carries the key clinical information. |
| Close prior work | TGVN and CoSMo-RecNet already do guided reconstruction on M4Raw. | Cite both; position our work as the analysis of when T1 helps and when it fails. |

**Lessons for future work.** End the report with a short "implications for method design" part that turns each finding into a method idea:

- a fast loss of benefit under small shifts → built-in alignment
- copying a wrong T1 → a learned trust map or TGVN-style limit on T1
- a big gain from averaged T1 → denoise or average the guide
- erased T2-only lesions → a loss that protects features absent from T1

## Sources

Full literature review, with each method's weaknesses and the evidence behind every decision: T1-Assisted T2 MRI Reconstruction: Methods and Weaknesses

- [M4Raw dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172399/) — Lyu et al., Scientific Data 2023
- [TGVN: trust-guided reconstruction with side information](https://arxiv.org/abs/2501.03021) — Atalık et al., IEEE TMI; code and M4Raw splits at [github.com/sodicksonlab/TGVN](https://github.com/sodicksonlab/TGVN)
- [CoSMo-RecNet](https://arxiv.org/abs/2609.01959) — Rao et al., preprint 2026
- [MoDL](https://doi.org/10.1109/TMI.2018.2865356) — Aggarwal et al., IEEE TMI 2019
- [Ultra-fast T2 reconstruction using T1 information](https://doi.org/10.1007/978-3-030-00928-1_25) — Xiang et al., MICCAI 2018
- [Spatial alignment network for multi-modal reconstruction](https://arxiv.org/abs/2108.05603) — Xuan et al., IEEE TMI 2022
- [jVN: joint multi-contrast variational network](https://doi.org/10.1002/mrm.28219) — Polak et al., MRM 2020
- [Implicit data crimes](https://doi.org/10.1073/pnas.2117203119) — Shimron et al., PNAS 2022
