# ASR Robustness in Noisy Environments: Whisper Baseline vs. Spectral Subtraction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg?logo=huggingface)](https://huggingface.co/)
[![Colab](https://img.shields.io/badge/Google%20Colab-Run%20Notebook-F9AB00.svg?logo=googlecolab)](https://colab.research.google.com/drive/1cVdPJF4eKLUXxuBnFlRTO9w9nqtW-u-7)
[![Course](https://img.shields.io/badge/Course-Speech%20Recognition%20%26%20Conversational%20Systems-green.svg)](#academic-metadata)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An empirical investigation into the acoustic robustness of end-to-end automatic speech recognition (ASR) under severe environmental degradation. This project benchmarks **OpenAI Whisper** across diverse noise topologies (multi-talker babble, infotainment music, and technical ambient noise) at 0 dB, 10 dB, and 20 dB SNR, and evaluates whether classical DSP front-end speech enhancement (**Boll's Power Spectral Subtraction**) can mitigate recognition degradation without model retraining.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Research Questions](#research-questions)
- [System Architecture & Methodology](#system-architecture--methodology)
  - [Evaluation Pipeline](#evaluation-pipeline)
  - [Acoustic Datasets & Synthetic Mixing](#acoustic-datasets--synthetic-mixing)
  - [Digital Signal Processing (DSP) Front-End](#digital-signal-processing-dsp-front-end)
  - [ASR Back-End & Length-Capped Decoding Patch](#asr-back-end--length-capped-decoding-patch)
- [Experimental Results](#experimental-results)
  - [Condition-Level Benchmark (Whisper-base.en)](#condition-level-benchmark-whisper-baseen)
  - [Model Capacity & Computational Footprint Comparison](#model-capacity--computational-footprint-comparison)
  - [Key Findings](#key-findings)
- [Qualitative Error Analysis](#qualitative-error-analysis)
- [Engineering Trade-Offs & Edge Deployment](#engineering-trade-offs--edge-deployment)
- [Project Structure](#project-structure)
- [Getting Started & Reproduction](#getting-started--reproduction)
  - [Prerequisites](#prerequisites)
  - [Option A: Google Colab (Recommended)](#option-a-google-colab-recommended)
  - [Option B: Local Environment Execution](#option-b-local-environment-execution)
- [Configuration Reference](#configuration-reference)
- [Academic Metadata & Citations](#academic-metadata--citations)

---

## Executive Summary

Voice assistants deployed in automotive cockpits and smart home environments face persistent, non-stationary acoustic interference (cockpit cabin rumble, multi-speaker dinner/cafe chatter, and background audio streaming). This case study analyzes:
1. The degradation profile of transformer-based ASR (**Whisper-tiny.en** [37.8M parameters] and **Whisper-base.en** [72.6M parameters]) when clean speech is degraded from 20 dB SNR down to 0 dB SNR.
2. The efficacy of adding a single-channel classical speech enhancement front-end (**Power Spectral Subtraction with over-subtraction and spectral flooring**) prior to log-Mel feature extraction.

### Core Takeaways
- **Clean Speech Baseline**: `Whisper-base.en` achieves a clean Word Error Rate (**WER**) of **3.7%** and Character Error Rate (**CER**) of **1.2%**.
- **Severe Degradation under Babble**: Overlapping speech (cafe chatter / 4-talker babble) at 0 dB is the most destructive noise type, escalating WER to **270.5%** for `base.en` and **648.9%** for `tiny.en` due to severe informational masking and decoder hallucination loops.
- **Spectral Subtraction Fails on Average**: Across the 9 noisy conditions, classical spectral subtraction increased average WER from **38.8%** to **43.9%** (+13.3% relative degradation). It reduced WER in only **2 of 9** conditions (notably 0 dB music, where sustained harmonics were attenuated: 30.8% $\to$ 22.9%).
- **DSP Artefacts vs. In-Domain Training**: Uncorrelated spectral subtraction introduces musical noise artifacts and removes unvoiced consonant energy, harming neural acoustic models trained primarily on natural acoustic distributions.
- **Model Capacity Trumps Heuristic Filtering**: Scaling from Whisper-tiny to Whisper-base reduces average noisy WER by more than half (**86.7% $\to$ 38.8%**) at ~1.8× latency cost, proving that internal model representations handle noise significantly better than hand-crafted DSP filters.

---

## Research Questions

| Question | Investigation Focus | Empirical Finding |
| :--- | :--- | :--- |
| **RQ1: Noise Degradation** | How do distinct noise classes (Babble, Music, Ambient) impact ASR accuracy across SNR levels? | Babble causes exponential degradation due to informational masking and token repetition loops; Music creates harmonic substitutions; Ambient noise primarily masks high-frequency consonants. |
| **RQ2: Front-End Enhancement** | Does classical single-channel spectral subtraction recover recognition performance without retraining? | **No.** Spectral subtraction degrades recognition in 7 of 9 conditions due to musical noise and formantic distortion, confirming that classical subtraction is unsuitable as a general front-end for modern neural ASR. |

---

## System Architecture & Methodology

### Evaluation Pipeline

The evaluation pipeline follows a dual-branch design to isolate the impact of front-end DSP enhancement:

```
                            ┌───────────────────────────────────────────────┐
                            │ Clean Speech (LibriSpeech test-clean, 16 kHz) │
                            └───────────────────────┬───────────────────────┘
                                                    │
                                                    ▼
                            ┌───────────────────────────────────────────────┐
                            │   Synthetic Noise Mixing (MUSAN at 0/10/20dB) │
                            │   Babble (4x Speech), Music, Ambient Noise    │
                            └───────────────────────┬───────────────────────┘
                                                    │
                         ┌──────────────────────────┴──────────────────────────┐
                         │                                                     │
                         ▼                                                     ▼
              [Branch 1: Baseline]                                 [Branch 2: Experimental]
           ┌───────────────────────┐                             ┌───────────────────────────┐
           │ Raw Noisy Audio (y)   │                             │ Power Spectral Subtraction│
           └───────────┬───────────┘                             │ α = 2.0, β = 0.02, 10% PSD│
                       │                                         └─────────────┬─────────────┘
                       │                                                       │
                       │                                                       ▼
                       │                                         ┌───────────────────────────┐
                       │                                         │ Enhanced Waveform (y_enh) │
                       │                                         └─────────────┬─────────────┘
                       │                                                       │
                       └──────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │  80-channel Log-Mel Feature Extract │
                               │     (25ms Hann Window, 10ms Hop)    │
                               └──────────────────┬──────────────────┘
                                                  │
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │  Whisper ASR (tiny.en / base.en)    │
                               │  Length-Capped Greedy Decoding      │
                               └──────────────────┬──────────────────┘
                                                  │
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │ Text Normalization (Whisper Normal.)│
                               │ Metric Computation (WER, CER, RTF)  │
                               └─────────────────────────────────────┘
```

### Acoustic Datasets & Synthetic Mixing

- **Clean Speech**: [LibriSpeech](https://www.openslr.org/12/) `test-clean` corpus. A balanced sample of **120 utterances across 40 distinct speakers** (3 utterances per speaker, duration: 2.0 s – 15.0 s, total duration: 13.9 minutes).
- **Interfering Noise**: [MUSAN Corpus](https://www.openslr.org/17/) (Music, Speech, and Noise).
  - **Babble (Cafe Chatter Proxy)**: 4 distinct speech streams sampled randomly, power-normalized, and mixed synchronously.
  - **Music**: Commercial and instrumental recordings.
  - **Ambient Noise**: Technical and environmental noise sources (appliances, air conditioners, room rumble).
- **Exact SNR Mixing**:
  For clean speech $s(t)$ and noise $n(t)$, noise gain $g$ is calibrated via root-mean-square powers:
  $$g = \sqrt{\frac{P_s}{P_n \cdot 10^{\text{SNR}_{\text{dB}} / 10}}}$$
  $$y(t) = s(t) + g \cdot n(t)$$
  Levels are normalized to avoid digital clipping ($\max |y(t)| \le 0.99$) without modifying relative SNR.

### Digital Signal Processing (DSP) Front-End

The experimental branch implements **Boll's Power Spectral Subtraction (1979)** with over-subtraction and spectral flooring:
1. **Short-Time Fourier Transform (STFT)**:
   - Window size: $N_{\text{win}} = 400$ samples (25 ms at 16 kHz) with Hann tapering.
   - Hop length: $N_{\text{hop}} = 160$ samples (10 ms).
   - FFT resolution: $N_{\text{fft}} = 512$ points.
2. **Noise Power Spectral Density (PSD) Estimation**:
   - Estimated non-parametrically from the lowest-energy 10% of frames in the STFT spectrogram:
     $$\hat{P}_N(k) = \frac{1}{|K_{\text{low}}|} \sum_{m \in K_{\text{low}}} |Y(k, m)|^2$$
3. **Spectral Over-Subtraction & Flooring**:
   $$|\hat{S}(k, m)|^2 = \max\left(|Y(k, m)|^2 - \alpha \hat{P}_N(k), \; \beta |Y(k, m)|^2\right)$$
   Where:
   - $\alpha = 2.0$ (over-subtraction factor to suppress musical noise spikes).
   - $\beta = 0.02$ (spectral floor preventing spectral nulls).
4. **Waveform Synthesis (iSTFT)**: Reconstructed using original noisy phase $\angle Y(k, m)$ via overlap-add synthesis.

### ASR Back-End & Length-Capped Decoding Patch

In severe babble noise (0 dB SNR), standard Whisper beam/greedy search exhibits pathological **hallucination and infinite repetition loops** (e.g., repeatedly generating phrases until hitting the hard 448-token limit). This causes artificial WER blowups (>300–800%).

To stabilize decoding and ensure valid real-time throughput metrics, a length-capping heuristic is enforced:
```python
TOKENS_PER_SEC_CAP = 8  # Normal English speech is ~4 tokens/sec (2x headroom)

def transcribe(pipe, wavs, bs, use_cuda):
    order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
    texts = [None] * len(wavs)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        # Restrict max new tokens proportional to the longest audio clip in batch
        cap = int(TOKENS_PER_SEC_CAP * max(len(wavs[i]) for i in idx) / SR) + 16
        out = pipe(
            [{"raw": wavs[i].astype(np.float32), "sampling_rate": SR} for i in idx],
            batch_size=len(idx),
            generate_kwargs={"max_new_tokens": cap}
        )
        for i, o in zip(idx, out):
            texts[i] = o["text"]
    return texts
```

---

## Experimental Results

All error rates are calculated over 120 test utterances using `jiwer` after applying Whisper's standard text normalizer. Statistical significance is verified via **95% Bootstrap Confidence Intervals** (1,000 resamples).

### Condition-Level Benchmark (`Whisper-base.en`)

| Condition | SNR (dB) | Baseline WER (%) | Spec. Sub. WER (%) | Baseline CER (%) | Spec. Sub. CER (%) | Relative WER Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Speech** | $\infty$ | **3.7** | 3.8 | 1.2 | 1.3 | +2.4% *(DSP penalty)* |
| **Babble (Cafe)** | 20 dB | **4.1** | 4.3 | 1.5 | 2.0 | +4.2% |
| **Babble (Cafe)** | 10 dB | **8.9** | 10.0 | 4.6 | 5.6 | +12.1% |
| **Babble (Cafe)** | 0 dB | **270.5** | 322.1 | 237.4 | 263.6 | +19.1% |
| **Music** | 20 dB | **4.0** | 4.1 | 1.4 | 1.4 | +4.3% |
| **Music** | 10 dB | **5.7** | 6.5 | 2.6 | 2.7 | +13.6% |
| **Music** | 0 dB | 30.8 | **22.9** | 19.6 | 14.3 | **-25.8% (Improved)** |
| **Ambient Noise** | 20 dB | **4.4** | 4.4 | 1.6 | 1.6 | +2.0% |
| **Ambient Noise** | 10 dB | 6.0 | **5.8** | 2.3 | 2.2 | **-3.6% (Improved)** |
| **Ambient Noise** | 0 dB | **14.7** | 15.2 | 8.4 | 8.7 | +3.5% |
| **Average (9 Noisy)** | — | **38.8** | **43.9** | 31.9 | 34.4 | **+13.3% (Overall Degradation)** |

### Model Capacity & Computational Footprint Comparison

Evaluated on an **NVIDIA Tesla T4 GPU (16 GB)** with batch size 16 across 13.9 minutes of test speech per condition:

| Metric | `openai/whisper-tiny.en` | `openai/whisper-base.en` | Relative Change / Scaling Impact |
| :--- | :---: | :---: | :---: |
| **Parameter Count** | 37.8 M | 72.6 M | +92.1% parameters |
| **FP32 Weight Size** | 151 MB | 290 MB | +92.1% memory footprint |
| **Clean Speech WER** | 5.1% | **3.7%** | -27.5% relative error reduction |
| **Mean Noisy WER (Baseline)**| 86.7% | **38.8%** | **-55.2% relative error reduction** |
| **Mean Noisy WER (Spec. Sub.)**| 91.8% | **43.9%** | -52.2% relative error reduction |
| **ASR Inference RTF (T4)** | **0.0431** | 0.0762 | 1.77× latency increase |
| **Mean Latency per Utt.** | **300.3 ms** | 530.4 ms | +230.1 ms per utterance |
| **Spectral Subtraction RTF** | ~0.0031 (CPU) | ~0.0031 (CPU) | Minimal computational overhead |

> **Real-Time Factor (RTF)**: $\text{RTF} = \frac{\text{Processing Time (s)}}{\text{Audio Duration (s)}}$. Values $\ll 1.0$ indicate faster-than-real-time throughput.

### Key Findings

1. **Hierarchy of Acoustic Vulnerability**:
   $$\text{Babble (Cafe Chatter)} \gg \text{Music} > \text{Technical Ambient Noise}$$
   Babble noise is by far the most hostile environment. Because human speech shares identical spectro-temporal modulation properties with the target talker, the attention heads cross-attend to interfering voices, causing substitution and insertion cascades.
2. **Why Spectral Subtraction Degrades ASR**:
   - **Stationarity Assumption Failure**: Spectral subtraction relies on noise remaining stationary between estimate frames. Babble and music are highly non-stationary.
   - **Musical Noise**: Isolated residual spectral peaks appear randomly in time-frequency bins, mimicking spurious phonetic cues.
   - **Energy Attenuation in Formants**: Crucial high-frequency unvoiced consonants (stops, fricatives) are stripped during over-subtraction, degrading phonetic discriminability.
3. **The Music 0 dB Anomaly**:
   Spectral subtraction significantly improved 0 dB music (**30.8% $\to$ 22.9%**). Sustained harmonic tones (such as synthetic pads or basslines) provide stable spectral peaks that Boll's subtraction successfully suppressed, reducing harmonic interference on vowels.

---

## Qualitative Error Analysis

Error attribution on 0 dB SNR test cases demonstrates how noise topology dictates specific failure modes:

| # | Condition | Utterance ID | Ground Truth (Normalized) | Whisper-base Prediction | Primary Error Mechanism |
| :-: | :--- | :---: | :--- | :--- | :--- |
| **1** | **Babble (0 dB)** | `121-121726-0008` | `hose man is excuse for wetting the walk` | `hose` | **Deletion-Dominant (WER 88%)**: Energetic masking of low-energy function words; hypothesis truncates prematurely. |
| **2** | **Music (0 dB)** | `61-70968-0034` | `a montfichet a montfichet gamewell to the rescue` | `a multi shade a multi shade being well to the rescue` | **Substitution-Dominant (WER 75%)**: Sustained harmonic frequencies overlap speech formants, causing acoustic confusion (*"montfichet"* $\to$ *"multi shade"*). |
| **3** | **Ambient (0 dB)**| `3570-5694-0017`| `the wearing of uniforms or liveries implies...` | `the wearing of beautiful and deliberation ties...` | **Substitution-Dominant (WER 77%)**: Broadband noise raises noise floor across high-frequency fricatives, distorting consonant perception. |
| **4** | **Babble (0 dB)** | `8224-274384-0006`| `the king stood up and called for that psalm...`| `the king is still in the cold for that sound...`| **Phonetic Drift (WER 86%)**: Competing talkers inject syllable tokens that warp language model decoding. |

---

## Engineering Trade-Offs & Edge Deployment

For edge deployment in automotive cockpits (ECUs, head units) or smart appliances:

1. **Compute vs. Robustness Trade-off**:
   - `Whisper-tiny.en` is compact (151 MB, ~300 ms/utt), but collapses in loud environments (86.7% mean noisy WER).
   - `Whisper-base.en` requires 290 MB RAM and 530 ms/utt, but provides acceptable accuracy down to 10 dB SNR.
2. **Avoid Ungated Heuristic Enhancement**:
   - Running static spectral subtraction permanently penalizes clean audio (3.7% $\to$ 3.8% WER) and degrades 78% of noisy conditions.
   - **Recommendation**: If DSP enhancement is required, it **must be gated** using a Voice Activity Detector (VAD) and SNR/stationarity estimator, activating only in stationary, low-SNR conditions.
3. **Recommended Production Alternatives**:
   - **Multi-condition Training**: Fine-tune Whisper encoder weights directly with synthetic noise and room impulse response (RIR) reverberation augmentation.
   - **Learned Deep Enhancement**: Replace spectral subtraction with lightweight neural front-ends (e.g., DeepFilterNet, Conv-TasNet, or DTLN) trained end-to-end with the acoustic model.
   - **Inference Optimization**: Quantize `Whisper-base` to INT8/FP8 via TensorRT-LLM or `onnxruntime` to match `tiny`'s memory footprint while maintaining `base`'s noise robustness.

---

## Project Structure

```
.
├── SRCS_CaseStudy_OptionA.ipynb     # Complete Jupyter / Google Colab pipeline notebook
├── experiments.py                   # Experiment runner: dataset downloading, audio mixing,
│                                    # spectral subtraction, batched Whisper inference, logging
├── report.py                        # Automated ReportLab PDF generator (tables, charts, layout)
├── results/                         # Output directory generated by experiments.py
│   ├── per_utt.csv                  # Fine-grained, utterance-level alignment & error counts
│   ├── timing.csv                   # Throughput, latency, and real-time factor logs
│   ├── meta.json                    # Hardware, software versions, and configuration dump
│   ├── spec_example.npz             # Spectrogram data arrays for clean, noisy, and enhanced audio
│   └── *.pdf                        # Generated publication-quality case study report
├── .gitignore                       # Git ignore file for virtual environments and large checkpoints
└── README.md                        # Project documentation and technical report
```

---

## Getting Started & Reproduction

### Prerequisites

- **Python**: 3.10 or higher (tested on 3.10, 3.12, 3.13)
- **Hardware**: Dedicated GPU recommended (e.g., NVIDIA T4, RTX 3060+) for batched Whisper inference. CPU execution is supported but significantly slower.

### Option A: Google Colab (Recommended)

1. Open [`SRCS_CaseStudy_OptionA.ipynb`](file:///e:/Semester-7/Speech%20recog%20%26/Case%20Study/SRCS_CaseStudy_OptionA.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Enable GPU acceleration: **Runtime > Change runtime type > T4 GPU**.
3. Set your student details in Cell 2:
   ```python
   STUDENT_NAME = "G K Mahanidhi"
   ROLL_NO = "23BTRCL139"
   ```
4. Execute **Runtime > Run all**. Execution takes ~35–50 minutes (streaming and caching MUSAN audio segments is the primary bottleneck).
5. The notebook automatically downloads the generated PDF report (`23BTRCL139_G_K_Mahanidhi_CaseStudy.pdf`).

### Option B: Local Environment Execution

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Case Study"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install transformers jiwer librosa soundfile reportlab accelerate torch torchvision torchaudio
   ```

4. **Execute benchmark experiments**:
   ```bash
   python experiments.py
   ```
   *This downloads LibriSpeech test-clean (~346 MB), streams MUSAN clips, runs inference across all 20 conditions (2 models × 10 conditions), and logs outputs to `./results`.*

5. **Generate the publication PDF report**:
   ```bash
   python report.py
   ```

---

## Configuration Reference

Key experimental parameters can be adjusted in `experiments.py`:

```python
DEFAULT_CFG = dict(
    work_dir="work",                   # Cache directory for audio tarballs
    out_dir="results",                 # Directory for CSVs, NPZ, and PDF outputs
    seed=42,                           # Seed for reproducible speaker and noise sampling
    utts_per_speaker=3,                # 40 speakers x 3 utts = 120 total utterances
    min_dur=2.0, max_dur=15.0,         # Utterance duration filter (seconds)
    snrs=[0, 10, 20],                  # Target Signal-to-Noise Ratios (dB)
    noise_types=["babble", "music", "noise"],
    noise_files_per_type=20,           # Number of MUSAN files extracted per category
    models=[
        "openai/whisper-tiny.en",      # Lightweight baseline model
        "openai/whisper-base.en"       # Primary evaluation model
    ],
    primary_model="openai/whisper-base.en",
    batch_size=16,                     # Batch size for Hugging Face ASR pipeline
    ss=dict(
        win=400,                       # STFT analysis window (25 ms at 16 kHz)
        n_fft=512,                     # FFT size
        hop=160,                       # Hop size (10 ms)
        alpha=2.0,                     # Over-subtraction factor
        beta=0.02,                     # Spectral floor parameter
        noise_pct=10                   # Lowest energy percentile used for noise PSD
    )
)
```

---

## Academic Metadata & Citations

### Course Information
- **Course**: Speech Recognition and Conversational Systems (Course Code: `22AIMLDE24`)
- **Institution**: School of Computer Science and Engineering (CSE - AIML), Faculty of Engineering and Technology, Jain (Deemed-to-be University)
- **Academic Year**: 2026 – 2027 (Semester VII)
- **Instructor**: Prof. Shobha H R
- **Student Author**: G K Mahanidhi (Roll No: `23BTRCL139`, Section: `AIML-C`)

### Academic References

```bibtex
@article{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}

@article{boll1979spectral,
  title={Suppression of Acoustic Noise in Speech Using Spectral Subtraction},
  author={Boll, Steven F.},
  journal={IEEE Transactions on Acoustics, Speech, and Signal Processing},
  volume={27},
  number={2},
  pages={113--120},
  year={1979}
}

@inproceedings{panayotov2015librispeech,
  title={LibriSpeech: An ASR Corpus Based on Public Domain Audio Books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5206--5210},
  year={2015}
}

@article{snyder2015musan,
  title={MUSAN: A Music, Speech, and Noise Corpus},
  author={Snyder, David and Chen, Guoguo and Povey, Daniel},
  journal={arXiv preprint arXiv:1510.08484},
  year={2015}
}
```
