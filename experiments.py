"""
Case Study - Option A: Robustness of ASR in noisy environments.

Clean speech (LibriSpeech test-clean) is mixed with MUSAN noise at 0/10/20 dB SNR.
Whisper is evaluated on the raw noisy audio (baseline) and on audio processed by
spectral subtraction (experimental). Everything is written to ./results for report.py.
"""
import os, io, sys, json, time, random, tarfile, pickle, platform, urllib.request
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

SR = 16000

DEFAULT_CFG = dict(
    work_dir="work",
    out_dir="results",
    seed=42,
    utts_per_speaker=3,          # test-clean has 40 speakers -> 120 utterances
    min_dur=2.0, max_dur=15.0,   # seconds
    snrs=[0, 10, 20],
    noise_types=["babble", "music", "noise"],   # babble = 4 summed MUSAN speech clips (cafe chatter proxy)
    noise_files_per_type=20,
    models=["openai/whisper-tiny.en", "openai/whisper-base.en"],
    primary_model="openai/whisper-base.en",
    batch_size=16,
    ss=dict(win=400, n_fft=512, hop=160, alpha=2.0, beta=0.02, noise_pct=10),
)

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------- data
def _extract(tar, dest):
    try:
        tar.extractall(dest, filter="data")
    except TypeError:
        tar.extractall(dest)


def prepare_librispeech(cfg):
    root = os.path.join(cfg["work_dir"], "LibriSpeech", "test-clean")
    if not os.path.isdir(root):
        os.makedirs(cfg["work_dir"], exist_ok=True)
        tgz = os.path.join(cfg["work_dir"], "test-clean.tar.gz")
        if not os.path.exists(tgz):
            log("Downloading LibriSpeech test-clean (~346 MB) ...")
            urllib.request.urlretrieve(LIBRISPEECH_URL, tgz)
        with tarfile.open(tgz) as t:
            _extract(t, cfg["work_dir"])
    items = []
    for spk in sorted(os.listdir(root)):
        for chap in sorted(os.listdir(os.path.join(root, spk))):
            d = os.path.join(root, spk, chap)
            with open(os.path.join(d, f"{spk}-{chap}.trans.txt")) as f:
                for line in f:
                    uid, txt = line.strip().split(" ", 1)
                    p = os.path.join(d, uid + ".flac")
                    dur = sf.info(p).duration
                    if cfg["min_dur"] <= dur <= cfg["max_dur"]:
                        items.append(dict(id=uid, speaker=spk, text=txt, path=p, dur=dur))
    rng = random.Random(cfg["seed"])
    by = {}
    for it in items:
        by.setdefault(it["speaker"], []).append(it)
    sel = []
    for spk in sorted(by):
        sel += rng.sample(by[spk], min(cfg["utts_per_speaker"], len(by[spk])))
    return sel


def _to_16k(x, sr):
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = signal.resample_poly(x, SR // g, int(sr) // g)
    return x.astype(np.float32)


def load_musan(cfg):
    """Stream the MUSAN tarball and keep a few clips per category (stops early)."""
    cache = os.path.join(cfg["work_dir"], "musan_subset.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    need = cfg["noise_files_per_type"]
    bank = {"speech": [], "music": [], "noise": []}
    names = {"speech": [], "music": [], "noise": []}
    log("Streaming MUSAN (11 GB archive; stops as soon as enough clips are found) ...")
    try:
        resp = urllib.request.urlopen(MUSAN_URL)
        with tarfile.open(fileobj=resp, mode="r|gz") as t:
            for m in t:
                if not (m.isfile() and m.name.endswith(".wav")):
                    continue
                parts = m.name.split("/")
                cat = parts[1] if len(parts) > 2 else None
                if cat in bank and len(bank[cat]) < need:
                    x, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32")
                    x = _to_16k(x, sr)[: 60 * SR]
                    if len(x) > 2 * SR and np.mean(x ** 2) > 1e-10:
                        bank[cat].append(x)
                        names[cat].append(os.path.basename(m.name))
                if all(len(v) >= need for v in bank.values()):
                    break
    except Exception as e:  # network drop: continue if we already have something usable
        log("MUSAN stream interrupted:", repr(e))
    if any(len(v) < 3 for v in bank.values()):
        raise RuntimeError(f"Not enough MUSAN clips collected: { {k: len(v) for k, v in bank.items()} }. Re-run the cell.")
    out = dict(bank=bank, names=names)
    os.makedirs(cfg["work_dir"], exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


# ----------------------------------------------------------------- signal processing
def _crop(x, n, rng):
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    st = int(rng.integers(0, len(x) - n + 1))
    return x[st:st + n]


def make_noise(kind, n, rng, bank):
    if kind == "babble":
        segs = [_crop(bank["speech"][int(rng.integers(len(bank["speech"])))], n, rng) for _ in range(4)]
        return sum(s / (np.sqrt(np.mean(s ** 2)) + 1e-8) for s in segs).astype(np.float32)
    return _crop(bank[kind][int(rng.integers(len(bank[kind])))], n, rng).astype(np.float32)


def mix_at_snr(s, n, snr_db):
    ps, pn = np.mean(s ** 2), np.mean(n ** 2) + 1e-12
    g = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    y = s + g * n
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * (0.99 / peak)          # level change only; SNR is unchanged
    return y.astype(np.float32)


def measured_snr(s, y_noisy_unscaled_noise):  # helper used in sanity checks
    return 10 * np.log10(np.mean(s ** 2) / (np.mean(y_noisy_unscaled_noise ** 2) + 1e-12))


def spectral_subtraction(x, p):
    """Power spectral subtraction (Boll, 1979) with over-subtraction and a spectral floor.
    Noise PSD = mean power of the lowest-energy p['noise_pct'] % of frames."""
    ov = p["win"] - p["hop"]
    _, _, Z = signal.stft(x, fs=SR, window="hann", nperseg=p["win"], noverlap=ov, nfft=p["n_fft"])
    pw = np.abs(Z) ** 2
    e = pw.sum(0)
    k = max(3, int(len(e) * p["noise_pct"] / 100))
    npsd = pw[:, np.argsort(e)[:k]].mean(axis=1, keepdims=True)
    clean_pw = np.maximum(pw - p["alpha"] * npsd, p["beta"] * pw)
    Zc = np.sqrt(clean_pw) * np.exp(1j * np.angle(Z))
    _, y = signal.istft(Zc, fs=SR, window="hann", nperseg=p["win"], noverlap=ov, nfft=p["n_fft"])
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


# ----------------------------------------------------------------- ASR + metrics
def get_normalizer(pipe):
    try:
        f = pipe.tokenizer.normalize
        f("Hello, World 20!")
        return f
    except Exception:
        import re
        return lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", s.lower())).strip()


TOKENS_PER_SEC_CAP = 8   # normal English speech is ~4 tokens/s, so this allows 2x headroom


def transcribe(pipe, wavs, bs, use_cuda):
    """Length-capped decoding: clips are batched by duration and each batch may emit at most
    TOKENS_PER_SEC_CAP tokens per second of its longest clip (+16). This stops Whisper's
    repetition/hallucination loops from running to the 448-token limit."""
    import torch
    order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
    texts = [None] * len(wavs)
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        cap = int(TOKENS_PER_SEC_CAP * max(len(wavs[i]) for i in idx) / SR) + 16
        out = pipe([{"raw": wavs[i].astype(np.float32), "sampling_rate": SR} for i in idx],
                   batch_size=len(idx), generate_kwargs={"max_new_tokens": cap})
        for i, o in zip(idx, out):
            texts[i] = o["text"]
    if use_cuda:
        torch.cuda.synchronize()
    return texts, time.perf_counter() - t0


def utt_counts(ref, hyp):
    import jiwer
    if not ref.strip():
        return None
    if not hyp.strip():
        n, c = len(ref.split()), len(ref)
        return (0, n, 0, n, 0, c, 0, c)
    w = jiwer.process_words(ref, hyp)
    c = jiwer.process_characters(ref, hyp)
    return (w.substitutions, w.deletions, w.insertions, w.substitutions + w.deletions + w.hits,
            c.substitutions, c.deletions, c.insertions, c.substitutions + c.deletions + c.hits)


# ----------------------------------------------------------------- main
def run(cfg=None):
    import torch
    from transformers import pipeline
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    log("Device:", device, torch.cuda.get_device_name(0) if use_cuda else "(no GPU: Runtime > Change runtime type > GPU is much faster)")

    utts = prepare_librispeech(cfg)
    clean = [_to_16k(*sf.read(u["path"], dtype="float32")) for u in utts]
    musan = load_musan(cfg)
    bank = musan["bank"]
    log(f"{len(utts)} utterances, {len({u['speaker'] for u in utts})} speakers, {sum(u['dur'] for u in utts) / 60:.1f} min")

    noises = {}
    for i in range(len(utts)):
        for k, kind in enumerate(cfg["noise_types"]):
            noises[(i, kind)] = make_noise(kind, len(clean[i]), np.random.default_rng(cfg["seed"] + 1000 * i + k), bank)

    conds = [("clean", None)] + [(n, s) for n in cfg["noise_types"] for s in cfg["snrs"]]
    wav_cache, ss_time = {}, {}

    def get_wavs(cond):
        if cond not in wav_cache:
            noise, snr = cond
            noisy = clean if noise == "clean" else [mix_at_snr(clean[i], noises[(i, noise)], snr) for i in range(len(utts))]
            t0 = time.perf_counter()
            enh = [spectral_subtraction(x, cfg["ss"]) for x in noisy]
            ss_time[cond] = time.perf_counter() - t0
            wav_cache[cond] = (noisy, enh)
        return wav_cache[cond]

    rows, timing, params = [], [], {}
    total_audio = float(sum(len(c) for c in clean)) / SR

    for model in cfg["models"]:
        log("\n=== Model:", model)
        pipe = pipeline("automatic-speech-recognition", model=model, device=0 if use_cuda else -1)
        params[model] = int(sum(p.numel() for p in pipe.model.parameters()))
        norm = get_normalizer(pipe)
        transcribe(pipe, clean[:2], 2, use_cuda)  # warm-up
        for cond in conds:
            noisy, enh = get_wavs(cond)
            for pname, wavs, extra in (("baseline", noisy, 0.0), ("spectral_sub", enh, ss_time[cond])):
                hyps, dt = transcribe(pipe, wavs, cfg["batch_size"], use_cuda)
                for u, h in zip(utts, hyps):
                    ref, hyp = norm(u["text"]), norm(h)
                    c = utt_counts(ref, hyp)
                    if c is None:
                        continue
                    rows.append(dict(model=model, pipeline=pname, noise=cond[0], snr=cond[1] if cond[1] is not None else np.nan,
                                     utt_id=u["id"], speaker=u["speaker"], ref=ref, hyp=hyp,
                                     wS=c[0], wD=c[1], wI=c[2], wN=c[3], cS=c[4], cD=c[5], cI=c[6], cN=c[7]))
                timing.append(dict(model=model, pipeline=pname, noise=cond[0], snr=cond[1] if cond[1] is not None else np.nan,
                                   asr_s=dt, ss_s=extra, audio_s=total_audio, n_utts=len(utts)))
                d = pd.DataFrame(rows)
                m = d[(d.model == model) & (d.pipeline == pname) & (d.noise == cond[0]) & (d.snr.fillna(-1) == (cond[1] if cond[1] is not None else -1))]
                log(f"{cond[0]:>6} {str(cond[1]):>4} dB {pname:>12}: WER {100 * (m.wS + m.wD + m.wI).sum() / m.wN.sum():5.1f}%  ({dt:5.1f}s)")
        del pipe
        if use_cuda:
            torch.cuda.empty_cache()

    # example for the annotated spectrogram
    idx = next((i for i, u in enumerate(utts) if 4.0 <= u["dur"] <= 8.0), 0)
    noisy, enh = (lambda c: (c[0][idx], c[1][idx]))(get_wavs(("babble", 0)))
    np.savez_compressed(os.path.join(out_dir, "spec_example.npz"), clean=clean[idx], noisy=noisy, enhanced=enh,
                        utt_id=utts[idx]["id"])

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_utt.csv"), index=False)
    pd.DataFrame(timing).to_csv(os.path.join(out_dir, "timing.csv"), index=False)
    import transformers, librosa, jiwer
    meta = dict(
        n_utts=len(utts), n_speakers=len({u["speaker"] for u in utts}), total_audio_s=total_audio,
        mean_dur_s=total_audio / len(utts), min_dur=cfg["min_dur"], max_dur=cfg["max_dur"],
        musan_counts={k: len(v) for k, v in bank.items()}, params=params, device=device,
        gpu=torch.cuda.get_device_name(0) if use_cuda else "CPU", cfg=cfg,
        versions=dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                      librosa=librosa.__version__, numpy=np.__version__),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log("\nDone. Results written to", out_dir)


if __name__ == "__main__":
    run()
