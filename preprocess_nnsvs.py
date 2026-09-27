"""NHVSing preprocess for nnsvs melf0 — 歌唱 wav ディレクトリ → 学習用 shard npz(nnsvs の特徴量で)。

ENUNU の melf0 音源(音響モデルが mel + lf0 + vuv を出す)でボコーダーとして使うための前処理。
推論時にボコーダーへ入るのは音響モデルの出力なので、学習データも nnsvs の MelF0
(nnsvs/data/data_source.py)と同じ計算で作る:

  - mel : parallel_wavegan.bin.preprocess.logmelfilterbank と同じ(librosa.stft center=True +
          reflect, hann(win) を n_fft の中央に置く, librosa の slaney mel, log10, 下限 mel_eps)。
          ★RMS 正規化はしない(nnsvs は ±1 の生波形で mel を取る)。
  - 再サンプル: 元の fs が低いとき、nnsvs と同じく 1e-7 の白色雑音を足す(乱数の種は長さ)。
  - F0  : harvest(範囲は config で固定)→ vuv は D4C の非周期性 ap[:,0] < d4c_threshold →
          log-F0 を slinear 補間 → f0_smoothing_cutoff Hz のゼロ位相ローパス。
          nnsvs が楽譜(ラベル)から決める F0 範囲と、F0 も音符も無い区間の埋め方は行わない
          (DiffSinger 用のラベルには音高が無いため)。
          保存は preprocess.py と同じく 0=無声(exp(平滑化 log-F0) × vuv)。

出力 npz キー: '<sid>|f0'(float32[T]), '<sid>|log_melspc'(float32[T, mel_dim] log10),
               '<sid>|wav'(float32[T*hop])。dataset.VocoderDataset がそのまま読む。
--segs_per_shard 0 のときは 1 segment = 1 ファイル(<sid>.npz、キー 'f0'/'log_melspc'/'wav')。
train_v3.py は test_dir の npz をこの形式で読む(TensorBoard の real/fake 記録)ので、検証用はこちらで作る。

Usage:
    python preprocess_nnsvs.py --indir <学習用 wav_dir ...> --out dataset_nnsvs/train --config config_nnsvs_48k.yaml
    python preprocess_nnsvs.py --indir <検証用 wav_dir ...> --out dataset_nnsvs/test  --config config_nnsvs_48k.yaml --segs_per_shard 0
"""
import os, sys, glob, argparse
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import librosa
import pyworld
import soundfile as sf
import yaml
from scipy import interpolate, signal
from tqdm import tqdm

from tools.cut_by_phrases import detect_regions, build_segments

SILENT_MEL_MAX = -4.0    # log10-mel の無音しきい値(dataset.py の ln -9.2 と同じ)
FRAME_PERIOD = 5.0       # ms(nnsvs melf0 の frame_period。hop_size と一致している必要がある)


def logmelfilterbank(audio, sampling_rate, fft_size, hop_size, win_length, num_mels,
                     fmin, fmax, eps):
    """parallel_wavegan.bin.preprocess.logmelfilterbank(window='hann', log_base=10)と同じ計算。"""
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
                          window='hann', pad_mode='reflect')
    spc = np.abs(x_stft).T                                   # (#frames, #bins)
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=fft_size, n_mels=num_mels,
                                    fmin=fmin, fmax=fmax)
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))
    return np.log10(mel)                                     # (#frames, num_mels)


def interp1d_f0(f0):
    """nnmnkwii.preprocessing.f0.interp1d(kind='slinear')と同じ。0 の区間を線形補間する。"""
    cf0 = f0.copy()
    nz = np.where(cf0 > 0)[0]
    if len(nz) == 0:
        return cf0
    cf0[0] = cf0[nz[0]]
    cf0[-1] = cf0[nz[-1]]
    nz = np.where(cf0 > 0)[0]
    fn = interpolate.interp1d(nz, cf0[cf0 > 0], kind='slinear')
    z = np.where(cf0 <= 0)[0]
    cf0[z] = fn(z)
    return cf0


def lowpass_filter(x, fs, cutoff, N=5):
    """nnsvs.dsp.lowpass_filter と同じ(butter N 次・filtfilt のゼロ位相)。"""
    b, a = signal.butter(N, cutoff / (fs // 2), 'lowpass')
    if len(x) <= max(len(a), len(b)) * (N // 2 + 1):
        return x
    return signal.filtfilt(b, a, x)


def smoothed_continuous_lf0(lf0, fs, cutoff):
    """nnsvs.pitch.extract_smoothed_continuous_f0 と同じ(負になったら cutoff を上げてやり直す)。"""
    y = lowpass_filter(lf0, fs, cutoff)
    next_cutoff = 50
    while (y < 0).any():
        y = lowpass_filter(lf0, fs, next_cutoff)
        next_cutoff *= 2
    return y


def extract_f0(y, cfg):
    """nnsvs MelF0 の F0 部分(楽譜を使う処理を除く)。返り: (f0[Hz, 0=無声], frames)。"""
    fs = cfg['sample_rate']
    f0_min = min(cfg['f0_min'], 500)          # nnsvs: CheapTrick の segfault 回避
    f0, t = pyworld.harvest(y, fs, frame_period=FRAME_PERIOD, f0_floor=f0_min,
                            f0_ceil=cfg['f0_max'])
    f0 = np.maximum(f0, 0)
    if not (f0 > 0).any():
        return None
    ap = pyworld.d4c(y, f0, t, fs, threshold=cfg.get('d4c_threshold', 0.5))
    vuv = ap[:, 0] < 0.5                       # nnsvs(harvest のとき)と同じく 0.5 固定
    lf0 = np.zeros_like(f0)
    lf0[f0 > 0] = np.log(f0[f0 > 0])
    lf0 = interp1d_f0(lf0)
    lf0 = smoothed_continuous_lf0(lf0, int(1000 / FRAME_PERIOD), cfg.get('f0_smoothing_cutoff', 20))
    return (np.exp(lf0) * vuv).astype(np.float32)


def load_wav(path, cfg):
    """nnsvs と同じく ±1 の float64 で読み、必要なら再サンプル(上げるときは 1e-7 の雑音を足す)。"""
    x, fs = sf.read(path, dtype='float64')
    if x.ndim > 1:
        x = x.mean(axis=1)
    sr = cfg['sample_rate']
    if fs != sr:
        x = librosa.resample(x, orig_sr=fs, target_sr=sr, res_type=cfg.get('res_type', 'soxr_hq'))
        if sr > fs:
            x = x + np.random.RandomState(len(x)).randn(len(x)) * 1e-7   # nnsvs: init_seed(len(x))
    return x


def process_file(args):
    path, cfg = args
    sr, hop = cfg['sample_rate'], cfg['hop_size']
    min_frames = cfg['data_filtering']['min_frames']
    cw = cfg['cut_wavs']
    x = load_wav(path, cfg)
    regions = detect_regions(x.astype(np.float32), sr, silence_thresh_db=cw['silence_thresh'],
                             min_silence_dur=cw['min_silence_dur'])
    segs = build_segments(regions, sr, max_dur=cw['max_dur'], long_silence=cw['long_silence'],
                          pad=cw['pad'], total_samples=len(x))
    stem = os.path.splitext(os.path.basename(path))[0]
    out, nskip = [], 0
    for idx, (s, e) in enumerate(segs):
        y = np.ascontiguousarray(x[s:e])
        mel = logmelfilterbank(y, sr, cfg['fft_size'], hop, cfg['win_size'], cfg['mel_dim'],
                               cfg['mel_min'], cfg['mel_max'], cfg.get('mel_eps', 1e-10))
        if len(mel) < min_frames or mel.max() < SILENT_MEL_MAX:
            nskip += 1; continue
        f0 = extract_f0(y, cfg)
        if f0 is None:
            nskip += 1; continue
        T = min(len(mel), len(f0))
        if T < min_frames:
            nskip += 1; continue
        wav = y[:T * hop]
        if len(wav) < T * hop:
            wav = np.pad(wav, (0, T * hop - len(wav)))
        out.append((f'{stem}_{idx:04d}', f0[:T], mel[:T].astype(np.float32), wav.astype(np.float32)))
    return path, out, nskip


def main():
    ap = argparse.ArgumentParser(description='NHVSing preprocess for nnsvs melf0 (harvest F0, log10 mel)')
    ap.add_argument('--indir', required=True, nargs='+', help='wav の入ったディレクトリ(再帰探索。複数可)')
    ap.add_argument('--out', required=True, help='shard npz の出力先')
    ap.add_argument('--config', default='config_nnsvs_48k.yaml')
    ap.add_argument('--exclude', nargs='*', default=['.lbp.caches'],
                    help='パスにこの文字列を含む wav は使わない(既定: vLabeler のキャッシュ)')
    ap.add_argument('--segs_per_shard', type=int, default=500,
                    help='1 shard あたりの segment 数。0 で 1 segment = 1 ファイル(検証用)')
    ap.add_argument('--num_workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['preprocess']
    assert cfg.get('mel_format') == 'nnsvs', 'config の preprocess.mel_format が nnsvs ではありません'
    assert cfg['hop_size'] == round(cfg['sample_rate'] * FRAME_PERIOD / 1000), \
        f"hop_size は {FRAME_PERIOD}ms(= sample_rate * 0.005)にすること"
    os.makedirs(args.out, exist_ok=True)

    wavs = []
    for d in args.indir:
        wavs += [p for p in glob.glob(os.path.join(d, '**', '*.wav'), recursive=True)
                 if not any(ex in p for ex in args.exclude)]
    wavs = sorted(set(wavs))
    print(f"{len(wavs)} wav files -> {args.out} ({cfg['sample_rate']}Hz hop{cfg['hop_size']} "
          f"{cfg['mel_dim']}mel {cfg['mel_min']:g}-{cfg['mel_max']:g}Hz log10, "
          f"harvest {cfg['f0_min']:g}-{cfg['f0_max']:g}Hz, workers={args.num_workers})")

    shard, shard_i, nseg, nskip = {}, 0, 0, 0

    def flush():
        nonlocal shard, shard_i
        if shard:
            np.savez_compressed(os.path.join(args.out, f'shard-{shard_i:04d}.npz'), **shard)
            shard = {}; shard_i += 1

    seen = set()
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for path, segs, ns in tqdm(ex.map(process_file, [(p, cfg) for p in wavs]), total=len(wavs)):
            nskip += ns
            for sid, f0, mel, wav in segs:
                if sid in seen:                     # 別フォルダに同名 wav があるとき
                    sid = f'{os.path.basename(os.path.dirname(path))}_{sid}'
                seen.add(sid)
                if args.segs_per_shard <= 0:
                    np.savez_compressed(os.path.join(args.out, f'{sid}.npz'),
                                        f0=f0, log_melspc=mel, wav=wav)
                    nseg += 1
                    continue
                shard[f'{sid}|f0'] = f0
                shard[f'{sid}|log_melspc'] = mel
                shard[f'{sid}|wav'] = wav
                nseg += 1
                if nseg % args.segs_per_shard == 0:
                    flush()
    flush()
    print(f'done: {nseg} segments, {nskip} skipped -> {args.out}')


if __name__ == '__main__':
    main()
