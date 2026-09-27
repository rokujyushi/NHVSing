"""nnsvs melf0 (48kHz / hop240 / 80-mel) で NHVSingV3 が破綻しないかの確認(pytest 不要・plain python)。

config_v3_2.yaml を元に sample_rate / hop_size / in_channels だけを差し替えた乱数初期化モデルで:
  - forward の出力長が T * 240、有限値(Hann / square OLA とも)
  - ccep=0(恒等フィルタ)のとき Hann WOLA が励起をそのまま通す(hop240 で COLA が成り立つ)
  - 長尺(frame_block=256 フレーム・励起 time_block=32768 サンプルを越える)でブロック処理が一括とビット一致
  - 48kHz の励起が F0 の周期(48000/f0 サンプル)で繰り返す
  - forward_train の backward が通り、勾配が有限
  - remove_weight_norm 後の state_dict round-trip(strict)
  - 同じ seed なら同じ波形(雑音は torch.normal)

Usage: python test_hop240.py
"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import yaml

from nhvsing.model import NHVSingV3
from nhvsing.dsp import generate_impulse_train

HERE = os.path.dirname(os.path.abspath(__file__))
SR, HOP, MEL = 48000, 240, 80
_pass = _fail = 0


def _ok(name, cond, extra=''):
    global _pass, _fail
    print(f'  [{"PASS" if cond else "FAIL"}] {name} {extra}')
    _pass += int(bool(cond)); _fail += int(not cond)


def _cfg(ola_mode='hann'):
    with open(os.path.join(HERE, 'config_v3_2.yaml'), encoding='utf-8') as f:
        m = yaml.safe_load(f)['model']
    vc, lc = copy.deepcopy(m['vocoder']), copy.deepcopy(m['ltv_filter'])
    vc.update(sample_rate=SR, hop_size=HOP, in_channels=MEL)
    lc.update(hop_size=HOP, in_channels=MEL, ola_mode=ola_mode)
    return vc, lc


def _inputs(T, B=1, f0=220.0):
    mel = torch.randn(B, T, MEL) * 0.5 - 2.0            # log10-mel 程度の値
    cf0 = torch.full((B, 1, T), f0)
    uv = torch.zeros(B, 1, T)
    uv[..., : T // 10] = 1.0                           # 先頭 1 割を無声に
    return mel, cf0, uv


def _build(ola_mode='hann', **ltv_over):
    vc, lc = _cfg(ola_mode)
    lc.update(ltv_over)
    torch.manual_seed(0)
    return NHVSingV3(vc, lc).eval()


@torch.no_grad()
def check_shapes():
    print('[出力長・有限値]')
    for mode in ('hann', 'square'):
        model = _build(mode)
        for T in (1, 2, 7, 64, 300):
            y = model(*_inputs(T, B=2))
            _ok(f'{mode} T={T}', y.shape == (2, T * HOP) and torch.isfinite(y).all(),
                f'shape={tuple(y.shape)} exp=(2,{T * HOP})')


@torch.no_grad()
def check_identity_filter():
    """ccep=0 → IR は δ → Hann WOLA は励起を素通しするはず(hop240 で COLA)。

    ただし末尾の 1 hop は後ろに重なるフレームが無いので、窓の下り半分で 0 へフェードする。
    これは hop256 でも同じ(元の設計)なので、hop256 と同じ形になっていることを確かめる。
    """
    print('[恒等フィルタの素通し(Hann WOLA)]')
    from nhvsing.dsp import complex_cepstrum_to_imp, hann_ltv_fir
    T = 50
    x = torch.randn(1, 1, T * HOP)
    imp = complex_cepstrum_to_imp(torch.zeros(1, T, 256), 1024)
    y = hann_ltv_fir(x, imp, HOP)
    err = (y - x)[..., :-HOP].abs().max().item()
    _ok('hann_ltv_fir(x, δ) == x(末尾 1 hop 以外)', y.shape == x.shape and err < 1e-5,
        f'max|diff|={err:.2e}')

    def tail_gain(hop):
        ones = torch.ones(1, 1, 8 * hop)
        g = hann_ltv_fir(ones, complex_cepstrum_to_imp(torch.zeros(1, 8, 256), 1024), hop)
        return g[0, 0, -hop:]
    g240 = tail_gain(HOP)
    ref = torch.nn.functional.interpolate(tail_gain(256).reshape(1, 1, -1), size=HOP,
                                          mode='linear', align_corners=True).flatten()
    d = (g240 - ref).abs().max().item()
    _ok('末尾 1 hop のフェードが hop256 と同じ形', d < 0.02 and g240[-1].abs() < 1e-3,
        f'max|diff|={d:.3f} 末尾ゲイン={g240[-1].item():.3f}')


@torch.no_grad()
def check_blocked_equals_bulk():
    """長尺でのブロック処理(既定)と一括処理(frame_block=0 / time_block=全長)がビット一致。"""
    print('[ブロック処理 == 一括(長尺)]')
    T = 600                                            # 600 フレーム > 256、144000 サンプル > 32768
    inp = _inputs(T)
    blocked, bulk = _build(frame_block=256), _build(frame_block=0)
    bulk.load_state_dict(blocked.state_dict())
    torch.manual_seed(1); yb = blocked(*inp)
    torch.manual_seed(1); ya = bulk(*inp)
    d = (yb - ya).abs().max().item()
    _ok('frame_block=256 vs 0', d == 0.0, f'max|diff|={d:.2e}')

    f0 = torch.nn.functional.interpolate(inp[1], scale_factor=HOP, mode='linear', align_corners=False)
    eb = generate_impulse_train(f0, 200, float(SR))
    ea = generate_impulse_train(f0, 200, float(SR), time_block=0)
    d = (eb - ea).abs().max().item()
    _ok('励起 time_block=32768 vs 一括', d == 0.0, f'max|diff|={d:.2e}')


@torch.no_grad()
def check_excitation_period():
    print('[48kHz 励起の周期]')
    for f0 in (100.0, 200.0, 400.0, 1000.0):
        period = SR / f0                               # 480 / 240 / 120 / 48 サンプル
        e = generate_impulse_train(torch.full((1, 1, SR // 2), f0), 200, float(SR))[0, 0]
        e = e[SR // 4:]                                # 立ち上がりを避ける
        peaks = (e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]) & (e[1:-1] > 0.5 * e.max())
        idx = torch.nonzero(peaks).flatten().float()
        gap = (idx[1:] - idx[:-1]).mean().item() if idx.numel() > 1 else float('nan')
        _ok(f'f0={f0:g}Hz', abs(gap - period) < 0.5, f'peak間隔={gap:.2f} exp={period:g}')


def check_backward():
    print('[forward_train の backward(学習時の形)]')
    model = _build().train()
    T = 64                                             # 学習クロップ相当
    mel, cf0, uv = _inputs(T, B=2)
    wav, harm, noise = model.forward_train(mel, cf0, uv)
    wav.pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    ok = all(g is not None and torch.isfinite(g).all() for g in grads)
    _ok('grad 有限', ok and wav.shape == (2, T * HOP), f'wav={tuple(wav.shape)} params={len(grads)}')


@torch.no_grad()
def check_roundtrip_and_determinism():
    print('[weight norm 除去・round-trip・決定性]')
    model = _build()
    inp = _inputs(100)
    torch.manual_seed(3); y0 = model(*inp)
    model.remove_weight_norm()
    torch.manual_seed(3); y1 = model(*inp)
    d = (y0 - y1).abs().max().item()
    _ok('remove_weight_norm で出力ほぼ不変', d < 1e-4, f'max|diff|={d:.2e}')

    vc, lc = _cfg()
    vc['use_weight_norm'] = False
    plain = NHVSingV3(vc, lc).eval()
    res = plain.load_state_dict(model.state_dict(), strict=True)
    _ok('use_weight_norm=false へ strict ロード', not res.missing_keys and not res.unexpected_keys)
    torch.manual_seed(3); y2 = plain(*inp)
    _ok('同じ seed で同じ波形', torch.equal(y1, y2))
    torch.manual_seed(4); y3 = plain(*inp)
    _ok('seed が違えば雑音が変わる', not torch.equal(y2, y3))


def main():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    check_shapes()
    check_identity_filter()
    check_blocked_equals_bulk()
    check_excitation_period()
    check_backward()
    check_roundtrip_and_determinism()
    print(f'\n=== {_pass} passed, {_fail} failed ===')
    sys.exit(1 if _fail else 0)


if __name__ == '__main__':
    main()
