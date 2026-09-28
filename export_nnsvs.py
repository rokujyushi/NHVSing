"""学習のスナップショット → nnsvs の packed model 形式のボコーダー(ENUNUServer で読む)。

出力(音源フォルダにそのまま置く):
  vocoder_model.pth  : {'model': {'generator': state_dict}}。weight norm は外してある
                       (サーバーで remove_weight_norm を呼ばずに済むように)。torch.load(weights_only=True) で読める。
  vocoder_model.yaml : generator._target_ = nhvsing.model.NHVSingV3 と、その引数(vocoder_cfg / ltv_filter_cfg。
                       use_weight_norm: false)。data.mel に学習時の mel の仕様を書き、サーバーが音響モデルの
                       mel と照らし合わせられるようにする。

サーバーでの読み方(hydra の instantiate + load_state_dict)と同じ手順で書き出したファイルを読み直し、
学習時のモデル(weight norm つき)と出力が一致することを確かめてから終わる。

Usage:
    python export_nnsvs.py --config config_nnsvs_48k.yaml --ckpt snapshots_nnsvs_48k/000400epoch.pth --out exported_models/nnsvs48k
"""
import os, sys, argparse, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import yaml

from nhvsing.model import NHVSingV3, select_model_class


def build_generator_cfg(cfg):
    vc = copy.deepcopy(cfg['model']['vocoder'])
    lc = copy.deepcopy(cfg['model']['ltv_filter'])
    assert select_model_class(vc, lc) is NHVSingV3, 'NHVSingV3 の設定(ltv_filter.use_v3 など)だけ書き出せます'
    vc['use_weight_norm'] = False
    vc['excit_phase_jitter'] = False          # 学習時だけの設定(推論では使われない)
    return vc, lc


def mel_spec(cfg):
    p = cfg['preprocess']
    assert p.get('mel_format') == 'nnsvs', 'preprocess.mel_format: nnsvs の設定で学習したモデルだけ書き出せます'
    return {
        'num_mels': int(p['mel_dim']),
        'fft_size': int(p['fft_size']),
        'win_length': int(p.get('win_size', p['fft_size'])),
        'hop_size': int(p['hop_size']),
        'fmin': float(p['mel_min']),
        'fmax': float(p['mel_max']),
        'eps': float(p.get('mel_eps', 1e-10)),
        'log_base': 10,
    }


def main():
    ap = argparse.ArgumentParser(description='NHVSing → nnsvs packed-model vocoder (vocoder_model.pth / .yaml)')
    ap.add_argument('--config', required=True, help='学習に使った config(preprocess.mel_format: nnsvs)')
    ap.add_argument('--ckpt', required=True, help='train_v3.py のスナップショット(*.pth)')
    ap.add_argument('--out', required=True, help='出力ディレクトリ')
    args = ap.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    vc, lc = build_generator_cfg(cfg)
    spec = mel_spec(cfg)
    assert spec['hop_size'] == int(vc['hop_size']) and spec['num_mels'] == int(vc['in_channels'])

    # 学習時と同じ形(weight norm つき)で読み、外してから保存する
    train_model = NHVSingV3(cfg['model']['vocoder'], cfg['model']['ltv_filter'])
    snap = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    train_sd = snap['model'] if 'model' in snap else snap
    train_model.load_state_dict(train_sd)
    train_model.eval()
    epoch = snap.get('epoch') if isinstance(snap, dict) else None
    # deepcopy は TorchScript の関数を持つので使えない。同じ重みを別に読んで weight norm を外す
    gen = NHVSingV3(cfg['model']['vocoder'], cfg['model']['ltv_filter'])
    gen.load_state_dict(train_sd)
    gen.remove_weight_norm()
    state_dict = {k: v.detach().clone().contiguous() for k, v in gen.state_dict().items()}

    os.makedirs(args.out, exist_ok=True)
    pth = os.path.join(args.out, 'vocoder_model.pth')
    yml = os.path.join(args.out, 'vocoder_model.yaml')
    torch.save({'model': {'generator': state_dict}}, pth)
    out_cfg = {
        'generator': {'_target_': 'nhvsing.model.NHVSingV3', 'vocoder_cfg': vc, 'ltv_filter_cfg': lc},
        'data': {'feat_names': ['mel'], 'sample_rate': int(vc['sample_rate']),
                 'hop_size': int(vc['hop_size']), 'mel': spec},
        'nhvsing': {'source_ckpt': os.path.basename(args.ckpt), 'epoch': epoch,
                    'config': os.path.basename(args.config)},
    }
    with open(yml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(out_cfg, f, allow_unicode=True, sort_keys=False)

    # サーバーと同じ読み方で読み直し、学習時のモデルと出力を比べる
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    loaded_cfg = OmegaConf.load(yml)
    served = instantiate(loaded_cfg.generator)
    ck = torch.load(pth, map_location='cpu', weights_only=True)
    served.load_state_dict(ck['model']['generator'])
    served.eval()
    T = 200
    g = torch.Generator().manual_seed(0)
    mel = torch.randn(1, T, spec['num_mels'], generator=g) - 4.0
    cf0 = torch.full((1, 1, T), 220.0)
    uv = torch.zeros(1, 1, T)
    uv[..., 150:] = 1.0
    with torch.no_grad():
        torch.manual_seed(1); a = train_model(mel, cf0, uv)
        torch.manual_seed(1); b = served(mel, cf0, uv)
    diff = (a - b).abs().max().item()
    assert a.shape == b.shape == (1, T * spec['hop_size']), (a.shape, b.shape)
    assert diff < 1e-4, f'書き出したモデルの出力が学習時と合いません(max|diff|={diff:.3g})'
    size = os.path.getsize(pth) / 2 ** 20
    print(f'wrote {pth} ({size:.1f} MB, epoch {epoch}) and {yml}')
    print(f'reload via hydra instantiate + weights_only load: OK (max|diff| vs training model = {diff:.2e})')


if __name__ == '__main__':
    main()
