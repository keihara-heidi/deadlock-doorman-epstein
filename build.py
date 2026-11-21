# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pillow==13.0.0", "numpy==2.5.5", "scipy==1.18.1", "vpk==1.4.0", "lz4==4.4.5", "ispc-texcomp==1.0.1", "meshoptimizer==0.2.30a0"]
# ///
"""uv run build.py path/to/pak16_dir.vpk

Input: extracted Grelgn G-Man v2.2 (GameBanana 657822). Output: local-use
adaptation only, not for redistribution under the base skin's BY-NC-ND terms.
Head: SigmaGrindsetMog's CC-BY 4.0 Jeffrey Epstein model, simplified and re-rigged (see donor.py).
Voice: synthetic F5-TTS readings using separate Valve Doorman reference reels
for the normal cast, desperation cast, and the greeting inside the hotel.
"""
import argparse
import hashlib
import struct
import tempfile
import wave
from pathlib import Path, PurePosixPath

import lz4.block
import vpk
from donor import read_face, swap_model, donor_texture

ASSETS = Path(__file__).resolve().parent / 'assets'
FACE = 'models/heroes_wip/doorman_v2/gman/textures/gman_head_color_png_fcd3833f.vtex_c'
PCM_TEMPLATE = 'sounds/gman/props/button15.vsnd_c'
# All three English Hotel Guest takes say the same sentence. Preserve the
# response rules and randomization, replacing no 'Enjoy your stay' takes.
WELCOME = [f'sounds/vo/doorman/doorman_{take}.vsnd_c' for take in (
    'use_power4_02', 'desperation_power4_03', 'power4_generic_01_short')]


def pinned(base, path, digest):
    data = base[path].read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Unexpected {path}; use G-Man v2.2, not a different layout')
    return data


def build_voice(template, wav):
    """Fill a verified v5 PCM16 CTRL template; the payload has no WAV header."""
    with wave.open(str(wav), 'rb') as audio:
        if (audio.getnchannels(), audio.getsampwidth()) != (1, 2):
            raise ValueError('Voice must be mono PCM16')
        n, rate = audio.getnframes(), audio.getframerate()
        pcm = audio.readframes(n)
    # In this exact template KV3 v5 has two LZ4 buffers (strings and values).
    # Update the decoded typed fields, not compressed bytes or string matches.
    ctrl = bytearray(template[832:1270])
    fields = bytearray(lz4.block.decompress(ctrl[320:434], uncompressed_size=127))
    for pos, val in ((24, rate), (52, n), (72, len(pcm))):
        struct.pack_into('<I', fields, pos, val)
    struct.pack_into('<d', fields, 96, n / rate)
    encoded = lz4.block.compress(bytes(fields), store_size=False)
    ctrl[320:434] = encoded
    struct.pack_into('<I', ctrl, 52, 200 + len(encoded))  # total compressed bytes
    struct.pack_into('<I', ctrl, 84, len(encoded))       # second buffer size
    header = bytearray(template[:832])
    struct.pack_into('<I', header, 0, len(header) + len(ctrl))
    struct.pack_into('<I', header, 48, len(ctrl))          # CTRL directory entry
    return bytes(header) + ctrl + pcm


def build(source, output):
    if source.resolve() == output.resolve():
        raise ValueError('Refusing to overwrite the base skin')
    base = vpk.open(str(source))
    sound = pinned(base, PCM_TEMPLATE, 'a985e561f0bfcdf2d5d18c036479d5c3dfcc87def6286d03663f2494a2580013')
    donor = read_face(Path.home() / 'Downloads/jeffrey_epstein.glb')
    model = 'models/heroes_wip/doorman_v2/doorman.vmdl_c'
    hair = FACE.replace('head_color_png_fcd3833f', 'hair_color_png_99288e7f')
    overrides = {
        model: swap_model(pinned(base, model, '32f16026fbee883fc058117623abfd102dfa38d90bf15c90e6e2c91c121e99eb'), donor),
        hair: donor_texture(pinned(base, hair, 'a5c6b3c8183bd3939c8a7b4cb2dfe8930a0f653e0954f9fdd411b276cc976de7'), donor[-1]),
        hair.replace('color_png_99288e7f','normal_png_2c24dd44'): base['models/heroes_wip/doorman_v2/gman/materials/gman_eye_vmat_g_tnormalroughness_47817ce7.vtex_c'].read(),
    }
    for target, filename in zip(WELCOME, ('welcome.wav', 'welcome-low.wav', 'welcome-hotel.wav'), strict=True):
        overrides[target] = build_voice(sound, ASSETS / filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        root = Path(temp) / 'resources'
        for name in sorted(set(base) | set(overrides)):
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
                raise ValueError(f'Unsafe package path: {name}')
            dst = root.joinpath(*path.parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(overrides[name] if name in overrides else base[name].read())
        packed = Path(temp) / 'skin.vpk'
        vpk.new(str(root)).save(str(packed))
        packed.replace(output)
    print(f'{output}: head sculpt, hair, face and {len(WELCOME)} welcome takes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_vpk', type=Path)
    parser.add_argument('--output', type=Path, default=Path('build/Doorman-Island.vpk'))
    args = parser.parse_args()
    build(args.base_vpk, args.output)
