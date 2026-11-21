# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=12,<14"]
# ///
"""uv run viewer/export.py /path/to/Source2Viewer-CLI [private-skin.vpk]"""
import argparse
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('cli')
parser.add_argument('vpk', nargs='?', type=Path, default=here.parent / 'build/Doorman-Island.vpk')
args = parser.parse_args()
out = here / 'public/model'
out.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as temp:
    glb = Path(temp) / 'skin.glb'
    subprocess.run([args.cli, '-i', str(args.vpk), '-f', 'models/heroes_wip/doorman_v2/doorman.vmdl_c',
                    '--gltf_export_format', 'glb', '--gltf_mesh_list', 'gman', '--gltf_export_materials',
                    '-o', str(glb)], check=True)
    with glb.open('rb') as f:
        f.seek(12)
        size, kind = struct.unpack('<I4s', f.read(8))
        assert kind == b'JSON'
        manifest = json.loads(f.read(size))
    # Browser image decoders may premultiply low alpha away. On opaque Source 2
    # albedos that channel is a shader mask; only retain alpha on cutout cards.
    opaque = {manifest['textures'][m['pbrMetallicRoughness']['baseColorTexture']['index']]['source']
              for m in manifest['materials'] if m.get('alphaMode', 'OPAQUE') == 'OPAQUE'}
    for index, image in enumerate(manifest['images']):
        name = image['uri']
        if index in opaque:
            with Image.open(glb.parent / name) as src:
                src.convert('RGB').save(out / name)
        else:
            shutil.copyfile(glb.parent / name, out / name)
    shutil.copyfile(glb, out / glb.name)
for name in ('welcome.wav', 'welcome-low.wav', 'welcome-hotel.wav'):
    shutil.copyfile(here.parent / 'assets' / name, out / name)
downloads = here / 'public/downloads'
downloads.mkdir(exist_ok=True)
for name in ('Doorman-Island.vpk', 'Doorman-Island-Voice-Only.vpk'):
    shutil.copyfile(here.parent / 'build' / name, downloads / name)
