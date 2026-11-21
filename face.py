"""Face-only paint layer for Grelgn's G-Man 2.2, in 512px atlas coordinates."""
import ispc_texcomp as bc
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import distance_transform_edt, map_coordinates

# Each row: target atlas x,y -> source portrait x,y. The outside ring fades to
# the original temples/jaw; the eyes and mouth must follow the animated mesh.
LANDMARKS = [
    (173,175,125,114), (214,168,180,99), (256,168,249,97), (300,168,317,101), (339,175,367,125),
    (341,220,387,204), (363,268,405,279), (377,302,414,351), (374,352,405,426),
    (346,394,378,506), (308,425,326,566), (256,439,248,601), (203,425,171,561),
    (163,393,133,516), (139,354,109,439), (134,300,99,356), (151,268,102,291), (172,222,110,205),
    # Brow tails, peaks, inner ends and glabella.
    (187,246,110,277), (210,232,151,248), (243,237,202,251), (256,241,233,265),
    (268,237,269,251), (297,233,312,246), (325,247,369,272),
    # Lid corners and lower-lid creases; eyeballs use a separate material.
    (193,266,123,302), (211,259,165,289), (228,259,200,284), (211,272,166,310),
    (282,259,276,283), (299,259,309,286), (317,267,358,295), (299,272,308,306),
    (206,283,162,325), (305,282,319,321),
    # Bridge, nose tip, nostrils, cheek folds and philtrum.
    (255,270,232,299), (255,303,242,365), (240,318,208,379), (270,318,279,378),
    (256,325,244,394), (225,313,194,372), (284,313,295,371),
    (206,347,151,429), (303,348,340,431), (256,339,242,428),
    # Lip edge, corners, cupid's bow and chin pad.
    (219,360,162,447), (237,353,200,443), (249,349,225,438), (257,351,240,441),
    (265,350,253,438), (276,354,283,447), (293,360,323,454), (256,359,241,455),
    (255,368,241,472), (256,384,242,507), (255,414,246,565),
]


def paint_face(atlas, portrait):
    if portrait.size != (504, 675):
        raise ValueError('Landmarks require the 504 × 675 Florida 2013 portrait')
    points = np.array(LANDMARKS)
    y, x = np.mgrid[:512, :512]
    xy = LinearNDInterpolator(points[:, :2], points[:, 2:])(x, y)
    inside = np.isfinite(xy[..., 0])
    coords = np.nan_to_num(xy)[..., ::-1].transpose(2, 0, 1)
    photo = np.asarray(portrait.convert('RGB'), dtype=float)
    warped = np.stack([map_coordinates(channel, coords, order=1) for channel in photo.transpose(2, 0, 1)], axis=-1)
    # Reduce flash colour cast, retaining the brow furrows and cheek folds.
    rgb = warped * .9 + warped.mean(axis=2, keepdims=True) * .1
    rgb = rgb * .90 + 2
    fade = np.clip(distance_transform_edt(inside) / 16, 0, 1)[..., None]
    weight = fade * fade * (3 - 2 * fade)
    rgba = np.array(atlas)
    # The head atlas includes hands, ears and neck. Only lift the dark scalp
    # between the upper islands and the hairline, excluding the eye sockets.
    level = rgba[..., :3].astype(float).mean(axis=2)
    scalp = (y > 55) & (y < 369) & ((y < 220) | (abs(x - 256) > 97))
    mix = (scalp * np.clip((89 - level) / 23, 0, 1))[..., None]
    silver = np.clip(3.0 * level[..., None] + np.array([13, 11, 8]), 0, 222)
    rgba[..., :3] = rgba[..., :3] * (1 - mix) + silver * mix
    rgba[..., :3] = np.rint(rgb * weight + rgba[..., :3] * (1 - weight)).clip(0, 255)
    return Image.fromarray(rgba)


def build_texture(template, portrait=None, *, kind='face', size=512):
    # Pinned templates have raw square BC7 mips and a 32² BC3 fallback.
    atlas = Image.frombytes('RGBA', (size, size), template[-size**2:], 'bcn', 7)
    if kind == 'face':
        painted = paint_face(atlas, portrait)
    else:
        rgba = np.array(atlas)
        level = rgba[..., :3].astype(float).mean(axis=2)
        if kind == 'hair':
            # Vary the salt-and-pepper strands without erasing their cutouts.
            rgba[..., :3] = np.clip(level[..., None] * 3.3 + [1, 0, -3], 0, 223)
        elif kind == 'brow':
            # The portrait already carries his brows. G-Man's dense black
            # eyebrow cards otherwise draw a second brow over each eyelid.
            rgba[..., :3] = np.clip(level[..., None] + [48, 41, 35], 0, 255)
            rgba[..., 3] = np.rint(rgba[..., 3] * .18)
        else:
            raise ValueError(kind)
        painted = Image.fromarray(rgba)
    stream = bytearray(template)
    settings = bc.BC7EncSettings.from_profile('alpha_slow')
    side = size
    header_size = int.from_bytes(template[:4], 'little')
    offset = header_size
    for size in (1 << n for n in range(2, side.bit_length())):
        # Independent channels: alpha is a shader mask, not transparency.
        mip = painted.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        mip.putalpha(painted.getchannel('A').resize((size, size), Image.Resampling.BOX))
        surface = bc.RGBASurface(mip.tobytes(), size, size)
        if size == 32:
            stream[header_size - 1024:header_size] = bc.compress_blocks_bc3(surface)
        encoded = bc.compress_blocks_bc7(surface, settings)
        if kind == 'face' and size == side:
            # Keep original compression bit-for-bit on untouched atlas islands.
            blocks = np.frombuffer(encoded, np.uint8).reshape(128, 128, 16)
            original = np.frombuffer(template, np.uint8, offset=offset).reshape(128, 128, 16)
            changed = (np.asarray(painted) != np.asarray(atlas)).reshape(128, 4, 128, 4, 4).any(axis=(1, 3, 4))
            encoded = np.where(changed[..., None], blocks, original).tobytes()
        stream[offset:offset + len(encoded)] = encoded
        offset += len(encoded)
    return bytes(stream)
