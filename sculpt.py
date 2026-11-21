"""Local-space rest-pose sculpt for the pinned G-Man 2.2 mesh layout."""
import struct

import meshoptimizer
import numpy as np

# From the model's CTRL: streams for face, eyes, lids, eyebrows, and hair.
# Positions are float3 at byte 0; all following UV, joint and weight data stays intact.
COUNTS = {0: 201, 2: 96, 4: 3076, 8: 962, 10: 12402, 14: 2752, 16: 11808, 18: 850}


def reshape(vertices):
    idx = vertices[:, 2] > 2.20 / .0254  # The head material also covers both hands.
    depth, x, y = (vertices[idx, :3].astype(float) * .0254).T
    def bell(value, center, radius):
        return np.exp(-((value - center) / radius) ** 2)
    front = np.clip((depth + .048) / .073, 0, 1)
    # Round G-Man's angular jaw into the narrower chin of the reference;
    # fill the cheek plane and soften the shelf above the eyes.
    dx = x * (-.17 * bell(y, 2.28, .040) + .11 * bell(y, 2.367, .044))
    dz = front * (.010 * bell(abs(x), .061, .030) * bell(y, 2.365, .040)
                  - .012 * bell(y, 2.439, .017) * bell(abs(x), .042, .027))
    # Broader, less pointed nose and a longer, rounder tip.
    tip = bell(x, 0, .032) * bell(y, 2.36, .026) * front
    dx += x * .33 * tip
    dy = -.005 * tip - .007 * bell(y, 2.26, .028) * front
    dz -= .010 * tip
    # Break the flat, squared pompadour: lower its corners, introduce a
    # side sweep, and draw the tall frontal crest back toward the scalp.
    crown = np.clip((y - 2.47) / .092, 0, 1)
    dx += crown * (.009 - .13 * x)
    dy -= crown * (.006 + .020 * np.clip(abs(x) / .085, 0, 1) + .003 * np.sin(x * 90))
    dz -= .018 * crown * front
    vertices[idx, :3] += np.stack((dz, dx, dy), axis=1) / .0254


def sculpt_model(model):
    count = struct.unpack_from('<I', model, 12)[0]
    result = bytearray(model[:16 + count * 12])
    for i in range(count):
        entry = 16 + i * 12
        tag, rel, size = struct.unpack_from('<4sII', model, entry)
        start = entry + 4 + rel
        data = model[start:start + size]
        if i in COUNTS:
            assert tag == b'MVTX'
            vertices = meshoptimizer.decode_vertex_buffer(COUNTS[i], 56, data)
            reshape(vertices)
            data = meshoptimizer.encode_vertex_buffer(vertices)
        result.extend(bytes(-len(result) % 16))
        struct.pack_into('<II', result, entry + 4, len(result) - entry - 4, len(data))
        result.extend(data)
    struct.pack_into('<I', result, 0, len(result))
    return bytes(result)
