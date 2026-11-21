"""Fit Face_Texture_0 from SigmaGrindsetMog's CC-BY 4.0 GLB to Grelgn's rig.
Source: https://sketchfab.com/3d-models/jeffrey-epstein-bd6c6bf543184c8da1f457fc7bf0c494
"""
import hashlib
import io
import json
import struct

import ispc_texcomp as bc
import meshoptimizer as mo
import numpy as np
from PIL import Image


def read_face(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '137fee38c9d839a39b1d045965ca80cadc97cf24741b530b5bea7bd478c0c46d':
        raise ValueError('Expected the SigmaGrindsetMog Sketchfab GLB')
    size = struct.unpack_from('<I', raw, 12)[0]
    j = json.loads(raw[20:20+size]); buf = memoryview(raw)[28+size:]
    p = next(m for m in j['meshes'] if m['name'] == 'Face_Texture_0')['primitives'][0]
    def array(i):
        a = j['accessors'][i]; v = j['bufferViews'][a['bufferView']]
        n = {'VEC2':2,'VEC3':3,'SCALAR':1}[a['type']]
        return np.ndarray((a['count'],n), {5126:'<f4',5125:'<u4'}[a['componentType']], buf,
                          v.get('byteOffset',0)+a.get('byteOffset',0), (v.get('byteStride',4*n),4))
    v = j['bufferViews'][j['images'][0]['bufferView']]; start = v['byteOffset']
    image = Image.open(io.BytesIO(buf[start:start+v['byteLength']])).convert('RGB')
    pos, normal, uv = (array(p['attributes'][k]) for k in ('POSITION','NORMAL','TEXCOORD_0'))
    indices = array(p['indices']).ravel(); reduced = np.empty_like(indices)
    count = mo.simplify(reduced, indices, pos, target_index_count=33948, target_error=.1)
    if count > 33948: raise ValueError('Donor exceeds target draw budget')
    used, idx = np.unique(reduced[:count],return_inverse=True)
    if len(used)>11808: raise ValueError('Too many vertices')
    return pos[used], normal[used], uv[used], idx.astype('u4'), image


def swap_model(model, donor):
    pos, normal, uv, idx, _ = donor
    count = struct.unpack_from('<I',model,12)[0]; blocks = []
    for i in range(count):
        tag,rel,size = struct.unpack_from('<4sII',model,16+12*i); start = 20+12*i+rel
        blocks.append((tag,model[start:start+size]))
    # Original face material includes the hands. Remove only head triangles.
    old = mo.decode_vertex_buffer(12402,56,blocks[10][1]); ids = mo.decode_index_buffer(70530,4,blocks[11][1]).reshape(-1,3)
    ids[(old[ids,2] * .0254 > 2.12).any(axis=1)] = 0
    blocks[11] = (b'MIDX',mo.encode_index_buffer(ids.ravel()))
    for i,n in ((1,756),(3,360),(5,17856),(9,2016),(15,6192),(19,1950)):
        # Discover the count for the last brows stream from its index codec by
        # keeping its buffer intact and collapsing vertices instead.
        if i == 19:
            brow = mo.decode_vertex_buffer(850,56,blocks[18][1]); brow[:,:3]=0
            blocks[18]=(b'MVTX',mo.encode_vertex_buffer(brow)); continue
        blocks[i]=(b'MIDX',mo.encode_index_buffer(np.zeros(n,'u4'),vertex_count=1))
    old_hair = mo.decode_vertex_buffer(11808,56,blocks[16][1]); raw = old_hair.view('u1')
    template = old_hair[np.flatnonzero((raw[:,48]==43)&(raw[:,52]==255))[0]]
    verts = np.tile(template,(11808,1)); n = len(pos)
    xyz = pos * 1.12 + [0, 2.56-1.89679*1.12, -.063]
    verts[:n,:3] = xyz[:,[2,0,1]]/.0254
    verts[:n].view('<i2')[:,8:10] = np.rint(uv * 32767)
    verts[:n].view('<i2')[:,10:12] = np.rint(uv * 32767)
    octa = normal[:,[2,0,1]] / np.maximum(abs(normal).sum(axis=1)[:,None],1e-10)
    xy = octa[:,:2].copy(); neg = octa[:,2]<0
    xy[neg] = (1-abs(xy[neg,::-1])) * np.where(xy[neg]>=0,1,-1)
    x,y = np.rint((np.clip(xy,-1,1)+1)*511.5).astype('u4').T
    verts[:n].view('<u4')[:,6] = (x<<12)|(y<<22)|1
    blocks[16]=(b'MVTX',mo.encode_vertex_buffer(verts)); blocks[17]=(b'MIDX',mo.encode_index_buffer(np.pad(idx,(0,33948-len(idx))),vertex_count=11808))
    out = bytearray(model[:16+count*12]);
    for i,(tag,data) in enumerate(blocks):
        out.extend(bytes(-len(out)%16)); struct.pack_into('<4sII',out,16+12*i,tag,len(out)-20-12*i,len(data));out.extend(data)
    struct.pack_into('<I',out,0,len(out));return bytes(out)


def donor_texture(template, image):
    header_size = struct.unpack_from('<I',template)[0]; out = bytearray(template[:header_size]); data = 1120
    # Keep the artist's atlas at its native resolution instead of shrinking
    # the face into a small patch of the old 512px hair texture.
    side = image.width; assert image.size == (2048,2048)
    struct.pack_into('<HH',out,data+20,side,side);out[data+27]=10
    settings=bc.BC7EncSettings.from_profile('basic')
    for size in (1<<n for n in range(2,12)):
        mip=image.resize((size,size),Image.Resampling.LANCZOS).convert('RGBA'); surface=bc.RGBASurface(mip.tobytes(),size,size)
        if size==32:out[header_size-1024:header_size]=bc.compress_blocks_bc3(surface)
        out.extend(bc.compress_blocks_bc7(surface,settings))
    return bytes(out)
