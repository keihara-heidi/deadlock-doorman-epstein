"""Seat the scratch head in the G-Man 2.2 shirt opening without resampling the face.
Usage: python fit_collar.py stock-pak16_dir.vpk scratch-head-dir output-dir
"""
import argparse
import io
import json
import struct
import tempfile
from pathlib import Path

import meshoptimizer as mo
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import vpk

from scratch_head import PROFILE, fade, write_glb

MODEL = 'models/heroes_wip/doorman_v2/doorman.vmdl_c'
OFFSET = np.array([0, 2.56 - 1.89679 * 1.12, -.063])
# Bind-space measurements (metres). Move the entire skull over the opening;
# pulling only the bottom ring back leaves a horizontal shelf under the hair.
HEAD_SHIFT = np.array([0., 0., -.056])
COLLAR_CENTER = np.array([0., 2.122, -.094])
COLLAR_RADII = (.066, .059)


def fit(p, uv):
    row = 37 + 707 * uv[:, 1] / .75
    a = 2 * np.pi * uv[:, 0] - np.pi
    c = np.cos(a)
    start = 480 + 138 * (c + 1) / 2
    t = np.clip((row - start) / (744 - start), 0, 1)
    t[uv[:, 1] > .75001] = 0  # ears and locks use separate texture islands
    world = p * 1.12 + OFFSET + HEAD_SHIFT

    # Loft from the actual tilted jaw/nape boundary to a level ring inside the
    # collar. Interpolating height independently keeps the rear rows from piling
    # up on one plane where the original sculpt turns the mandibular corner.
    rx, front, rear = PROFILE(start).T
    y = start - 83 * fade(449, 617, start) * (1-c) * .64
    anchor = np.column_stack((rx*np.sin(a), y-37, (front+rear+(front-rear)*c)/2))
    anchor = (anchor * [.000515, -.000515, .000515] + [0, 1.896, .075]) * 1.12 + OFFSET + HEAD_SHIFT
    bottom = np.tile(COLLAR_CENTER, (len(p), 1))
    bottom[:,0] += COLLAR_RADII[0] * np.sin(a)
    bottom[:,2] += COLLAR_RADII[1] * c
    cap = (row > 743.8) & (abs(p[:,0]) < 1e-5) & (abs(p[:,2] - .0323) < .002)
    bottom[cap] = COLLAR_CENTER
    loft = anchor + (bottom-anchor)*fade(0, 1, t)[:,None]
    loft[:,1] = anchor[:,1] + (bottom[:,1]-anchor[:,1])*t
    shift = (loft-world)*fade(0, .16, t)[:,None] + HEAD_SHIFT
    return (p + shift / 1.12).astype(p.dtype), fade(0, .72, t)


def read_glb(path):
    b = path.read_bytes(); n = struct.unpack_from('<I', b, 12)[0]
    j = json.loads(b[20:20+n]); raw = memoryview(b)[28+n:]
    def view(i):
        v = j['bufferViews'][i]; s = v.get('byteOffset',0); return raw[s:s+v['byteLength']]
    def acc(i, dtype, k): return np.frombuffer(view(j['accessors'][i]['bufferView']),dtype).reshape(-1,k)
    pr = j['meshes'][0]['primitives'][0]; att = pr['attributes']
    p,n,uv = (acc(att[k],'<f4',w) for k,w in [('POSITION',3),('NORMAL',3),('TEXCOORD_0',2)])
    iid = pr['indices']; indices = acc(iid,{5123:'<u2',5125:'<u4'}[j['accessors'][iid]['componentType']],1).ravel().astype('u4')
    image = Image.open(io.BytesIO(view(j['images'][0]['bufferView']))).convert('RGB')
    return p,n,uv,indices,image


def refit(mesh):
    p,n,uv,idx,image = mesh
    q,blend = fit(p,uv); tri = idx.reshape(-1,3)
    # Preserve existing normals except at the refitted rings; weld the UV seam.
    _,g = np.unique(q.round(7),axis=0,return_inverse=True); sums = np.zeros((g.max()+1,3))
    d = q[tri]; fn = np.cross(d[:,1]-d[:,0],d[:,2]-d[:,0]);
    for k in range(3): np.add.at(sums,g[tri[:,k]],fn)
    new = n.copy(); normal = sums[g]; length = np.linalg.norm(normal,axis=1); active=(blend>0)&(length>1e-9)
    new[active] = normal[active]/length[active,None]
    return (q,new,uv,idx,image),blend


def get_blocks(resource):
    blocks=[]
    for i in range(struct.unpack_from('<I',resource,12)[0]):
        offset=16+12*i; tag,rel,size=struct.unpack_from('<4sII',resource,offset); start=offset+4+rel
        blocks.append((tag,resource[start:start+size]))
    return blocks


def patch_rig(current, original, mesh, blend):
    blocks=get_blocks(current); stock=get_blocks(original)
    p,n,uv,idx,_=mesh; assert idx.max()<11808 and blocks[16][0]==b'MVTX'
    verts=mo.decode_vertex_buffer(11808,56,blocks[16][1]); data=verts.view('u1');active=np.flatnonzero(blend>0)
    all_world = p * 1.12 + OFFSET
    verts[:len(p),:3] = all_world[:,[2,0,1]]/.0254
    world = all_world[active]
    octa=n[active][:,[2,0,1]]/np.maximum(abs(n[active]).sum(axis=1)[:,None],1e-10)
    xy=octa[:,:2].copy(); neg=octa[:,2]<0;xy[neg]=(1-abs(xy[neg,::-1]))*np.where(xy[neg]>=0,1,-1)
    x,y=np.rint((xy.clip(-1,1)+1)*511.5).astype('u4').T;verts.view('<u4')[active,6]=(x<<12)|(y<<22)|1
    # Both streams belong to the same embedded gman mesh / joint palette.
    # Sample the stock skinned neck, not a hair card or the nearby tie.
    donor=mo.decode_vertex_buffer(12402,56,stock[10][1]); pos=donor[:,:3][:,[1,2,0]]*.0254
    mask=(abs(pos[:,0])<.14)&(pos[:,1]>2.12)&(pos[:,1]<2.31)
    neck=donor[mask].copy().view('u1'); tree=cKDTree(pos[mask]); ds,nn=tree.query(world,k=6)
    influence=1/np.maximum(ds,.002)**2; influence/=influence.sum(axis=1)[:,None]
    for i,vertex in enumerate(active):
        weights=np.bincount(neck[nn[i],48:52].ravel(),weights=(neck[nn[i],52:56]*influence[i,:,None]*blend[vertex]).ravel(),minlength=256)
        weights[43]+=255*(1-blend[vertex]); joints=np.argsort(-weights)[:4]; w=weights[joints]; w=np.rint(255*w/w.sum()).astype(int);w[0]+=255-w.sum()
        data[vertex,48:52]=np.where(w>0,joints,0);data[vertex,52:56]=w
    blocks[16]=(b'MVTX',mo.encode_vertex_buffer(verts));out=bytearray(current[:16+12*len(blocks)])
    for i,(tag,b) in enumerate(blocks):
        out.extend(bytes(-len(out)%16));struct.pack_into('<4sII',out,16+12*i,tag,len(out)-20-12*i,len(b));out.extend(b)
    struct.pack_into('<I',out,0,len(out));return bytes(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stock',type=Path);ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for name in ('original-head.glb','head-game.glb'):
        mesh,blend=refit(read_glb(args.source/name));write_glb(mesh,args.output/name)
    base=vpk.open(str(args.source/'Doorman-Original-Head.vpk'));stock=vpk.open(str(args.stock));patched=patch_rig(base[MODEL].read(),stock[MODEL].read(),mesh,blend)
    with tempfile.TemporaryDirectory() as tmp:
        for path in base:
            dest=Path(tmp)/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(patched if path==MODEL else base[path].read())
        vpk.new(tmp).save(str(args.output/'Doorman-Collar-Fit.vpk'))
