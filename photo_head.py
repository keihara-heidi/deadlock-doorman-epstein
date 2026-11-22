"""Fit the licensed template to the supplied portrait; no Meshy asset is used.
Usage: python photo_head.py PORTRAIT.png TEMPLATE.glb face_landmarker.task OUTPUT_DIRECTORY
"""
import argparse
import io
import json
import struct
from pathlib import Path

import cv2
import mediapipe as mp
import moderngl as gl
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from donor import read_face


def raster(ctx, pixels, depth, attrs, indices, width, height):
    """An orthographic G-buffer in image coordinates (row zero is top)."""
    clip = np.column_stack((2*pixels[:, 0]/width-1, 2*pixels[:, 1]/height-1, -depth))
    buffer = ctx.buffer(np.column_stack((clip, attrs)).astype('<f4').tobytes())
    elements = ctx.buffer(indices.astype('<u4').tobytes())
    prog = ctx.program(vertex_shader='#version 330\nin vec3 p; in vec3 a; out vec3 v; void main(){v=a; gl_Position=vec4(p,1);}',
                       fragment_shader='#version 330\nin vec3 v; out vec4 c; void main(){c=vec4(v,1);}')
    fb = ctx.simple_framebuffer((width,height), components=4,dtype='f4')
    vao = ctx.vertex_array(prog, [(buffer,'3f 3f','p','a')],elements)
    fb.use(); fb.clear(0,0,0,0);ctx.enable(gl.DEPTH_TEST);vao.render()
    out = np.frombuffer(fb.read(components=4,dtype='f4'),'<f4').reshape(height,width,4)
    for resource in (vao,fb,buffer,elements,prog): resource.release()
    return out


def render(ctx, mesh, angle=0):
    p,n,uv,ids,texture = mesh
    rot = Rotation.from_euler('y',angle,degrees=True)
    pt = rot.apply(p-[0,1.72,0]); xy = pt[:,:2]*[1720,-1720]+[300,314]
    geom = raster(ctx,xy,pt[:,2],p,ids,600,660)
    illum = np.clip(rot.apply(n) @ [-.25,.3,.78],0,1)*.24+.84
    atlas = raster(ctx,xy,pt[:,2],np.column_stack((uv,illum)),ids,600,660)
    image = np.asarray(texture); st = atlas[:,:,:2]*[image.shape[1],image.shape[0]]-.5
    colors = cv2.remap(image,st.astype('f4'),None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    colors = np.clip(colors*atlas[:,:,2:3],0,255).astype('u1'); colors[geom[:,:,3]==0]=[33,41,41]
    return colors,geom


def points(detector,image):
    im=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(image))
    faces=detector.detect(im).face_landmarks
    if len(faces)!=1: raise ValueError(f'Expected one foreground face, detected {len(faces)}')
    return np.array([(p.x*image.shape[1],p.y*image.shape[0],p.z*image.shape[1]) for p in faces[0][:468]])


def fit_mesh(ctx,mesh,photo,model):
    pos,normal,uv,idx,texture=mesh
    rendered,geom=render(ctx,mesh)
    options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),num_faces=1)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as det:
        detected=points(det,rendered); target=points(det,photo[0:650,360:830]);target[:,0]+=360
    iy,ix=np.rint(detected[:,[1,0]]).astype(int).T
    nearest=distance_transform_edt(geom[:,:,3]==0,return_distances=False,return_indices=True)
    surface=geom[nearest[0,iy,ix],nearest[1,iy,ix],:3].astype(float)
    center=surface.mean(0)
    def view(p,params):
        r=Rotation.from_rotvec(params[:3]); q=r.apply(p-center);return q[:,:2]*[params[3],-params[3]]+params[4:],q[:,2]
    opt=least_squares(lambda params:(view(surface,params)[0]-target[:,:2]).ravel(),[0,0,.25,1550,595,350],loss='soft_l1',f_scale=5)
    r=Rotation.from_rotvec(opt.x[:3]);scale=opt.x[3];proj,depth=view(surface,opt.x)
    shift=(target[:,:2]-proj)/[scale,-scale]
    # The face tracker estimates relative depth, not metric depth. Give this
    # weak cue less weight than the photograph's observed x/y positions.
    mz=-target[:,2]/scale; mz-=np.median(mz-depth)
    shift3=r.inv().apply(np.column_stack((shift,.25*np.clip(mz-depth,-.020,.020))))
    # Average points sharing a raster cell (closed lips and eyelids). Pin scalp,
    # back and neck so a single image cannot distort the unobserved geometry.
    groups,gi=np.unique(np.round(surface/.004).astype(int),axis=0,return_inverse=True)
    counts=np.bincount(gi)[:,None]; s=np.zeros((len(groups),3));d=s.copy()
    np.add.at(s,gi,surface);np.add.at(d,gi,shift3);s/=counts;d/=counts
    back=pos[(pos[:,2]<-.035)|(pos[:,1]<1.615)|(pos[:,1]>1.873)]
    _,keep=np.unique(np.round(back/.028),axis=0,return_index=True);back=back[keep]
    tps=RBFInterpolator((np.vstack((s,back))-center)*10,np.vstack((d,np.zeros_like(back))),smoothing=.002)
    edited=pos+np.clip(tps((pos-center)*10),-.023,.023)
    # Recalculate normals on welded positions while retaining UV seams.
    tri=idx.reshape(-1,3); xyz=edited[tri]; fn=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);
    _,groups=np.unique(edited.round(6),axis=0,return_inverse=True); summed=np.zeros((groups.max()+1,3))
    for corner in range(3): np.add.at(summed,groups[tri[:,corner]],fn)
    normals=summed[groups]; lens=np.linalg.norm(normals,axis=1,keepdims=True); normals=np.where(lens>1e-10,normals/np.maximum(lens,1e-10),normal)
    if np.median((normals*normal).sum(axis=1))<0:normals=-normals
    # The original atlas reuses texels across separate triangles. An asymmetric
    # photograph needs unique UVs, otherwise parts of the cheek overwrite one another.
    import xatlas
    remap, tri_new, uv_new = xatlas.parametrize(edited.astype('f4'), tri.astype('u4'))
    edited, normals = edited[remap], normals[remap]
    source_uv, uv, idx = uv[remap], uv_new, tri_new.ravel()
    xy,z=view(edited,opt.x)
    # UV-space bake: reject self-occluded surfaces with a camera depth buffer.
    h,w=photo.shape[:2]; side=texture.width
    visible=raster(ctx,xy,z,np.tile(z[:,None],(1,3)),idx,w,h)[:,:,0].copy()
    facing=r.apply(normals)[:,2]; grid=raster(ctx,uv*side,np.zeros(len(uv)),np.column_stack((xy,z)),idx,side,side)
    dirs=raster(ctx,uv*side,np.zeros(len(uv)),np.tile(facing[:,None],(1,3)),idx,side,side)[:,:,0]
    st=grid[:,:,:2].copy().astype('f4'); rgb=cv2.remap(photo,st,None,cv2.INTER_LINEAR)
    front=cv2.remap(visible,st,None,cv2.INTER_NEAREST); valid=(grid[:,:,3]>0)&(abs(front-grid[:,:,2])<.004)
    # Hand-traced foreground silhouette, excluding jacket, room and bystander.
    outline=np.array([(393,213),(405,153),(434,94),(489,54),(566,25),(636,24),(685,38),(730,83),(766,133),(780,207),(779,270),(795,317),(791,378),(779,395),(769,467),(751,520),(728,559),(690,586),(650,599),(610,597),(562,571),(524,531),(506,474),(485,469),(465,432),(451,388),(424,351),(410,300)],np.int32)
    mask=np.zeros((h,w),'u1');cv2.fillPoly(mask,[outline],1);distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    fade=np.clip(cv2.remap(distance,st,None,cv2.INTER_LINEAR)/23,0,1)*np.clip((dirs+.03)/.45,0,1)*valid
    src=raster(ctx,uv*side,np.zeros(len(uv)),np.column_stack((source_uv,np.zeros(len(uv)))),idx,side,side)
    coord=(src[:,:,:2]*side-.5).copy();atlas=cv2.remap(np.asarray(texture),coord,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    mix=fade[:,:,None];new=np.rint(mix*rgb+(1-mix)*atlas).clip(0,255).astype('u1')
    outside=grid[:,:,3]==0; ds,nn=distance_transform_edt(outside,return_indices=True); pad=outside&(ds<7)
    new[pad]=new[nn[0,pad],nn[1,pad]]
    return edited.astype('<f4'),normals.astype('<f4'),uv,idx,Image.fromarray(new)


def save_glb(mesh,path):
    p,n,uv,idx,im=mesh
    # Do not ship unrelated donor clothing imagery: only store texels sampled
    # by the head (plus padding for filtering at UV seams).
    mask=np.zeros((im.height,im.width),'u1'); ts=np.rint(uv[idx].reshape(-1,3,2)*[im.width,im.height]).astype('i4')
    for t in ts: cv2.fillConvexPoly(mask,t,1)
    mask=cv2.dilate(mask,np.ones((13,13),'u1'));tex=np.asarray(im).copy();tex[mask==0]=[100,88,81]
    png=io.BytesIO(); Image.fromarray(tex).save(png,'PNG')
    views=[]; access=[]; body=bytearray()
    def view(data):
        body.extend(bytes(-len(body)%4));views.append({'buffer':0,'byteOffset':len(body),'byteLength':len(data)});body.extend(data);return len(views)-1
    for array,kind,typ in ((p,'VEC3',5126),(n,'VEC3',5126),(uv,'VEC2',5126),(idx,'SCALAR',5123)):
        a={'bufferView':view(array.astype('<u2' if typ==5123 else '<f4').tobytes()),'componentType':typ,'count':len(array),'type':kind}
        if array is p:a.update(min=p.min(0).tolist(),max=p.max(0).tolist())
        access.append(a)
    image_view=view(png.getvalue()); contentlen=len(body);body.extend(bytes(-len(body)%4))
    asset={'version':'2.0','extras':{'source':'User-supplied photograph; fitted geometry and unobserved head from SigmaGrindsetMog, https://sketchfab.com/3d-models/jeffrey-epstein-bd6c6bf543184c8da1f457fc7bf0c494 (CC BY 4.0).'}}
    j={'asset':asset,'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'mesh':0,'name':'Portrait head'}],'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1,'TEXCOORD_0':2},'indices':3,'material':0}]}],'materials':[{'pbrMetallicRoughness':{'baseColorTexture':{'index':0},'roughnessFactor':.9,'metallicFactor':0}}],'textures':[{'source':0}],'images':[{'bufferView':image_view,'mimeType':'image/png'}],'buffers':[{'byteLength':contentlen}],'bufferViews':views,'accessors':access}
    meta=json.dumps(j,separators=(',',':')).encode();meta+=b' '*(-len(meta)%4)
    path.write_bytes(struct.pack('<4sII',b'glTF',2,28+len(meta)+len(body))+struct.pack('<I4s',len(meta),b'JSON')+meta+struct.pack('<I4s',len(body),b'BIN\0')+body)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('photo','template','landmarker','output'):ap.add_argument(key,type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    ctx=gl.create_standalone_context();mesh=read_face(args.template)
    fitted=fit_mesh(ctx,mesh,np.asarray(Image.open(args.photo).convert('RGB')),args.landmarker)
    save_glb(fitted,args.output/'portrait-head.glb')
    sheet=Image.new('RGB',(900,330))
    for k,a in enumerate((0,38,83)):
        image,_=render(ctx,fitted,a);sheet.paste(Image.fromarray(image).resize((300,330),Image.Resampling.LANCZOS),(300*k,0))
    sheet.save(args.output/'turnaround.jpg',quality=94)
    print(f'{len(fitted[0]):,} vertices, {len(fitted[3])//3:,} triangles')
