"""Build new portrait geometry from a single photo, not a template mesh.
Input: the supplied 1200x900 portrait and MediaPipe's face-landmark detector.
The camera-facing landmarks constrain facial relief. Off-camera skull, ears,
neck and hair flow are artist-defined; they are not measured by the photograph.
"""
import argparse
import io
import json
import struct
from pathlib import Path

import cv2
import mediapipe as mp
import meshoptimizer as mo
import moderngl as gl
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator, RBFInterpolator

from photo_head import points, render

# Cross sections in the levelled portrait's pixel space:
# height from crown, half-width, foremost depth, rearmost depth.
SECTIONS=np.array([
    (37,2,-103,-107),(42,38,-62,-156),(55,82,-26,-191),
    (78,117,8,-225),(110,150,36,-254),(158,172,58,-274),
    (213,181,69,-285),(267,181,76,-283),(313,175,62,-273),
    (355,164,65,-253),(399,151,67,-229),(440,145,64,-202),
    (476,143,64,-183),(516,135,62,-165),(552,121,61,-151),
    (584,102,60,-146),(606,86,51,-144),(620,77,32,-145),
    (636,84,2,-149),(656,94,-12,-153),(701,100,-13,-158),
    (744,100,-6,-160)],float)
PROFILE=PchipInterpolator(SECTIONS[:,0],SECTIONS[:,1:],axis=0)

def fade(a,b,t):
    q=np.clip((t-a)/(b-a),0,1);return q*q*(3-2*q)

def hairline(a):
    return np.interp(abs((a+np.pi)%(2*np.pi)-np.pi),[0,.38,.66,.88,1.04,1.19,1.38,1.60,1.85,2.15,2.65,np.pi],
        [119,115,128,161,220,315,373,341,349,423,455,465])+4*np.sin(a*9+.3)

def sweep(a,y):
    return a + .37*np.sin((y-55)/97+a*.7)+.0033*(y-145)+.12*np.sin(2*a)

class Head:
    def __init__(self,photo,detector):
        image=np.asarray(Image.open(photo).convert('RGB'));assert image.shape[:2]==(900,1200)
        M=cv2.getRotationMatrix2D((601,287),-15.8,1);M[:,2]-=[341,-25]
        self.im=cv2.warpAffine(image,M,(520,752),borderMode=cv2.BORDER_REPLICATE)
        opts=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(detector)))
        with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as d: lm=points(d,self.im)
        # Depth from the tracker, calibrated against the forehead and nasal tip.
        # Keep the photograph's x/y positions (smile asymmetry, lids and folds).
        xy=lm[:,:2]; target=57-lm[:,2]*1.04
        y=xy[:,1];rx,front,rear=PROFILE(y).T
        co=np.sqrt(np.maximum(0,1-((xy[:,0]-260)/rx)**2));base=(front+rear)/2+(front-rear)/2*co
        good=(co>.24);xy=xy[good];val=(target-base)[good]
        cells,g=np.unique(np.round(xy/3),axis=0,return_inverse=True);nn=np.bincount(g)
        s=np.stack([np.bincount(g,weights=v)/nn for v in xy.T],axis=1);v=np.bincount(g,weights=val)/nn
        # Zero-depth anchors lock the forehead, jaw and sides outside landmark coverage.
        border=[]
        for yy in np.linspace(135,620,20):
            r=PROFILE(yy)[0];border.extend(((260-r,yy),(260+r,yy)))
        border.extend((xx,144) for xx in np.linspace(145,375,8));border.extend((xx,611) for xx in np.linspace(207,313,8))
        self.relief=RBFInterpolator(np.vstack((s,border))/100,np.r_[v,np.zeros(len(border))],smoothing=.012)

    def shape(self,a,y,detail=True):
        a,y=np.broadcast_arrays(a,y);rx,front,rear=np.moveaxis(PROFILE(np.clip(y,37,744)),-1,0)
        co=np.cos(a);si=np.sin(a);x=260+rx*si
        below=fade(449,617,y)*(1-fade(630,744,y))
        yy=y-83*below*(1-co)*.64
        z=(front+rear)/2+(front-rear)/2*co
        if detail:
            mask=(co>.04)&(yy>143)&(yy<613);corr=np.zeros_like(z)
            if np.any(mask):corr[mask]=self.relief(np.stack((x[mask],yy[mask]),axis=1)/100)
            central=1-.68*fade(65,123,abs(x-260))*fade(375,442,yy)
            z=z+np.clip(corr,-53,82)*central*fade(.18,.60,co)*fade(142,194,yy)*(1-fade(560,610,yy))
        hair=(1-fade(-5,3,y-hairline(a)))*fade(37,61,y)
        # Broad locks in the skull outline, including asymmetric lift over the part.
        ridge=(1.7*np.sin(27*sweep(a,y))+.75*np.sin(61*sweep(a,y)+.5))*hair
        x=x+si*ridge; z=z+co*ridge; yy=yy-hair*5*np.exp(-((a+.25)/.59)**2-((y-80)/38)**2)
        return np.stack((x,yy,z),axis=-1)

    def head_texture(self,side=2048):
        # Cylindrical chart. Tiling behind the ears uses high-frequency residuals
        # from a real skin patch, not a stretched copy of the front-facing ears.
        out=np.zeros((side,side,3),'u1'); height=1536
        skinpatch=self.im[351:398,170:214].astype('f4'); skinpatch-=cv2.GaussianBlur(skinpatch,(0,0),5)
        temple=self.im[186:269,93:128].astype('f4'); crown=self.im[83:107,196:275].astype('f4')
        for y0 in range(0,height,64):
            row=np.arange(y0,min(y0+64,height))+.5
            u,v=np.meshgrid((np.arange(side)+.5)/side,row/height);a=u*2*np.pi-np.pi;y=37+707*v
            xyz=self.shape(a,y,False);x,yy=xyz[:,:,0],xyz[:,:,1];co=np.cos(a)
            def lookup(im,xx,rr):return cv2.remap(im,xx.astype('f4'),rr.astype('f4'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            # The visible front stays photographic, with mild flash falloff correction.
            front=lookup(self.im,x,yy).astype(float)
            variation=lookup(skinpatch,151*a+3*np.sin(yy/61),yy*.72)
            skin=np.array([176,126,99])+variation*.75+4*np.sin(yy/49+a*2)[:,:,None]
            # Swept, irregular filaments at three scales, broken up by darker clumps.
            q=sweep(a,y);bend=1.5*np.sin(16*q + .015*y)+.8*np.sin(39*q-.011*y)
            fine=np.sin(680*q+2.3*bend)+.65*np.sin(1073*q+4*bend+np.sin(y*.11))
            lock=np.sin(48*q+1.4*np.sin(y*.021))+.48*np.sin(79*q-1.2*bend)
            silver=100+17*lock+12*fine+18*np.maximum(0,np.sin(233*q+2*bend))**8+10*np.cos(a*2)
            fibers=lookup(temple,89*q+2*bend,y*.83).mean(2)-temple.mean()
            tufts=lookup(crown,y*.57+3*np.sin(9*a),47*q+1.7*bend).mean(2)-crown.mean()
            silver=.5*silver+48+.74*fibers+.36*tufts
            wool=np.stack((silver+5,silver+3,silver),axis=-1)
            is_hair=1-fade(-5,4,y-hairline(a)+2*np.sin(a*81))
            base=skin*(1-is_hair[:,:,None])+wool*is_hair[:,:,None]
            # Fade before the silhouette to keep wall and dark collar out of the bake.
            blend=fade(.25,.64,co)*(1-fade(592,627,yy))*fade(37,62,y)
            if np.any(is_hair):
                lum=front.mean(2,keepdims=True);front+=is_hair[:,:,None]*(.11*(lum-front)+9)
            out[y0:y0+len(row)]=np.clip(front*blend[:,:,None]+base*(1-blend[:,:,None]),0,255)
        # Dedicated UV swatches for ears and fine hair geometry.
        yy,xx=np.mgrid[:512,:1024]; patch=cv2.remap(self.im,(172+xx%35).astype('f4'),(352+yy%41).astype('f4'),cv2.INTER_LINEAR)
        ear_x=(xx/2048-.24)/.17;ear_y=((yy+1536)/2048-.874)/.112;r=np.hypot(ear_x,ear_y)
        cavity=np.exp(-((ear_x-.15)/.31)**2-((ear_y+.06)/.42)**2)
        anti=np.exp(-((r-.47-.09*ear_x)/.12)**2);rim=np.exp(-((r-.81)/.09)**2)
        detail=(12*anti+15*rim-52*cavity-21*np.exp(-((r-.65)/.11)**2))[:,:,None]
        out[1536:,:1024]=np.clip(.34*patch+.66*np.array([179,128,105])+detail,0,255)
        yy,xx=np.mgrid[:512,:1024];t=xx/1023; b=yy/511
        strand=91+35*b+11*np.sin(yy*.54+2*np.sin(t*7))+5*np.sin(yy*2.7+t*11)+7*np.sin(yy*.12+t*3)
        out[1536:,1024:]=np.clip(np.stack((strand+5,strand+3,strand),axis=-1),0,255)
        return Image.fromarray(out)

class Mesh:
    def __init__(self):self.part=[]
    def grid(self,p,uv,flip=False):
        h,w=p.shape[:2];a=np.arange((h-1)*w).reshape(h-1,w)[:,:-1];tri=np.stack((a,a+w,a+1,a+1,a+w,a+w+1),axis=-1).reshape(-1,3)
        self.part.append((p.reshape(-1,3),uv.reshape(-1,2),tri[:,::-1] if flip else tri))
    def finish(self):
        pp,uu,ii=[],[],[];offset=0
        for p,uv,idx in self.part:pp.append(p);uu.append(uv);ii.append(idx+offset);offset+=len(p)
        p=np.vstack(pp);uv=np.vstack(uu);idx=np.vstack(ii); pp=(p-[260,37,0])*[.000515,-.000515,.000515]+[0,1.896,.075]
        # Preserve the skull seam across the wrapping UV chart.
        _,g=np.unique(pp.round(7),axis=0,return_inverse=True);norm=np.zeros((g.max()+1,3));tri=pp[idx];face=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);
        for k in range(3):np.add.at(norm,g[idx[:,k]],face)
        norm=norm[g];norm/=np.maximum(1e-12,np.linalg.norm(norm,axis=1,keepdims=True));return pp.astype('f4'),norm.astype('f4'),uv.astype('f4'),idx.ravel().astype('u4')

def sculpt(head):
    m=Mesh(); aa=np.linspace(-np.pi,np.pi,149); rows=np.unique(np.r_[np.linspace(37,744,90),[315,319,324,389,395,403,409,415,422,461,467,471,476,480,488]])
    a,y=np.meshgrid(aa,rows);p=head.shape(a,y); uv=np.stack(((a+np.pi)/(2*np.pi),(y-37)/707*.75),axis=-1);m.grid(p,uv)
    # Close the tiny crown pole and neck opening.
    for end,ring,uvring in [(0,p[0],uv[0]),(-1,p[-1],uv[-1])]:
        c=ring.mean(0);pos=np.vstack((c,ring));tex=np.vstack((uvring.mean(0),uvring));ids=np.array([(0,i+1,i+2) for i in range(len(ring)-1)]);m.part.append((pos,tex,ids if end==0 else ids[:,::-1]))
    for sign in (-1,1):
        # Auricle: a tilted helix rim, antihelix ridge, recessed bowl and lobule.
        t,r=np.meshgrid(np.linspace(0,2*np.pi,43),np.linspace(.015,1,15));yy=401+51*r*np.cos(t);zz=-99+27*r*np.sin(t)-.19*(yy-401)
        rise=13*np.exp(-((r-.83)/.14)**2)+7*np.exp(-((r-.48-.10*np.sin(t))/.13)**2)-6*np.exp(-(r/.3)**2)
        xx=260+sign*(151+15*(1-r*r)+rise+3*np.cos(t)*r);xyz=np.stack((xx,yy,zz),axis=-1)
        uv=np.stack((.24+.17*r*np.sin(t),.874+.112*r*np.cos(t)),axis=-1);m.grid(xyz,uv,flip=sign>0)
    rng=np.random.default_rng(726)
    # Each tapered lock has a convex cross section. Fine flyaways break the outline.
    for i in range(730):
        fly=i>=330; root=rng.uniform(-np.pi,np.pi);lo,hi=57,max(58,hairline(root)-10)
        height=rng.uniform(lo,hi);length=rng.uniform(28,86);s=np.linspace(0,1,10);direction=1 if root>-.28 else -1
        angle=root+direction*(.24+.20*np.cos(root))*s+.04*np.sin(np.pi*s);y=height+length*s*(.4+.58*abs(np.sin(root))) - (12+8*np.cos(root))*np.sin(s*np.pi)
        y=np.maximum(y,42); bottom=hairline(angle)+rng.uniform(-5,7);y=np.minimum(y,bottom)
        center=head.shape(angle,y,False);radial=np.stack((np.sin(angle),-.65*fade(174,48,y),np.cos(angle)),axis=-1)
        radial/=np.linalg.norm(radial,axis=1,keepdims=True);tangent=np.gradient(center,axis=0);cross=np.cross(tangent,radial);cross/=np.maximum(.001,np.linalg.norm(cross,axis=1,keepdims=True));width=rng.uniform(.18,.45) if fly else rng.uniform(.9,3.5)
        envelope=.03+.97*np.sin(np.pi*s)**.64; lift=(.4+(rng.uniform(2,7) if fly else rng.uniform(1.4,4))*np.sin(np.pi*s))*envelope
        strand=center+lift[:,None]*radial
        cols=np.array([-1,0,1]);geo=strand[:,None,:]+cross[:,None,:]*(width*envelope[:,None,None])*cols[None,:,None]+radial[:,None,:]*(1-cols**2)[None,:,None]*width*.4*envelope[:,None,None]
        v=rng.uniform(.758,.99);uv=np.stack(np.broadcast_arrays(.51+.48*s[:,None],v+cols[None,:]*.004),axis=-1);m.grid(geo,uv)
    return (*m.finish(),head.head_texture())


def write_glb(mesh,path):
    p,n,uv,idx,im=mesh;png=io.BytesIO();im.save(png,'PNG');views=[];body=bytearray();access=[]
    def view(b):body.extend(bytes(-len(body)%4));views.append(dict(buffer=0,byteOffset=len(body),byteLength=len(b)));body.extend(b);return len(views)-1
    for array,type_,comp in [(p,'VEC3',5126),(n,'VEC3',5126),(uv,'VEC2',5126),(idx,'SCALAR',5125)]:
        ac=dict(bufferView=view(array.tobytes()),componentType=comp,count=len(array),type=type_)
        if array is p:ac.update(min=p.min(0).tolist(),max=p.max(0).tolist())
        access.append(ac)
    image=view(png.getvalue());l=len(body);body.extend(bytes(-len(body)%4));mat={'name':'Portrait skin, ears and swept silver hair','doubleSided':True,'pbrMetallicRoughness':{'baseColorTexture':{'index':0},'roughnessFactor':.80,'metallicFactor':0}}
    j={'asset':{'version':'2.0','extras':{'credit':'Original geometry; user-supplied portrait. Meshy post used as visual reference only.'}},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'name':'Original head','mesh':0}],'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1,'TEXCOORD_0':2},'indices':3,'material':0}]}],'materials':[mat],'textures':[{'source':0}],'images':[{'bufferView':image,'mimeType':'image/png'}],'buffers':[{'byteLength':l}],'bufferViews':views,'accessors':access};meta=json.dumps(j,separators=(',',':')).encode();meta+=b' '*(-len(meta)%4);path.write_bytes(struct.pack('<4sII',b'glTF',2,28+len(meta)+len(body))+struct.pack('<I4s',len(meta),b'JSON')+meta+struct.pack('<I4s',len(body),b'BIN\0')+body)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('portrait',type=Path);ap.add_argument('landmarker',type=Path);ap.add_argument('out',type=Path);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    mesh=sculpt(Head(args.portrait,args.landmarker));write_glb(mesh,args.out/'original-head.glb');print('Original mesh:',len(mesh[0]),'vertices,',len(mesh[3])//3,'triangles')
    ctx=gl.create_standalone_context();sheet=Image.new('RGB',(1200,396));
    for k,a in enumerate((0,-42,84,155)):
        im,_=render(ctx,mesh,a);sheet.paste(Image.fromarray(im).resize((300,330),Image.Resampling.LANCZOS),(300*k,30))
    sheet=sheet.crop((0,30,1200,360));sheet.save(args.out/'turnaround.jpg',quality=95)
