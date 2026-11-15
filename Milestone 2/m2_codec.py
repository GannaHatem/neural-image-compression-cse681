"""CPU M2 image codec. RGB PSNR uses decoded 8-bit pixels; rate includes headers.
COIN-inspired Fourier MLP, not a reproduction of C3. No external training set.
"""
import argparse, csv, io, json, math, struct, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn

class FourierMLP(nn.Module):
    def __init__(self, width=64, frequencies=8):
        super().__init__()
        self.register_buffer('freqs', 2.0 ** torch.arange(frequencies) * math.pi)
        self.net = nn.Sequential(nn.Linear(4*frequencies,width),nn.ReLU(),
            nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),
            nn.Linear(width,3),nn.Sigmoid())
    def forward(self, xy):
        angles=xy[...,None]*self.freqs
        return self.net(torch.cat([angles.sin(),angles.cos()],-1).flatten(1))

def coords(indices,w,h):
    return torch.stack([(indices%w).float()/max(w-1,1),
                        (indices//w).float()/max(h-1,1)],1)

@torch.inference_mode()
def reconstruct(model,w,h):
    out=np.empty((w*h,3),dtype=np.uint8)
    for start in range(0,w*h,4096):
        end=min(start+4096,w*h)
        rgb=model(coords(torch.arange(start,end),w,h)).numpy()
        out[start:end]=np.clip(np.rint(rgb*255),0,255).astype(np.uint8)
    return out.reshape(h,w,3)

def psnr(a,b):
    mse=np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return float('inf') if mse==0 else 10*math.log10(255**2/mse)

def encode(model,path,w,h,width,bits):
    # Per-tensor float32 endpoints and genuinely packed integer symbols.
    header={'version':1,'width':width,'frequencies':8,'w':w,'h':h,
            'bits':bits,'tensors':[]}
    payload=bytearray()
    for name,p in model.named_parameters():
        a=p.detach().numpy().astype(np.float32)
        lo,hi=float(a.min()),float(a.max())
        q=np.zeros(a.size,dtype=np.uint16) if hi==lo else np.rint(
            (a.astype(np.float64).ravel()-lo)/(hi-lo)*((1<<bits)-1)).astype(np.uint16)
        binary=((q[:,None] >> np.arange(bits-1,-1,-1)) & 1).astype(np.uint8)
        packed=np.packbits(binary.ravel()).tobytes()
        header['tensors'].append({'name':name,'shape':list(a.shape),'lo':lo,
            'hi':hi,'bytes':len(packed)})
        payload.extend(packed)
    raw=json.dumps(header,separators=(',',':')).encode()
    path.write_bytes(b'INR1'+struct.pack('<I',len(raw))+raw+payload)

def load_codec(path):
    raw=path.read_bytes()
    if raw[:4]!=b'INR1': raise ValueError('Invalid codec magic')
    n=struct.unpack('<I',raw[4:8])[0]
    meta=json.loads(raw[8:8+n]); pos=8+n
    if meta['version']!=1 or not 1<=meta['bits']<=16: raise ValueError('Unsupported format')
    model=FourierMLP(meta['width'],meta['frequencies'])
    params=dict(model.named_parameters()); bits=meta['bits']
    with torch.no_grad():
        for t in meta['tensors']:
            count=int(np.prod(t['shape'])); size=t['bytes']
            packed=np.frombuffer(raw[pos:pos+size],dtype=np.uint8); pos+=size
            binary=np.unpackbits(packed)[:count*bits].reshape(count,bits)
            q=(binary.astype(np.uint32)*(1<<np.arange(bits-1,-1,-1))).sum(1)
            a=t['lo']+q.astype(np.float64)/((1<<bits)-1)*(t['hi']-t['lo'])
            params[t['name']].copy_(torch.from_numpy(a.reshape(t['shape']).astype(np.float32)))
    if pos!=len(raw): raise ValueError('Unexpected payload length')
    return model.eval(),meta

def save_csv(path,rows):
    temp=path.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)

def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    configuration={'steps':args.steps,'widths':args.widths,'bits':args.bits,
        'seed':42,'batch':args.batch,'threads':args.threads,'dataset':str(args.dataset.resolve()),
        'torch':str(torch.__version__),'numpy':np.__version__,'resolution':'original'}
    manifest=args.out/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=configuration:
        raise ValueError('Output configuration differs; choose a new --out folder.')
    manifest.write_text(json.dumps(configuration,indent=2))
    for number in args.images:
        name=f'kodim{number:02d}'
        with Image.open(args.dataset/(name+'.png')) as im: reference=np.array(im.convert('RGB'))
        h,w=reference.shape[:2]; folder=args.out/name;folder.mkdir(exist_ok=True)
        Image.fromarray(reference).save(folder/'reference.png')
        jpeg=[]
        for quality in [1,3,5,10,20,30,50,70,90]:
            buf=io.BytesIO();start=time.perf_counter()
            Image.fromarray(reference).save(buf,format='JPEG',quality=quality,subsampling=0)
            seconds=time.perf_counter()-start;raw=buf.getvalue()
            (folder/f'jpeg_q{quality}.jpg').write_bytes(raw)
            decoded=np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
            jpeg.append(dict(image=name,quality=quality,bpp=8*len(raw)/(w*h),
                psnr_db=psnr(reference,decoded),file_bytes=len(raw),encode_seconds=seconds))
        save_csv(folder/'jpeg.csv',jpeg)
        pixels=torch.from_numpy(reference.copy()).float().reshape(-1,3)/255
        for width in args.widths:
            result=folder/f'width{width}.csv'
            if result.exists():
                print(f'Skipping completed {name} width={width}',flush=True);continue
            torch.manual_seed(42);model=FourierMLP(width)
            opt=torch.optim.Adam(model.parameters(),lr=0.001)
            history=[];start=time.perf_counter()
            for step in range(1,args.steps+1):
                idx=torch.randint(0,w*h,(args.batch,))
                loss=((model(coords(idx,w,h))-pixels[idx])**2).mean()
                opt.zero_grad();loss.backward();opt.step()
                if step%500==0 or step==args.steps:
                    elapsed=time.perf_counter()-start
                    history.append(dict(step=step,batch_mse=loss.item(),seconds=elapsed))
                    print(f'{name} width={width} step={step}/{args.steps} {elapsed:.1f}s',flush=True)
            training=time.perf_counter()-start
            save_csv(folder/f'history_width{width}.csv',history)
            torch.save(model.state_dict(),folder/f'checkpoint_width{width}.pt')
            rows=[]
            for bits in args.bits:
                path=folder/f'width{width}_{bits}bit.inr';start=time.perf_counter()
                encode(model,path,w,h,width,bits);serialization=time.perf_counter()-start
                # Decode into a fresh model solely from the file (no checkpoint/reference).
                start=time.perf_counter();decoded_model,meta=load_codec(path)
                decoded=reconstruct(decoded_model,meta['w'],meta['h'])
                decoding=time.perf_counter()-start
                Image.fromarray(decoded).save(path.with_suffix('.png'))
                row=dict(image=name,width=width,bits=bits,parameters=sum(p.numel() for p in model.parameters()),
                    bpp=8*path.stat().st_size/(w*h),psnr_db=psnr(reference,decoded),
                    file_bytes=path.stat().st_size,training_seconds=training,
                    encode_seconds=training+serialization,decode_seconds=decoding)
                rows.append(row);print(row,flush=True)
            save_csv(result,rows)
    print('Completed. CSVs and actual .inr files are in',args.out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--out',type=Path,default=Path('results_m2'))
    p.add_argument('--images',nargs='+',type=int,default=list(range(1,25)))
    p.add_argument('--widths',nargs='+',type=int,default=[32,64,128])
    p.add_argument('--bits',nargs='+',type=int,default=[6,8,10])
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--batch',type=int,default=2048)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--decode',type=Path,help='Decode one .inr file without source image')
    p.add_argument('--png',type=Path,default=Path('decoded.png'))
    a=p.parse_args()
    if a.steps<1 or a.batch<1 or any(b<1 or b>16 for b in a.bits):p.error('Invalid steps/batch/bits')
    torch.set_num_threads(a.threads)
    if a.decode:
        model,m=load_codec(a.decode);Image.fromarray(reconstruct(model,m['w'],m['h'])).save(a.png)
    else:
        if a.dataset is None:p.error('--dataset required for training')
        run(a)
if __name__=='__main__':main()
