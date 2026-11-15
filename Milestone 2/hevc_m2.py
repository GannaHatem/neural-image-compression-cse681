"""Full-resolution Kodak HEVC-intra baseline, RGB PSNR, actual raw-stream bpp."""
import argparse,csv,json,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image

def run(cmd):
 r=subprocess.run(cmd,capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr[-4000:])

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset',required=True,type=Path)
 p.add_argument('--out',type=Path,default=Path('results_hevc_m2'))
 p.add_argument('--ffmpeg',default='ffmpeg')
 p.add_argument('--crf',nargs='+',type=int,default=[10,20,30,40,45,51])
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 manifest={'dataset':str(a.dataset.resolve()),'crf':a.crf,'preset':'medium','pix_fmt':'yuv444p','threads':4}
 mp=a.out/'config.json'
 if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Choose a new output directory for changed settings')
 mp.write_text(json.dumps(manifest,indent=2))
 version=subprocess.run([a.ffmpeg,'-version'],capture_output=True,text=True,check=True)
 (a.out/'ffmpeg_version.txt').write_text(version.stdout)
 rows=[]
 for n in range(1,25):
  name=f'kodim{n:02d}';folder=a.out/name;folder.mkdir(exist_ok=True)
  source=a.dataset/(name+'.png')
  with Image.open(source) as im:ref=np.array(im.convert('RGB'))
  h,w=ref.shape[:2];reference=folder/'reference.png';Image.fromarray(ref).save(reference)
  for crf in a.crf:
   stream=folder/f'crf{crf}.hevc';decoded=folder/f'crf{crf}.png';record=folder/f'crf{crf}.json'
   if record.exists() and stream.exists() and decoded.exists():
    rows.append(json.loads(record.read_text()));continue
   start=time.perf_counter()
   run([a.ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(reference),
        '-frames:v','1','-pix_fmt','yuv444p','-c:v','libx265','-preset','medium',
        '-x265-params','keyint=1:min-keyint=1:scenecut=0:log-level=error:pools=4:frame-threads=1',
        '-crf',str(crf),'-f','hevc',str(stream)])
   encoding=time.perf_counter()-start;start=time.perf_counter()
   run([a.ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(stream),'-frames:v','1',str(decoded)])
   decoding=time.perf_counter()-start
   with Image.open(decoded) as im:rec=np.array(im.convert('RGB'))
   if rec.shape!=ref.shape:raise ValueError('Decoded dimensions differ')
   mse=float(np.mean((ref.astype(np.float64)-rec.astype(np.float64))**2))
   row=dict(image=name,crf=crf,width=w,height=h,file_bytes=stream.stat().st_size,
       bpp=8*stream.stat().st_size/(w*h),psnr_db=float('inf') if mse==0 else float(10*np.log10(255**2/mse)),
       encode_seconds=encoding,decode_seconds=decoding,pixel_format='yuv444p',preset='medium')
   record.write_text(json.dumps(row,indent=2));rows.append(row);print(row,flush=True)
  with (a.out/'hevc_all.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print('Completed:',a.out/'hevc_all.csv')
if __name__=='__main__':main()
