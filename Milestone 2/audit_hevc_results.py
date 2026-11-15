from pathlib import Path
import csv,json,math,shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results_hevc_m2');neural=Path('results_m2_full');out=Path('regenerated_review');out.mkdir(exist_ok=True)
for f in Path('review_images').glob('*'):
 if f.is_file():shutil.copy(f,out/f.name)
shutil.copy('m2_codec.py',out/'m2_codec.py')
rows=list(csv.DictReader((p/'hevc_all.csv').open()));assert len(rows)==144
errors=[]
for r in rows:
 for k in ['crf','width','height','file_bytes','bpp','psnr_db','encode_seconds','decode_seconds']:r[k]=float(r[k])
 d=p/r['image'];ref=np.array(Image.open(d/'reference.png').convert('RGB'));rec=np.array(Image.open(d/f"crf{int(r['crf'])}.png").convert('RGB'))
 assert np.array_equal(ref,np.array(Image.open(neural/r['image']/'reference.png')))
 mse=np.mean((ref.astype(float)-rec.astype(float))**2);score=10*math.log10(255**2/mse)
 assert abs(score-r['psnr_db'])<1e-8
 size=(d/f"crf{int(r['crf'])}.hevc").stat().st_size
 assert size==r['file_bytes'] and abs(size*8/(ref.shape[0]*ref.shape[1])-r['bpp'])<1e-10
shutil.copy(p/'hevc_all.csv',out/'hevc_all.csv');shutil.copy(p/'ffmpeg_version.txt',out/'ffmpeg_version.txt')
summary=[]
for crf in [10,20,30,40,45,51]:
 rr=[r for r in rows if r['crf']==crf];assert len(rr)==24
 summary.append(dict(crf=crf,bpp=float(np.mean([r['bpp'] for r in rr])),psnr_db=float(np.mean([r['psnr_db'] for r in rr])),encode_seconds=float(np.mean([r['encode_seconds'] for r in rr]))))
with (out/'hevc_summary.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=summary[0]);wr.writeheader();wr.writerows(summary)
ns=list(csv.DictReader((out/'neural_summary.csv').open()));js=list(csv.DictReader((out/'jpeg_summary.csv').open()))
fig,ax=plt.subplots(figsize=(8,4.8))
ax.plot([float(r['mean_bpp']) for r in js],[float(r['mean_psnr_db']) for r in js],'o-',label='JPEG 4:4:4')
hs=sorted(summary,key=lambda r:r['bpp']);ax.plot([r['bpp'] for r in hs],[r['psnr_db'] for r in hs],'s-',label='HEVC-intra 4:4:4')
for width in [32,64,128]:
 rr=[r for r in ns if int(r['width'])==width];ax.plot([float(r['mean_bpp']) for r in rr],[float(r['mean_psnr_db']) for r in rr],'o-',label=f'Neural width {width}')
ax.set(xlabel='Mean actual bits per pixel (headers included)',ylabel='Mean RGB PSNR (dB)',title='Kodak 24: preliminary rate-distortion comparison');ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'rd_all.pdf');fig.savefig(out/'rd_all.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(3,4,figsize=(13,8))
nr=list(csv.DictReader((out/'neural_all.csv').open()));jr=list(csv.DictReader((out/'jpeg_all.csv').open()))
for i,n in enumerate([1,13,23]):
 name=f'kodim{n:02d}';d=neural/name
 r=next(r for r in nr if r['image']==name and int(float(r['width']))==64 and int(float(r['bits']))==8);rate=float(r['bpp'])
 j=min([j for j in jr if j['image']==name],key=lambda j:abs(float(j['bpp'])-rate));hh=min([j for j in rows if j['image']==name],key=lambda j:abs(j['bpp']-rate))
 files=[d/'reference.png',d/f"jpeg_q{int(float(j['quality']))}.jpg",p/name/f"crf{int(hh['crf'])}.png",d/'width64_8bit.png']
 titles=[name+' original',f"JPEG {float(j['bpp']):.3f} bpp\n{float(j['psnr_db']):.2f} dB",f"HEVC {hh['bpp']:.3f} bpp\n{hh['psnr_db']:.2f} dB",f"Neural {rate:.3f} bpp\n{float(r['psnr_db']):.2f} dB"]
 for ax,f,t in zip(axes[i],files,titles):ax.imshow(Image.open(f));ax.set_title(t,fontsize=10);ax.axis('off')
fig.suptitle('Nearest sampled baseline rates; actual rates differ',fontsize=12);fig.tight_layout();fig.savefig(out/'examples_all.jpg',dpi=150);plt.close(fig)
print(json.dumps(summary,indent=2));print('Verified all 144 HEVC records; identical reference pixels to neural/JPEG.')
