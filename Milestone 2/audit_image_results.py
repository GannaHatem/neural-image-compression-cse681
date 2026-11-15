from pathlib import Path
import csv,json,math,struct
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results_m2_full');out=Path('review_images');out.mkdir(exist_ok=True)
rows=[];jpeg=[];errors=[]
for n in range(1,25):
 d=p/f'kodim{n:02d}';ref=np.array(Image.open(d/'reference.png').convert('RGB'));h,w=ref.shape[:2]
 if sorted((w,h))!=[512,768]:errors.append(f'{d.name}: unexpected dimensions')
 def score(im):
  mse=np.mean((ref.astype(float)-im.astype(float))**2)
  return 10*math.log10(255**2/mse)
 for width in [32,64,128]:
  rr=list(csv.DictReader((d/f'width{width}.csv').open()))
  assert len(rr)==3
  for r in rr:
   for k in r:
    if k!='image':r[k]=float(r[k])
   bits=int(r['bits']);stem=d/f'width{width}_{bits}bit'
   raw=stem.with_suffix('.inr').read_bytes();z=struct.unpack('<I',raw[4:8])[0];header=json.loads(raw[8:8+z])
   assert len(raw)==8+z+sum(t['bytes'] for t in header['tensors'])
   assert header['w']==w and header['h']==h
   value=score(np.array(Image.open(stem.with_suffix('.png')).convert('RGB')))
   if abs(value-r['psnr_db'])>1e-8:errors.append(str(stem)+' PSNR mismatch')
   if len(raw)!=r['file_bytes'] or abs(8*len(raw)/(w*h)-r['bpp'])>1e-10:errors.append(str(stem)+' rate mismatch')
   rows.append(r)
 for r in csv.DictReader((d/'jpeg.csv').open()):
  r={k:(v if k=='image' else float(v)) for k,v in r.items()};q=int(r['quality']);file=d/f'jpeg_q{q}.jpg'
  if abs(score(np.array(Image.open(file).convert('RGB')))-r['psnr_db'])>1e-8:errors.append(str(file)+' PSNR mismatch')
  if file.stat().st_size!=r['file_bytes']:errors.append(str(file)+' size mismatch')
  jpeg.append(r)
def write(name,rr):
 with (out/name).open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=rr[0]);wr.writeheader();wr.writerows(rr)
write('neural_all.csv',rows);write('jpeg_all.csv',jpeg)
summary=[]
for width in [32,64,128]:
 for bits in [6,8,10]:
  rr=[r for r in rows if r['width']==width and r['bits']==bits]
  summary.append(dict(width=width,bits=bits,images=len(rr),parameters=int(rr[0]['parameters']),
   mean_bpp=np.mean([r['bpp'] for r in rr]),mean_psnr_db=np.mean([r['psnr_db'] for r in rr]),
   mean_training_seconds=np.mean([r['training_seconds'] for r in rr]),mean_decode_seconds=np.mean([r['decode_seconds'] for r in rr])))
write('neural_summary.csv',summary)
js=[]
for q in [1,3,5,10,20,30,50,70,90]:
 rr=[r for r in jpeg if r['quality']==q];js.append(dict(quality=q,mean_bpp=np.mean([r['bpp'] for r in rr]),mean_psnr_db=np.mean([r['psnr_db'] for r in rr])))
write('jpeg_summary.csv',js)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8,5))
ax.plot([r['mean_bpp'] for r in js],[r['mean_psnr_db'] for r in js],'o-',label='JPEG 4:4:4')
for width in [32,64,128]:
 rr=[r for r in summary if r['width']==width];ax.plot([r['mean_bpp'] for r in rr],[r['mean_psnr_db'] for r in rr],'o-',label=f'Neural width {width}')
ax.set(xlabel='Mean actual bits per pixel (headers included)',ylabel='Mean RGB PSNR (dB)',title='Kodak 24 images: 5,000 training steps per image / width')
ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'rd_curve.png',dpi=180);fig.savefig(out/'rd_curve.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
for bits in [6,8,10]:
 rr=[r for r in summary if r['bits']==bits];ax.plot([r['parameters'] for r in rr],[r['mean_psnr_db'] for r in rr],'o-',label=f'{bits}-bit weights')
ax.set(xlabel='Network parameter count',ylabel='Mean decoded RGB PSNR (dB)',title='Parameter-count sweep on all 24 Kodak images');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'parameter_sweep.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(3,3,figsize=(12,9))
for i,n in enumerate([1,13,23]):
 d=p/f'kodim{n:02d}';r=next(r for r in rows if r['image']==d.name and r['width']==128 and r['bits']==8)
 j=min([j for j in jpeg if j['image']==d.name],key=lambda j:abs(j['bpp']-r['bpp']))
 for ax,file,title in zip(axes[i],[d/'reference.png',d/f"jpeg_q{int(j['quality'])}.jpg",d/'width128_8bit.png'],
  [f'{d.name}: original',f"JPEG: {j['bpp']:.3f} bpp / {j['psnr_db']:.2f} dB",f"Neural w128 / 8 bit: {r['bpp']:.3f} bpp / {r['psnr_db']:.2f} dB"]):
  ax.imshow(Image.open(file));ax.set_title(title,fontsize=10);ax.axis('off')
fig.suptitle('Qualitative examples: nearest sampled JPEG rate (rates are not identical)',fontsize=13);fig.tight_layout();fig.savefig(out/'qualitative_examples.png',dpi=160);plt.close(fig)
# Per-image interpolation only inside sampled JPEG support, never extrapolation.
deltas=[]
for s in summary:
 diff=[]
 for r in rows:
  if r['width']!=s['width'] or r['bits']!=s['bits']:continue
  jj=sorted([j for j in jpeg if j['image']==r['image']],key=lambda j:j['bpp'])
  if jj[0]['bpp']<=r['bpp']<=jj[-1]['bpp']:
   baseline=np.interp(r['bpp'],[j['bpp'] for j in jj],[j['psnr_db'] for j in jj]);diff.append(r['psnr_db']-baseline)
 deltas.append(dict(width=s['width'],bits=s['bits'],overlap_images=len(diff),mean_neural_minus_interpolated_jpeg_db=float(np.mean(diff)) if diff else None))
write('jpeg_overlap_comparison.csv',deltas)
total=sum(r['training_seconds'] for r in rows if r['bits']==8)
report=['# M2 results review','',f'Validated {len(rows)} neural records and {len(jpeg)} JPEG records across 24 full-resolution Kodak images.',f'Checks: PNG-based RGB PSNR, actual file bytes, bpp, and neural payload lengths. Errors: {len(errors)}.', 'This audit does not independently rerun neural inference from the bitstreams.',f'Total training time across 72 distinct image/width fits: {total/60:.2f} minutes. Three precisions reuse each fit; do not count that training three times.','', '| Width | Bits | Parameters | Mean bpp | Mean PSNR (dB) | Mean training (s) |','|---|---|---|---|---|---|']
for r in summary:report.append(f"| {r['width']} | {r['bits']} | {r['parameters']} | {r['mean_bpp']:.4f} | {r['mean_psnr_db']:.3f} | {r['mean_training_seconds']:.2f} |")
report+=['','## Protocol and limits','- Arithmetic mean of per-image RGB PSNR, and arithmetic mean of bpp; all images have 393,216 pixels.','- Neural bytes include JSON metadata, bit-packed weights and header; no entropy coding.','- JPEG is 4:4:4. Curves are measured at fixed JPEG quality or fixed neural width/precision, not exact matched rates.','- jpeg_overlap_comparison.csv interpolates JPEG PSNR linearly in bpp per image, only inside measured support. It is an approximate matched-rate comparison; no extrapolation.','- Qualitative examples use the nearest sampled JPEG rate, with both actual rates shown.','- No HEVC results supplied yet. Do not report HEVC numbers until that baseline has run.','- Per-image fitting intentionally optimizes the evaluated image; a conventional training/test split does not apply. Architecture/budget tuning should be disclosed.','- Actual format uses JSON endpoints and architecture metadata, deviating from the compact fixed-header estimate in M1.','- CPU timing is specific to this machine and software setup. JPEG timings cover in-memory encoding; neural encoding covers training plus serialization.','', '## Next steps','Run the supplied HEVC baseline on the same original images, then combine its measured CSV with these results. Add the M2 report using the course template, contribution statement and AI-assistance disclosure.']
(out/'RESULTS_REVIEW.md').write_text('\n'.join(report))
print(json.dumps({'errors':errors,'training_minutes':total/60,'summary':summary,'jpeg':js,'overlap':deltas},indent=2))
