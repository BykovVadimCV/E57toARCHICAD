"""Build self-contained, lazily decoded native-resolution overlap viewers."""
from pathlib import Path
import json,base64,zipfile
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'review/rectangle_overlap_penalty_v1';OLD=ROOT/'review/wall_window_barrier_decomposition_v3'
WINDOWS_INCLUDED=True
BEFORE_LABEL='Previous fit'
AFTER_LABEL='Overlap penalty enabled'

def data(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
def main():
 for key,name in [('good/L1','wall_overlap_good.html'),('medium/L1','wall_overlap_barklaya_L1.html'),('medium/L2','wall_overlap_barklaya_L2.html')]:
  folder=OUT/key;cloud=ROOT/'review/wall_point_overlay'/key
  from PIL import Image
  with Image.open(cloud/'wall_height.png') as im:w,h=im.size
  models={}
  for model in ['new_wall','new_plan']:
   models[model]={'paths':{},'images':{},'stats':{}}
   for version,base in [('before',OLD),('after',OUT)]:
    d=json.loads((base/key/(model+'_decomposition.json')).read_text());paths=[]
    if version=='after':
     guard=d.get('axis_regularization',{})
     structure=d.get('simplification',{}).get('after') or d.get('structure',{}).get('after')
     models[model]['note']=(f"Matched scan faces {structure['coverage']:.1%}; longest uncovered run {structure['longest_unexplained_m']:.2f}m" if structure else ('Previous fit retained to preserve connected walls' if guard.get('accepted') is False else ''))
     if d.get('merging'):
      m=d['merging'];models[model]['note']+=f"; pieces {m['before']} → {m['after']}; "+('outline refined' if m.get('changed_native_pixels',0) else 'native outline unchanged')
    if version=='after' and d.get('footprint_cleanup'):models[model]['note']=f"{len(d['primitives'])} active footprint regions; {len(d['footprint_cleanup']['review_features'])} detached features in review layer"
    if version=='after' and d.get('thickness_regularization'):models[model]['note']+=f"; {len(d['thickness_regularization']['edits'])} width-stabilized spans"
    for c in d['primitives']:paths.append(' '.join('M '+' L '.join(f'{x:.4f},{y:.4f}' for x,y in ring)+' Z' for ring in [c['polygon']]+c.get('holes',[])))
    models[model]['paths'][version]=paths
    models[model]['images']['overlap_'+version]=data(folder/(model+'_overlap_'+version+'.png'))
   for layer,file in [('original','original'),('refined','refined'),('errors','vector_errors'),('excluded','excluded'),('barriers','barriers')]:models[model]['images'][layer]=data(folder/(model+'_'+file+'.png'))
   # Optional version-specific masks for checkpoint comparisons; legacy reviews
   # without these files keep their existing shared current-mask overlays.
   for version,base in [('before',OLD),('after',OUT)]:
    for layer in ['original','refined','probability','excluded','barriers']:
     p=base/key/(model+'_'+layer+'.png')
     if p.exists():models[model]['images'][layer+'_'+version]=data(p)
   if (folder/(model+'_review.png')).exists():models[model]['images']['review']=data(folder/(model+'_review.png'))
   if (folder/(model+'_removed_geometry.png')).exists():models[model]['images']['removed']=data(folder/(model+'_removed_geometry.png'))
  rows=json.loads((OUT/'summary.json').read_text())
  for r in rows:
   if r['case']==key:models[r['model']]['stats']=r['rectangle_overlap_comparison']
  payload={'models':models,'backgrounds':{k:data(cloud/(k+'.png')) for k in ['wall_height','slice_1m']},'width':w,'height':h}
  page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wall overlap comparison</title><style>body{margin:0;background:#111;color:#eee;font:15px system-ui}header{padding:10px}label,button,select{margin:4px 7px}canvas{display:block;touch-action:none}#error{color:#ff8b82;white-space:pre-wrap}small{color:#bbb}</style><header><b>TITLE — wall overlap comparison</b><br><select id="model"><option>new_wall</option><option>new_plan</option></select><select id="version"><option value="after">Overlap penalty enabled</option><option value="before">Previous fit</option></select><select id="cloud"><option value="wall_height">Wall-height points</option><option value="slice_1m">1.0–1.2m slice</option></select><br><select id="overlay"><option value="none">No raster overlay</option><option value="overlap">Rectangle overlap (pink)</option><option value="original">Original mask</option><option value="refined">Refined mask</option><option value="errors">Vector errors (current fit)</option><option value="excluded">Excluded doors/windows/openings</option><option value="barriers">Dense scan barriers</option></select><label>Vectors <input id="vectors" type="checkbox" checked></label><label>Fill <input id="opacity" type="range" min="0" max=".8" step=".01" value=".2"></label><button id="fit">Fit</button><button id="native">1:1 pixels</button><span id="status">Loading selected layers…</span><br><small>Wheel zoom · drag pan · 2.5mm point display. Windows included. Only the selected background and raster layer are decoded.</small><div id="error"></div></header><canvas id="canvas"></canvas><script>
 'use strict';
 window.onerror=function(message){document.getElementById('error').textContent='Viewer error: '+message;};
 const DATA=PAYLOAD;
 const el=id=>document.getElementById(id),canvas=el('canvas'),ctx=canvas.getContext('2d');
 let scale=1,panX=0,panY=0,drag=null;
 const slots={background:{key:null,image:null},overlay:{key:null,image:null}};
 const paths={};
 for(const [model,entry] of Object.entries(DATA.models)){paths[model]={};for(const [version,list] of Object.entries(entry.paths))paths[model][version]=list.map(d=>new Path2D(d));}
 function release(slot){if(slot.image){slot.image.onload=null;slot.image.onerror=null;slot.image.src='';}slot.key=null;slot.image=null;}
 function selectedImage(slot,key,url){if(!url){release(slot);return null;}if(slot.key!==key){release(slot);const image=new Image();slot.key=key;slot.image=image;image.onload=()=>{if(slot.image===image)draw();};image.onerror=()=>{el('error').textContent='An image layer could not load. Try downloading and opening the standalone HTML file.';};image.src=url;}return slot.image.complete&&slot.image.naturalWidth?slot.image:null;}
 function draw(){const model=el('model').value,version=el('version').value,layer=el('overlay').value;const bg=selectedImage(slots.background,el('cloud').value,DATA.backgrounds[el('cloud').value]);const key=layer==='overlap'?'overlap_'+version:layer;const selectedKey=DATA.models[model].images[key+'_'+version]?key+'_'+version:key;const overlay=selectedImage(slots.overlay,model+'_'+selectedKey,layer==='none'?null:DATA.models[model].images[selectedKey]);ctx.globalAlpha=1;ctx.fillStyle='#000';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.save();ctx.translate(panX,panY);ctx.scale(scale,scale);ctx.imageSmoothingEnabled=false;if(bg)ctx.drawImage(bg,0,0,DATA.width,DATA.height);if(el('vectors').checked){for(const path of paths[model][version]){ctx.globalAlpha=+el('opacity').value;ctx.fillStyle='#9be141';ctx.fill(path,'evenodd');ctx.globalAlpha=1;ctx.strokeStyle='#ffb000';ctx.lineWidth=1/scale;ctx.stroke(path);}}if(overlay){ctx.globalAlpha=['overlap','errors','excluded','barriers'].includes(layer)?0.85:Math.max(.2,+el('opacity').value);ctx.drawImage(overlay,0,0,DATA.width,DATA.height);}ctx.restore();ctx.globalAlpha=1;const metric=DATA.models[model].stats[version];el('status').textContent=(scale*100).toFixed(0)+'% · duplicate rectangle area '+metric.redundant_area_m2.toFixed(3)+'m²'+(!bg||(layer!=='none'&&!overlay)?' · loading…':'');}
 function fit(){scale=Math.min(canvas.width/DATA.width,canvas.height/DATA.height);panX=(canvas.width-DATA.width*scale)/2;panY=(canvas.height-DATA.height*scale)/2;draw();}
 function resize(){canvas.width=Math.max(1,innerWidth);canvas.height=Math.max(200,innerHeight-canvas.getBoundingClientRect().top);fit();}
 canvas.addEventListener('wheel',e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,next=Math.max(.02,Math.min(64,scale*Math.exp(-e.deltaY*.0015)));panX=x-(x-panX)*next/scale;panY=y-(y-panY)*next/scale;scale=next;draw();},{passive:false});
 canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(drag){panX+=e.clientX-drag[0];panY+=e.clientY-drag[1];drag=[e.clientX,e.clientY];draw();}};canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;
 for(const id of ['model','version','cloud','overlay','vectors','opacity'])el(id).oninput=draw;
 el('fit').onclick=fit;el('native').onclick=()=>{const x=canvas.width/2,y=canvas.height/2;panX=x-(x-panX)/scale;panY=y-(y-panY)/scale;scale=1;draw();};window.onresize=resize;resize();
 </script>'''
  if any('review' in m['images'] for m in models.values()):page=page.replace('<option value="original">','<option value="review">Unconfirmed detached features</option><option value="original">')
  if any('probability_after' in m['images'] for m in models.values()):page=page.replace('<option value="original">','<option value="probability">P(wall): blue 0 → red 1</option><option value="original">')
  page=page.replace('TITLE',key).replace('PAYLOAD',json.dumps(payload,separators=(',',':')))
  page=page.replace('Overlap penalty enabled',AFTER_LABEL).replace('Previous fit',BEFORE_LABEL)
  if any('removed' in m['images'] for m in models.values()):page=page.replace('<option value="errors">','<option value="removed">Removed geometry (pink)</option><option value="errors">')
  page=page.replace("metric.redundant_area_m2.toFixed(3)+'m²'", "metric.redundant_area_m2.toFixed(3)+'m²'+(version==='after'&&DATA.models[model].note?' · '+DATA.models[model].note:'')")
  if not WINDOWS_INCLUDED:page=page.replace('Windows included.','Windows excluded.')
  from window_viewer import decorate
  page=decorate(page,folder)
  from door_viewer import decorate as decorate_doors
  page=decorate_doors(page,folder)
  from small_feature_viewer import decorate as decorate_features
  page=decorate_features(page,folder)
  (ROOT/name).write_text(page);(folder/'viewer_light.html').write_text(page);(folder/'viewer.html').write_text(page)
  print(name,len(page))
 with zipfile.ZipFile(ROOT/'wall_overlap_viewers.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in ['wall_overlap_good.html','wall_overlap_barklaya_L1.html','wall_overlap_barklaya_L2.html']:z.write(ROOT/name,name)
  z.writestr('README.txt','Extract this ZIP, then open any HTML file in Safari or Chrome. Files are self-contained; no server, account or network connection is required. The viewer decodes only the selected background and raster overlay. Native 2.5mm images are preserved. Use the fit selector for previous/current decomposition; Rectangle overlap highlights repeated rectangle coverage.\n')

if __name__=='__main__':main()
