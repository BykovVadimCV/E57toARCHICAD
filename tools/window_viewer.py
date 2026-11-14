"""Optional window-vector controls for the native footprint viewer."""
from pathlib import Path
import json,base64


def decorate(page,folder):
    folder=Path(folder);data={}
    def path(g):
        return ' '.join('M '+' L '.join(f'{x:.5f},{y:.5f}' for x,y in ring)+' Z' for ring in [g['polygon']]+g.get('holes',[]))
    for model in ['new_wall','new_plan']:
        source=folder/(model+'_windows.json')
        if not source.exists():return page
        d=json.loads(source.read_text());windows=[]
        for w in d['windows']:
            windows.append(dict(id=w['id'],paths=[path(g) for g in w['footprint_regions']],axis=w['axis'],width=w['width_m'],length=w['length_m']))
        review=[]
        for w in d['window_review']:
            x,y,ww,hh=w['seed_bbox_px'];x=x*8-.5;y=y*8-.5;ww*=8;hh*=8
            review.append(dict(paths=[path(dict(polygon=[[x,y],[x+ww,y],[x+ww,y+hh],[x,y+hh]]))],reason=', '.join(w['reasons'])))
        mask='data:image/png;base64,'+base64.b64encode((folder/(model+'_window_mask.png')).read_bytes()).decode()
        data[model]=dict(windows=windows,review=review,mask=mask)
    setup='const WINDOW_DATA='+json.dumps(data,separators=(',',':'))+";for(const [model,d] of Object.entries(WINDOW_DATA)){DATA.models[model].images.window_mask=d.mask;DATA.models[model].note+='; '+d.windows.length+' windows; '+d.review.length+' review';}\n"
    page=page.replace('const el=id=>',setup+'const el=id=>')
    setup2="const windowPaths=Object.fromEntries(Object.entries(WINDOW_DATA).map(([m,d])=>[m,d.windows.map(w=>({...w,paths:w.paths.map(p=>new Path2D(p))}))]));const reviewPaths=Object.fromEntries(Object.entries(WINDOW_DATA).map(([m,d])=>[m,d.review.flatMap(w=>w.paths.map(p=>new Path2D(p)))]));\n"
    page=page.replace(' function release(',setup2+' function release(')
    page=page.replace('<label>Vectors', '<label>Window vectors <input id="windows" type="checkbox" checked></label><label>Review seeds <input id="windowreview" type="checkbox"></label><select id="focuswindow"></select><button id="focus">Focus window</button><label>Vectors')
    page=page.replace('<option value="original">','<option value="window_mask">Original window mask</option><option value="original">')
    draw="if(version==='after'&&el('windows').checked){for(const w of windowPaths[model]){for(const p of w.paths){ctx.globalAlpha=.4;ctx.fillStyle='#00bfff';ctx.fill(p,'evenodd');ctx.globalAlpha=1;ctx.strokeStyle='#00d9ff';ctx.lineWidth=1.5/scale;ctx.stroke(p);}}}if(el('windowreview').checked){ctx.strokeStyle='#e248d8';ctx.globalAlpha=.8;ctx.lineWidth=1/scale;for(const p of reviewPaths[model])ctx.stroke(p);ctx.globalAlpha=1;}"
    page=page.replace('if(overlay){',draw+'if(overlay){')
    handlers="""function refreshWindows(){el('focuswindow').innerHTML=WINDOW_DATA[el('model').value].windows.map((w,i)=>'<option value="'+i+'">'+w.id+' · '+w.length.toFixed(2)+'m</option>').join('');}el('windows').oninput=draw;el('windowreview').oninput=draw;el('model').oninput=()=>{refreshWindows();draw();};el('focus').onclick=()=>{const w=WINDOW_DATA[el('model').value].windows[+el('focuswindow').value];if(!w)return;const a=w.axis[0],b=w.axis[1];scale=Math.min(canvas.width/(Math.abs(b[0]-a[0])+300),canvas.height/(Math.abs(b[1]-a[1])+300));panX=canvas.width/2-(a[0]+b[0])/2*scale;panY=canvas.height/2-(a[1]+b[1])/2*scale;draw();};refreshWindows();"""
    page=page.replace('window.onresize=resize;resize();',handlers+'window.onresize=resize;resize();')
    page=page.replace('Previous thickness pass','Walls only').replace('Assembled paired-face thickness','Walls + window gap closers')
    page=page.replace('Windows excluded.','Windows are separate gap closers; doors remain excluded. Blue = fitted window, magenta = review seed. Window heights are not inferred.')
    page=page.replace('wall overlap comparison','wall + window gap closers')
    return page
