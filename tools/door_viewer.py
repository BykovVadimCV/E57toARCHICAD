"""Door opening controls for the native wall/window viewer; no guessed swing arcs."""
from pathlib import Path
import json,base64


def decorate(page,folder):
    folder=Path(folder);data={}
    def path(g):return ' '.join('M '+' L '.join(f'{x:.5f},{y:.5f}' for x,y in ring)+' Z' for ring in [g['polygon']]+g.get('holes',[]))
    for model in ['new_wall','new_plan']:
        source=folder/(model+'_doors.json')
        if not source.exists():return page
        d=json.loads(source.read_text());doors=[];review=[]
        for door in d['doors']:
            doors.append(dict(id=door['id'],paths=[path(g) for g in door['footprint_regions']],axis=door['axis'],
                span=door['opening_width_m'],thickness=door['host_thickness_m'],scan=door['scan_evidence']['status']))
        factor=d.get('probability_pixel_m',.02)/d.get('pixel_m',.0025)
        for door in d['door_review']:
            x,y,w,h=door['seed_bbox_px'];x=x*factor-.5;y=y*factor-.5;w*=factor;h*=factor
            review.append(dict(id='R'+str(door['seed_component']),paths=[path(dict(polygon=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]))],
                axis=[[x,y],[x+w,y+h]],reason=', '.join(door['reasons'])))
        mask='data:image/png;base64,'+base64.b64encode((folder/(model+'_door_mask.png')).read_bytes()).decode()
        data[model]=dict(doors=doors,review=review,mask=mask)
    setup='const DOOR_DATA='+json.dumps(data,separators=(',',':'))+";for(const [model,d] of Object.entries(DOOR_DATA)){DATA.models[model].images.door_mask=d.mask;DATA.models[model].note+='; '+d.doors.length+' doors; '+d.review.length+' door review';}\n"
    page=page.replace('const el=id=>',setup+'const el=id=>')
    paths="const doorPaths=Object.fromEntries(Object.entries(DOOR_DATA).map(([m,d])=>[m,d.doors.map(v=>({...v,paths:v.paths.map(p=>new Path2D(p))}))]));const doorReviewPaths=Object.fromEntries(Object.entries(DOOR_DATA).map(([m,d])=>[m,d.review.flatMap(v=>v.paths.map(p=>new Path2D(p)))]));\n"
    page=page.replace(' function release(',paths+' function release(')
    page=page.replace('<label>Vectors','<label>Doors <input id="doors" type="checkbox" checked></label><label>Door review <input id="doorreview" type="checkbox"></label><select id="focusdoor"></select><button id="doorfocus">Focus door</button><label>Vectors')
    page=page.replace('<option value="original">','<option value="door_mask">Original door mask</option><option value="original">')
    draw="if(version==='after'&&el('doors').checked){for(const d of doorPaths[model]){for(const p of d.paths){ctx.globalAlpha=.3;ctx.fillStyle='#ff6446';ctx.fill(p,'evenodd');ctx.globalAlpha=1;ctx.strokeStyle='#ff7855';ctx.lineWidth=1.6/scale;ctx.stroke(p);}}}if(el('doorreview').checked){ctx.strokeStyle='#f06bea';ctx.globalAlpha=.9;ctx.lineWidth=1/scale;for(const p of doorReviewPaths[model])ctx.stroke(p);ctx.globalAlpha=1;}"
    page=page.replace('if(overlay){',draw+'if(overlay){')
    handlers="""function refreshDoors(){const d=DOOR_DATA[el('model').value];el('focusdoor').innerHTML=d.doors.map((v,i)=>'<option value="d'+i+'">'+v.id+' · '+v.span.toFixed(2)+'m</option>').concat(d.review.map((v,i)=>'<option value="r'+i+'">'+v.id+' · review: '+v.reason+'</option>')).join('');}const previousModelHandler=el('model').oninput;el('model').oninput=()=>{refreshDoors();previousModelHandler();};el('doors').oninput=draw;el('doorreview').oninput=draw;el('doorfocus').onclick=()=>{const value=el('focusdoor').value,d=DOOR_DATA[el('model').value],review=value[0]==='r',v=(review?d.review:d.doors)[+value.slice(1)];if(!v)return;if(review)el('doorreview').checked=true;const a=v.axis[0],b=v.axis[1];scale=Math.min(canvas.width/(Math.abs(b[0]-a[0])+200),canvas.height/(Math.abs(b[1]-a[1])+200));panX=canvas.width/2-(a[0]+b[0])/2*scale;panY=canvas.height/2-(a[1]+b[1])/2*scale;draw();};refreshDoors();"""
    page=page.replace('window.onresize=resize;resize();',handlers+'window.onresize=resize;resize();')
    page=page.replace('doors remain excluded.','doors are separate coral opening footprints.').replace('Window heights are not inferred.','Window/door heights, hinges and swing directions are not inferred. Coral = fitted door opening; magenta = review seed.')
    page=page.replace('wall + window gap closers','walls, windows and doors')
    return page
