"""Focus recovered short piers/returns in the existing native plan viewer."""
from pathlib import Path
import json,base64
import numpy as np


def decorate(page,folder):
    folder=Path(folder);data={}
    for model in ['new_wall','new_plan']:
        path=folder/(model+'_decomposition.json')
        if not path.exists():return page
        d=json.loads(path.read_text());recovery=d.get('small_feature_recovery')
        if recovery is None:return page
        features=[]
        for i,edit in enumerate(recovery['edits']):
            line=edit.get('line')
            if line is None:continue
            u=np.array(line['u']);v=np.array(line['normal']);axis=[(u*line[t]+v*line['offset']).tolist() for t in ['t0','t1']]
            features.append(dict(id=f'P{i+1}',axis=axis,length=edit['length_m']))
        image=folder/(model+'_small_feature_mask.png')
        data[model]=dict(features=features,mask=('data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()) if image.exists() else None)
    setup='const SMALL_FEATURE_DATA='+json.dumps(data,separators=(',',':'))+";for(const [m,d] of Object.entries(SMALL_FEATURE_DATA)){if(d.mask)DATA.models[m].images.small_features=d.mask;DATA.models[m].note+='; '+d.features.length+' short sections recovered';}\n"
    page=page.replace('const el=id=>',setup+'const el=id=>')
    page=page.replace('<label>Vectors','<select id="focusfeature"></select><button id="featurefocus">Focus short section</button><label>Vectors')
    page=page.replace('<option value="original">','<option value="small_features">Recovered short sections (green)</option><option value="original">')
    handlers="""function refreshFeatures(){el('focusfeature').innerHTML=SMALL_FEATURE_DATA[el('model').value].features.map((v,i)=>'<option value="'+i+'">'+v.id+' · '+v.length.toFixed(2)+'m</option>').join('');}const featureModelHandler=el('model').oninput;el('model').oninput=()=>{refreshFeatures();featureModelHandler();};el('featurefocus').onclick=()=>{const v=SMALL_FEATURE_DATA[el('model').value].features[+el('focusfeature').value];if(!v)return;const a=v.axis[0],b=v.axis[1];scale=Math.min(canvas.width/(Math.abs(b[0]-a[0])+180),canvas.height/(Math.abs(b[1]-a[1])+180));panX=canvas.width/2-(a[0]+b[0])/2*scale;panY=canvas.height/2-(a[1]+b[1])/2*scale;draw();};refreshFeatures();"""
    return page.replace('window.onresize=resize;resize();',handlers+'window.onresize=resize;resize();')
