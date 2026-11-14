"""Audit split grouping, native payload hashes, counts and completion before training."""
from pathlib import Path
import argparse,hashlib,json
from simscan.dataset_contract import complete_level
from simscan.hdmask import complete_hd_level


def audit(root,deep=True):
    root=Path(root);membership={};groups={};errors=[];levels=0;native_bytes=0
    from simscan.semantic_contract import COARSE_LABELS_VERSION, COARSE_META_VERSION, OPENING_SEMANTICS_VERSION
    meta=json.loads((root/'meta.json').read_text(encoding='utf-8'))
    if (meta.get('version')!=COARSE_META_VERSION or
            meta.get('labels')!=COARSE_LABELS_VERSION or
            meta.get('opening_semantics')!=OPENING_SEMANTICS_VERSION):
        errors.append('Dataset uses obsolete opening semantics; regenerate labels')
    for split in ['train','val','test']:
        entries=(root/'splits'/(split+'.txt')).read_text(encoding='utf-8').split()
        for entry in entries:
            if entry in membership:errors.append('Repeated level: '+entry)
            membership[entry]=split;folder=root/'scenes'/entry
            try:
                scene=json.loads((folder/'scene.json').read_text(encoding='utf-8'));key=scene['layout_key']
                if (scene.get('labels')!=COARSE_LABELS_VERSION or
                        scene.get('opening_semantics')!=OPENING_SEMANTICS_VERSION):
                    errors.append('Wrong scene opening contract: '+entry)
                if key in groups and groups[key]!=split:errors.append('Layout crosses splits: '+key)
                groups[key]=split
                if scene.get('floor_void_policy')!='physical_missing_floor_only_v2':errors.append('Wrong floor-void policy: '+entry)
                hd=json.loads((folder/'wall_mask_hd.json').read_text(encoding='utf-8'))
                if not complete_level(folder) or not complete_hd_level(folder,hd['frame']['pixel_m']*1000,hd['min_size_px']):
                    errors.append('Incomplete level: '+entry);continue
                marker=json.loads((folder/'hd_complete.json').read_text(encoding='utf-8'))
                for name,item in marker['files'].items():
                    native_bytes+=item['bytes']
                    if deep:
                        h=hashlib.sha256()
                        with (folder/name).open('rb') as f:
                            for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
                        if h.hexdigest()!=item['sha256']:errors.append('Checksum mismatch: '+entry+'/'+name)
                levels+=1
            except (OSError,ValueError,KeyError) as exc:errors.append(entry+': '+str(exc))
    actual={f.parent.relative_to(root/'scenes').as_posix() for f in (root/'scenes').glob('*/L*/scene.json')}
    if actual!=set(membership):errors.append('Split lists do not cover exactly the generated levels')
    if not levels:errors.append('No complete native levels')
    report=dict(levels=levels,layout_groups=len(groups),native_bytes=native_bytes,
                splits={s:sum(v==s for v in membership.values()) for s in ['train','val','test']},errors=errors,sha256_checked=deep)
    (root/'hd_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if errors:raise ValueError(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--quick',action='store_true');a=p.parse_args()
    print(json.dumps(audit(a.root,not a.quick),indent=2))
