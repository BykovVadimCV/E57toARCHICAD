"""Export decomposition footprints with hosted IFC openings and bounded smoothing."""
from pathlib import Path
import argparse, json
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely import union_all, make_valid, set_precision
import ifcopenshell
import ifcopenshell.api as api
from simscan.netinput import affine
from simscan.export_ifc import _new_file, _placement
from simscan.wall_footprint import regions, primitive


def build(source, out, tolerance_m=None, floor_z=-.02, wall_height=3.02):
    d=json.loads(Path(source).read_text()); px=float(d['frame']['pixel_m'])
    if tolerance_m is None:
        tolerance_m=.0025 if d.get('final_face_gravity',{}).get('accepted') or d.get('dense_face_alignment') else .025
    if not 0<=tolerance_m<=.04:raise ValueError('Smoothing tolerance must be between 0 and 40 mm')
    original=union_all([Polygon(p['polygon'],p.get('holes',[])) for p in d['primitives']])
    # A 5 mm geometry grid suppresses sub-millimetre sliver edges created by
    # intersecting near-coincident raster polygons. Use the same grid for both
    # hosts and their openings so the shared boundaries still match exactly.
    grid=.005/px
    apertures=[(kind,item,set_precision(Polygon(p['polygon'],p.get('holes',[])),grid)) for kind in ('doors','windows') for item in d.get(kind,[]) for p in item['footprint_regions']]
    if any(g.is_empty or g.geom_type!='Polygon' for _,_,g in apertures):raise ValueError('Opening collapsed on precision grid')
    forbidden=union_all([p for _,_,p in apertures])
    # Preserve topology, never simplify directly through an aperture.
    simplified=set_precision(make_valid(original.simplify(tolerance_m/px,preserve_topology=True)),grid).difference(forbidden)
    # Rasterized wall regions can enclose pinholes that become full-height
    # shafts and degenerate triangles on extrusion. Fill only holes smaller
    # than 0.0025 m² that are unrelated to a semantic opening.
    filled=0;clean=[]
    for g in regions(simplified):
        kept=[]
        for ring in g.interiors:
            hole=Polygon(ring)
            if hole.area*px*px < .0025 and hole.intersection(forbidden).area < 1e-8:
                filled+=1
            else:kept.append(ring.coords)
        clean.append(Polygon(g.exterior.coords,kept))
    simplified=union_all(clean).difference(forbidden)
    if not simplified.is_valid or abs(simplified.area-original.area)/original.area > .02:
        raise ValueError('Smoothing changes wall area by more than 2%')
    polys=[];removed_islands=0;removed_slivers=0
    components=regions(simplified)
    major=union_all([g for g in components if g.area*px*px >= .05])
    for g in components:
        x0,y0,x1,y1=g.bounds
        if g.area*px*px < .0001 and max(x1-x0,y1-y0)*px < .05:
            removed_islands+=1
        elif (not major.is_empty and g.area*px*px < .01 and
              max(x1-x0,y1-y0)*px < .6 and 2*g.area/max(g.length,1e-9)*px < .02 and
              g.distance(major)*px < .04):
            # Narrow detached strips alongside a substantial wall are raster
            # fragments. Real short piers must retain a substantial width.
            removed_slivers+=1
        else:polys.append(g)
    simplified=union_all(polys)
    M=np.asarray(affine(d['frame']))
    def world(p):return (M@np.r_[p,1.])[:2].tolist()
    f,ctx,building=_new_file('Good scan - hosted openings')
    storey=api.run('root.create_entity',f,ifc_class='IfcBuildingStorey',name='L1')
    storey.Elevation=float(floor_z);storey.ObjectPlacement=_placement(f,building.ObjectPlacement,(0,0,floor_z))
    api.run('aggregate.assign_object',f,products=[storey],relating_object=building)
    def polyline(coords):
        pts=[world(p) for p in list(coords)[:-1]];pts.append(pts[0])
        return f.createIfcPolyline([f.createIfcCartesianPoint([float(x),float(y)]) for x,y in pts])
    def add(g,kind,name,z,height,container=True):
        ex=polyline(g.exterior.coords);holes=[polyline(r.coords) for r in g.interiors]
        profile=f.createIfcArbitraryProfileDefWithVoids('AREA',None,ex,holes) if holes else f.createIfcArbitraryClosedProfileDef('AREA',None,ex)
        pos=f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([0.,0.,float(z)]),None,None)
        solid=f.createIfcExtrudedAreaSolid(profile,pos,f.createIfcDirection([0.,0.,1.]),float(height))
        rep=f.createIfcShapeRepresentation(ctx['body'],'Body','SweptSolid',[solid])
        obj=api.run('root.create_entity',f,ifc_class=kind,name=name)
        obj.ObjectPlacement=_placement(f,storey.ObjectPlacement)
        obj.Representation=f.createIfcProductDefinitionShape(None,None,[rep])
        if container:api.run('spatial.assign_container',f,products=[obj],relating_structure=storey)
        return obj
    for i,g in enumerate(polys):add(g,'IfcWall',f'Wall region {i+1}',0,wall_height)
    entries=[]
    for kind,item,g in apertures:
        sill,head=(.9,2.1) if kind=='windows' else (0.,2.1)
        if not g.is_valid or g.is_empty:raise ValueError('Invalid opening '+item['id'])
        # Host wall is an extrusion of the opening footprint itself: a proper
        # IFC void cuts out only the door/window height, retaining sill/lintel.
        host=add(g,'IfcWall','Opening host '+item['id'],0,wall_height)
        host.Description='Opening heights are assumed; verify against scan.'
        opening=add(g,'IfcOpeningElement','Opening '+item['id'],sill,head-sill,False)
        api.run('feature.add_feature',f,feature=opening,element=host)
        axis=LineString(item['axis']);thin=g.intersection(axis.buffer(.025/px,cap_style='flat'))
        choices=regions(thin)
        if not choices:raise ValueError('No opening filler footprint for '+item['id'])
        filler=add(max(choices,key=lambda p:p.area),'IfcWindow' if kind=='windows' else 'IfcDoor',item['id'],sill,head-sill)
        filler.Description='Provisional solid panel; sill/head assumed, frame, glazing and swing not surveyed.'
        api.run('feature.add_filling',f,opening=opening,element=filler)
        entries.append(dict(id=item['id'],kind=kind,sill_m=sill,head_m=head,assumed=True,host=host.GlobalId))
    Path(out).parent.mkdir(parents=True,exist_ok=True);f.write(str(out))
    reopened=ifcopenshell.open(str(out))
    assert len(reopened.by_type('IfcOpeningElement'))==len(entries)
    assert len(reopened.by_type('IfcRelVoidsElement'))==len(entries)
    assert len(reopened.by_type('IfcRelFillsElement'))==len(entries)
    def vertices(g):return sum(len(p.exterior.coords)+sum(len(h.coords) for h in p.interiors) for p in regions(g))
    report={'source':str(source),'ifc':str(out),'smoothing_tolerance_m':tolerance_m,
            'area_before_m2':original.area*px*px,'area_after_m2':simplified.area*px*px,
            'changed_area_m2':original.symmetric_difference(simplified).area*px*px,
            'vertices_before':vertices(original),'vertices_after':vertices(simplified), 'small_holes_filled':filled,'tiny_islands_removed':removed_islands,'detached_slivers_removed':removed_slivers,'precision_grid_m':.005,
            'wall_regions':len(polys),'hosted_openings':entries,
            'height_note':'Storey height and opening sill/head are assumptions, not measured.',
            'coverage_note':'Existing saved fit was smoothed; no new lower-target refit was performed.'}
    Path(out).with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('out');p.add_argument('--smoothing-m',type=float)
    p.add_argument('--floor-z',type=float,default=0.);p.add_argument('--wall-height',type=float,default=2.7)
    a=p.parse_args();print(json.dumps(build(a.source,a.out,a.smoothing_m,a.floor_z,a.wall_height),indent=2))
