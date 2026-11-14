import numpy as np
import cv2
from simscan.decompose import rectangle,arc,rasterize,_local_mask,rectangle_overlap_metrics
from simscan.wall_merge import merge_primitives,polygon_union,merge_result


def union(parts,shape):
 m=np.zeros(shape,bool)
 for p in parts:m|=rasterize(p,shape)
 return m


def test_same_wall_merges_without_boundary_change_and_keeps_offsets():
 parts=[rectangle([60,50],80,10,0),rectangle([130,50.2],70,10.2,.002)]
 shape=(120,200);merged,stat=merge_primitives(parts,shape,.005)
 assert len(merged)==1 and stat['removed_seams']==1
 assert np.array_equal(union(parts,shape),union(merged,shape))
 assert len(merged[0]['parts'])==2
 assert merged[0]['width_range_m'][1]>merged[0]['width_range_m'][0]
 # Verify the displayed analytic outline independently at off-boundary samples.
 p=merged[0]
 for y in np.arange(40.3,61,1.7):
  for x in np.arange(15.3,170,2.7):
   old=any(cv2.pointPolygonTest(np.array(c['polygon'],np.float32),(float(x),float(y)),False)>=0 for c in parts)
   new=cv2.pointPolygonTest(np.array(p['polygon'],np.float32),(float(x),float(y)),False)>=0
   assert old==new


def test_aperture_gap_width_step_and_corner_do_not_merge():
 base=rectangle([50,50],60,10,0)
 for other in [rectangle([111,50],60,10,0),rectangle([110,50],60,18,0),rectangle([80,80],60,10,np.pi/2)]:
  merged,_=merge_primitives([base,other],(160,180),.005)
  assert len(merged)==2


def test_shared_edge_is_mergeable_but_point_contact_is_not():
 a=rectangle([40,40],40,10,0);b=rectangle([80,40],40,10,0)
 assert len(merge_primitives([a,b],(120,140),.005)[0])==1
 assert polygon_union([a['polygon'],rectangle([80,50],40,10,0)['polygon']]) is None


def test_hole_outline_and_world_export_remain_consistent():
 # Test the arrangement with a hole, independent of same-width grouping policy.
 parts=[rectangle([60,20],100,10,0),rectangle([60,100],100,10,0),rectangle([15,60],90,10,np.pi/2),rectangle([105,60],90,10,np.pi/2)]
 boundary=polygon_union([p['polygon'] for p in parts]);assert boundary is not None and len(boundary['holes'])==1
 pair=[rectangle([40,40],40,10,0),rectangle([80,40],40,10,0)];shape=(120,140);mask=union(pair,shape)
 result=merge_result(dict(primitives=pair,mask=mask,pixel_to_world=[[.005,0,10],[0,-.005,20]],metrics={}),.005)
 assert result['world_regions'][0]['kind']=='wall_run'
 assert np.array_equal(_local_mask(result['primitives'][0],np.array([10,10]),(80,110)),mask[10:90,10:120])
 assert rectangle_overlap_metrics(pair,shape,.005)==rectangle_overlap_metrics(result['primitives'],shape,.005)


def test_arc_is_preserved_and_repeated_merge_is_stable():
 parts=[rectangle([40,40],40,10,0),rectangle([80,40],40,10,0),arc([70,70],30,8,0,1)]
 merged,_=merge_primitives(parts,(150,150),.005)
 assert any(c['kind']=='arc' for c in merged)
 assert len(merge_primitives(merged,(150,150),.005)[0])==len(merged)


def test_contained_cross_tile_is_absorbed_but_real_branch_survives():
 a=rectangle([40,40],40,10,0);b=rectangle([80,40],40,10,0)
 tile=rectangle([60,40],8,3,np.pi/2);branch=rectangle([80,60],40,8,np.pi/2)
 merged,stats=merge_primitives([a,b,tile,branch],(120,140),.005)
 assert len(merged)==2 and stats['absorbed_redundant_tiles']==1
 assert np.array_equal(union([a,b,tile,branch],(120,140)),union(merged,(120,140)))
 assert any(c['kind']=='rectangle' for c in merged)
