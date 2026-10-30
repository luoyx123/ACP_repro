import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from acp.data.adapters.php import read_motion, terrain_variants, write_terrain


def test_php_named_joint_permutation_preserves_root_and_target_angles(tmp_path):
    p=tmp_path/'motion.npz'
    q=np.array([[1,2,3,1,0,0,0,.8,.2],[4,5,6,1,0,0,0,.7,.3]])
    np.savez(p,joint_pos=q,joint_names=['right','left'],fps=[50])
    canonical,fps=read_motion(p,['left','right'])
    np.testing.assert_array_equal(canonical[:,:4],q[:,3:7])
    np.testing.assert_array_equal(canonical[:,4:7],q[:,:3])
    np.testing.assert_array_equal(canonical[:,7:],q[:,[8,7]])
    assert fps==50
    with pytest.raises(ValueError,match='joint-name'):
        read_motion(p,['left','missing'])


def test_php_box_axis_full_extents_and_rotation(tmp_path):
    boxes=np.zeros((2,10,10)); boxes[:,:,3]=1; boxes[:,:,7:]=[2,3,4]
    boxes[1,7,:3]=[5,6,7]
    boxes[1,7,3:7]=[np.sqrt(.5),0,0,np.sqrt(.5)]
    source=tmp_path/'terrain.npy';np.save(source,boxes)
    loaded=terrain_variants(source)
    assert loaded.shape==(2,10,10)
    path=tmp_path/'terrain.urdf';write_terrain(loaded[:,7,:],path)
    root=ET.parse(path).getroot()
    np.testing.assert_allclose(np.fromstring(root.findall('joint')[1].find('origin').get('xyz'),sep=' '),[5,6,7])
    np.testing.assert_array_equal(np.fromstring(root.findall('link')[1].find('collision/geometry/box').get('size'),sep=' '),[2,3,4])
    rpy=np.fromstring(root.findall('joint')[1].find('origin').get('rpy'),sep=' ')
    np.testing.assert_allclose(Rotation.from_euler('xyz',rpy).apply([1,0,0]),[0,1,0],atol=1e-15)
    np.save(source,np.zeros((0,10,10)))
    assert terrain_variants(source).shape==(0,10,10)
