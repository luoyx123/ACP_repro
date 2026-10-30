import numpy as np
from acp.robot.collision import joint_housing_overlap


def test_allowance_is_limited_to_named_pair_depth_and_joint_neighbourhood():
    rule={'body_pair':['knee','ankle'],'joint':'ankle_pitch',
          'maximum_penetration_m':.04,'maximum_contact_point_distance_to_joint_m':.035}
    anchors={'ankle_pitch':np.zeros(3)}
    assert joint_housing_overlap('ankle','knee',[.02,0,0],-.02,anchors,[rule])
    assert not joint_housing_overlap('hand','knee',[.02,0,0],-.02,anchors,[rule])
    assert not joint_housing_overlap('ankle','knee',[.06,0,0],-.02,anchors,[rule])
    assert not joint_housing_overlap('ankle','knee',[.02,0,0],-.06,anchors,[rule])
