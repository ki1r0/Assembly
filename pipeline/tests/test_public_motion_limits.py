from robot_kinematics import public_motion_reference_limits


def test_torque_controller_does_not_claim_joint_drive_limits():
    for mode in ['isaac_osc', 'factory', 'factory_damped']:
        limits = public_motion_reference_limits(mode)
        assert limits['joint_target_increment_rad_per_control'] is None
        assert limits['joint_reference_tracking_lead_limit_rad'] is None
        assert limits['joint_integral_bias_limit_rad'] is None


def test_reference_limits_match_actual_cartesian_reference():
    for mode in ['isaac_osc', 'reference_ik_high_pd', 'reference_ik']:
        limits = public_motion_reference_limits(mode)
        assert limits['reference_cartesian_speed_mps'] == .010
        assert limits['reference_angular_speed_radps'] == .060
        assert limits['reference_rotation_tracking_pause_rad'] == .02
    assert public_motion_reference_limits('reference_ik')['joint_target_increment_rad_per_control'] == .0025
