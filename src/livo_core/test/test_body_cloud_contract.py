from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_body_cloud_publisher_contract():
    header = (ROOT / "include" / "LIVMapper.h").read_text()
    source = (ROOT / "src" / "LIVMapper.cpp").read_text()

    assert "pubLaserCloudBody" in header
    assert "publish_frame_body" in header
    assert '"/cloud_undistorted_body"' in source
    assert "publish_frame_body(pubLaserCloudBody)" in source
    assert "void LIVMapper::publish_frame_body" in source
    assert "void transformLidarToBody" in header
    assert "void LIVMapper::transformLidarToBody" in source
    assert "dense_map_en ? feats_undistort : feats_down_body" in source
    assert "transformLidarToBody(*published_cloud, cloud_body)" in source
    assert "point_body = extR * point_lidar + extT" in source
    assert "pcl::toROSMsg(cloud_body" in source
    assert 'laserCloudmsg.header.frame_id = local_fast_mode_ ? "local_body" : "aft_mapped"' in source


def test_livo_outputs_use_measurement_stamp():
    header = (ROOT / "include" / "LIVMapper.h").read_text()
    source = (ROOT / "src" / "LIVMapper.cpp").read_text()

    assert "rclcpp::Time current_lio_stamp() const" in header
    assert "LidarMeasures.last_lio_update_time" in source
    assert "laserCloudmsg.header.stamp = current_lio_stamp()" in source
    assert "odomAftMapped.header.stamp = current_lio_stamp()" in source
    assert "msg_body_pose.header.stamp = current_lio_stamp()" in source
