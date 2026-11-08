#include "LIVMapper.h"

int main(int argc, char **argv)
{
  // Keep each Eigen/OpenCV reduction serial for repeatable replay results.
  // Independent validation, image statistics and point matching use at most
  // one helper per stage; do not enable nested library thread pools.
  Eigen::setNbThreads(1);
  cv::setNumThreads(1);

  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);

  rclcpp::Node::SharedPtr nh;
  image_transport::ImageTransport it_(nh);
  LIVMapper mapper(nh, "laserMapping", options);
  mapper.initializeSubscribersAndPublishers(nh, it_);
  mapper.run(nh);
  rclcpp::shutdown();
  return 0;
}
