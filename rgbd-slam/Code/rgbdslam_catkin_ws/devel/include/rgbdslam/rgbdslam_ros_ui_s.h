// Generated by gencpp from file rgbdslam/rgbdslam_ros_ui_s.msg
// DO NOT EDIT!


#ifndef RGBDSLAM_MESSAGE_RGBDSLAM_ROS_UI_S_H
#define RGBDSLAM_MESSAGE_RGBDSLAM_ROS_UI_S_H

#include <ros/service_traits.h>


#include <rgbdslam/rgbdslam_ros_ui_sRequest.h>
#include <rgbdslam/rgbdslam_ros_ui_sResponse.h>


namespace rgbdslam
{

struct rgbdslam_ros_ui_s
{

typedef rgbdslam_ros_ui_sRequest Request;
typedef rgbdslam_ros_ui_sResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct rgbdslam_ros_ui_s
} // namespace rgbdslam


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::rgbdslam::rgbdslam_ros_ui_s > {
  static const char* value()
  {
    return "406bad1a44daaa500258274f332bb924";
  }

  static const char* value(const ::rgbdslam::rgbdslam_ros_ui_s&) { return value(); }
};

template<>
struct DataType< ::rgbdslam::rgbdslam_ros_ui_s > {
  static const char* value()
  {
    return "rgbdslam/rgbdslam_ros_ui_s";
  }

  static const char* value(const ::rgbdslam::rgbdslam_ros_ui_s&) { return value(); }
};


// service_traits::MD5Sum< ::rgbdslam::rgbdslam_ros_ui_sRequest> should match 
// service_traits::MD5Sum< ::rgbdslam::rgbdslam_ros_ui_s > 
template<>
struct MD5Sum< ::rgbdslam::rgbdslam_ros_ui_sRequest>
{
  static const char* value()
  {
    return MD5Sum< ::rgbdslam::rgbdslam_ros_ui_s >::value();
  }
  static const char* value(const ::rgbdslam::rgbdslam_ros_ui_sRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::rgbdslam::rgbdslam_ros_ui_sRequest> should match 
// service_traits::DataType< ::rgbdslam::rgbdslam_ros_ui_s > 
template<>
struct DataType< ::rgbdslam::rgbdslam_ros_ui_sRequest>
{
  static const char* value()
  {
    return DataType< ::rgbdslam::rgbdslam_ros_ui_s >::value();
  }
  static const char* value(const ::rgbdslam::rgbdslam_ros_ui_sRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::rgbdslam::rgbdslam_ros_ui_sResponse> should match 
// service_traits::MD5Sum< ::rgbdslam::rgbdslam_ros_ui_s > 
template<>
struct MD5Sum< ::rgbdslam::rgbdslam_ros_ui_sResponse>
{
  static const char* value()
  {
    return MD5Sum< ::rgbdslam::rgbdslam_ros_ui_s >::value();
  }
  static const char* value(const ::rgbdslam::rgbdslam_ros_ui_sResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::rgbdslam::rgbdslam_ros_ui_sResponse> should match 
// service_traits::DataType< ::rgbdslam::rgbdslam_ros_ui_s > 
template<>
struct DataType< ::rgbdslam::rgbdslam_ros_ui_sResponse>
{
  static const char* value()
  {
    return DataType< ::rgbdslam::rgbdslam_ros_ui_s >::value();
  }
  static const char* value(const ::rgbdslam::rgbdslam_ros_ui_sResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // RGBDSLAM_MESSAGE_RGBDSLAM_ROS_UI_S_H