//
//  ros2_pytorch.hpp
//
//  Created by Samer Labban on 07/13/2022 
//  Copyright © 2022 Samer Labban. All rights reserved.
//

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <iostream>

#include <torch/script.h> // One-stop header.
#include <ATen/ATen.h>


#include "rclcpp/rclcpp.hpp"
#include "rcpputils/asserts.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <eigen/unsupported/Eigen/CXX11/Tensor>



typedef struct predictions
{
    at::Tensor segm_out;
    at::Tensor depth_out;
} predictions;





class PyTorchNode : public rclcpp::Node
{
public:
    PyTorchNode();

    
private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    cv::Mat ros_to_cv(const sensor_msgs::msg::Image::SharedPtr& msg, cv::Mat& cv_img);
    std::vector<torch::jit::IValue> prepare_input(at::Tensor& input_tensor);
    void cv_to_tensor(const cv::Mat& img_data, const sensor_msgs::msg::Image::SharedPtr& msg, at::Tensor& input_tensor);
    predictions predict(const std::vector<torch::jit::IValue>& inputs);
    void print_output(const predictions& preds);
    cv::Mat segm_to_cv(at::Tensor& segm, const cv::Mat& msg);
    cv::Mat depth_to_cv(at::Tensor& depth, const cv::Mat& msg);
    void publish_segmentation_image(cv::Mat& segm);
    void publish_depth_image(cv::Mat& depth);
    

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_img_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmentation_img_publisher_;
    torch::jit::script::Module module_;
    c10::DeviceType device = at::kCPU;

    std::array<std::array<int,3>, 40> cmap = {{ 
    { 128, 128, 192},
    {128, 0, 0},
    {0, 128,   0},
    {128, 128,   0},
    {0,   0, 128},
    {128,   0, 128},
    {0, 128, 128},
    {128, 128, 128},
    {64,   0,   0},
    {192,  0,   0},
    {64, 128,   0},
    {192, 128,   0},
    {64,  0, 128},
    {192,  0, 128},
    {64, 128, 128},
    {192, 128, 128},
    {0,   64,   0},
    {128,  64,   0},
    {0, 192,   0},
    {128, 192,   0},
    {0,  64, 128},
    {128,  64, 128},
    {0, 192, 128},
    {128, 192, 128},
    {64,  64,   0},
    {192, 64,   0},
    {64,  192,   0},
    {192, 192,   0},
    {64,  64, 128},
    {192,  64, 128},
    {64, 192, 128},
    {192, 192, 128},
    {0,   0,  64},
    {128,  0,  64},
    {0, 128,  64},
    {128, 128,  64},
    {0,   0, 192},
    {128, 0, 192},
    {0, 128, 192},
    {0, 0, 0}
    }};
    

};
