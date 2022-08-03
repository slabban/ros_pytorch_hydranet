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

// TODO: Add cmap for segmentation 



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
    void publish_depth_image(cv::Mat& depth);
    // TODO: create segmentation map from segm output
    cv::Mat depth_to_cv( at::Tensor& depth, const cv::Mat& msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    torch::jit::script::Module module_;
    c10::DeviceType device = at::kCPU;
    

};
