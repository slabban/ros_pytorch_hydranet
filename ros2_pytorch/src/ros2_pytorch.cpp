
//
//  ros2_pytorch.cpp
//
//  Created by Andreas Klintberg on 11/17/18.
//  Updated, modified, and maintained by Samer Labban on 07/13/2022 
//  Copyright © 2022 Samer Labban. All rights reserved.
//

#include "ros2_pytorch.hpp"

using std::placeholders::_1;



PyTorchNode::PyTorchNode() : Node("pytorch_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/image_raw", 10, std::bind(&PyTorchNode::topic_callback, this, _1));
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic_out", 10);

        this->declare_parameter("GPU", 0);
        int cuda = this->get_parameter("GPU").as_int();

        if(cuda>0){
            device = at::kCUDA;
            
        }
        
        try 
        {
            module_ = torch::jit::load("/home/ros2_ws/src/ros2_pytorch/traced_hydranet.pt");

        }
        catch (const c10::Error& e)
        {
            throw std::invalid_argument("Failed to load model: " + e.msg());
        }

    }
    
    void PyTorchNode::topic_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
    
        cv::Mat imageMat = ros_to_cv(msg);

        prepare_image(imageMat);

        at::Tensor tensor_image = cv_to_tensor(imageMat, msg);

        
        std::vector<torch::jit::IValue> inputs;

        module_.to(device);
        
        tensor_image = tensor_image.to(device).toType(at::kFloat);
        inputs.emplace_back(tensor_image);


        // Execute the model and turn its output into a tuple.
        auto outputs = module_.forward(inputs).toTuple();

        auto segm_out = outputs->elements()[0].toTensor();
        auto depth_out = outputs->elements()[1].toTensor();

        std::ostringstream stream;
        stream << "Segmentation Tensor Size is"<< ' ' <<segm_out.sizes() << '\n' << "Depth Tensor Size is" << ' ' << depth_out.sizes();
        std::string tensor_string = stream.str();

        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", tensor_string.c_str());
        
        auto message = std_msgs::msg::String();
        message.data = "Prediction done";
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
        
    }

    cv::Mat PyTorchNode::ros_to_cv(const sensor_msgs::msg::Image::SharedPtr& msg)
    {
        std::shared_ptr<cv_bridge::CvImage> image_ = cv_bridge::toCvCopy(msg, "rgb8");
        return image_->image;
    }

    void PyTorchNode::prepare_image(cv::Mat& img_data)
    {
        img_data *= (1.0/255);
    }

    at::Tensor PyTorchNode::cv_to_tensor(const cv::Mat& img_data, const sensor_msgs::msg::Image::SharedPtr& msg)
    {
        at::TensorOptions options(at::ScalarType::Byte);
        std::vector<int64_t> sizes = {1, 3, msg->height, msg->width};
        at::Tensor tensor_image = torch::from_blob(img_data.data, at::IntList(sizes), options);
        return tensor_image;
    }

    
