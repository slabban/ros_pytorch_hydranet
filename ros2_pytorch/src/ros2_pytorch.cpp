
//
//  ros2_pytorch.cpp
//
//  Created by Andreas Klintberg on 11/17/18.
//  Updated, modified, and maintained by Samer Labban on 07/13/2022 
//  Copyright © 2022 Samer Labban. All rights reserved.
//

#define DEBUG 1
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

        module_.to(device);

    }
    
    void PyTorchNode::topic_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        at::Tensor image_tensor;
        cv::Mat cv_image;
        ros_to_cv(msg, cv_image);
        cv_to_tensor(cv_image, msg, image_tensor);
        auto inputs = prepare_input(image_tensor);
        predictions preds = predict(inputs);
        #if DEBUG
        print_output(preds);
        #endif
        publish_message();
    }

    void PyTorchNode::ros_to_cv(const sensor_msgs::msg::Image::SharedPtr& msg, cv::Mat& cv_img)
    {
        std::shared_ptr<cv_bridge::CvImage> image_ = cv_bridge::toCvCopy(msg, "rgb8");
        cv_img = image_->image;
    }

    std::vector<torch::jit::IValue> PyTorchNode::prepare_input(at::Tensor& input_tensor)
    {
        input_tensor = input_tensor.to(device).toType(at::kFloat).mul(1.0/255);

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input_tensor);
        return inputs;
    }


    void PyTorchNode::cv_to_tensor(const cv::Mat& img_data, const sensor_msgs::msg::Image::SharedPtr& msg, at::Tensor& input_tensor)
    {
        at::TensorOptions options(at::ScalarType::Byte);
        std::vector<int64_t> sizes = {1, 3, msg->height, msg->width};
        input_tensor= torch::from_blob(img_data.data, at::IntList(sizes), options);
    }

    predictions PyTorchNode::predict(const std::vector<torch::jit::IValue>& inputs)
    {
        auto outputs = module_.forward(inputs).toTuple();
        predictions preds = { };
        preds.segm_out = outputs->elements()[0].toTensor();
        preds.depth_out = outputs->elements()[1].toTensor();
        return preds;
    }

    void PyTorchNode::print_output(const predictions& preds)
    {
        std::ostringstream stream;
        stream << "Segmentation Tensor Size is"<< ' ' <<preds.segm_out.sizes() << '\n' << "Depth Tensor Size is" << ' ' << preds.depth_out.sizes();
        std::string tensor_string = stream.str();

        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", tensor_string.c_str());
    }

    void PyTorchNode::publish_message()
    {
        auto message = std_msgs::msg::String();
        message.data = "Prediction done";
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }

    
