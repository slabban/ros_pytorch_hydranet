
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
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/image_raw", 1, std::bind(&PyTorchNode::topic_callback, this, _1));
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("topic_out", 1);

        this->declare_parameter("GPU", 0);
        int cuda = this->get_parameter("GPU").as_int();

        if(cuda>0){
            device = at::kCUDA;
        }
        
        try 
        {
            module_ = torch::jit::load("/home/ros2_ws/src/ros2_pytorch/traced_nyud_hydranet.pt");

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
        
        cv::Mat cv_out = ros_to_cv(msg, cv_image);
        cv_to_tensor(cv_out, msg, image_tensor);
        auto inputs = prepare_input(image_tensor);
        predictions preds = predict(inputs);
        #if DEBUG
        print_output(preds);
        #endif
        cv::Mat output_depth_Mat = depth_to_cv(preds.depth_out, cv_out);
        publish_message(output_depth_Mat);
        double min_val, max_val;
        cv::Mat depth_visual;
        cv::minMaxLoc(output_depth_Mat, &min_val, &max_val);
        output_depth_Mat = 255 * (output_depth_Mat - min_val) / (max_val - min_val);
        output_depth_Mat.convertTo(depth_visual, CV_8U);
        cv::applyColorMap(depth_visual, depth_visual, 4); 
        cv::namedWindow("FULL", cv::WINDOW_AUTOSIZE);
        cv::imshow("FULL", depth_visual);
        cv::waitKey(1);
    }

    cv::Mat PyTorchNode::ros_to_cv(const sensor_msgs::msg::Image::SharedPtr& msg, cv::Mat& cv_img)
    {
        std::cout << msg->width;
        std::shared_ptr<cv_bridge::CvImage> image_ = cv_bridge::toCvCopy(msg, "bgr8");
        cv_img = image_->image;
        cv::Mat cv_out;
        cv::cvtColor(cv_img, cv_out, CV_BGR2RGB);
        cv::resize(cv_out, cv_out, cv::Size(msg->width, msg->height));
        //cv::Mat input_cv_ = cv::Mat(msg->height, msg->width, CV_32FC3);
        cv_out.convertTo(cv_out, CV_32FC3, 1.0f/255.0f);
        return cv_out;
    }

    std::vector<torch::jit::IValue> PyTorchNode::prepare_input(at::Tensor& input_tensor)
    {
        input_tensor = input_tensor.to(device);

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input_tensor);
        return inputs;
    }


    void PyTorchNode::cv_to_tensor(const cv::Mat& img_data, const sensor_msgs::msg::Image::SharedPtr& msg, at::Tensor& input_tensor)
    {

        auto n_channels = img_data.channels();
        input_tensor = torch::from_blob(img_data.data, {1, img_data.rows, img_data.cols, n_channels});
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        input_tensor[0][0].sub(0.485).div(0.229);
        input_tensor[0][1].sub(0.456).div(0.224);
        input_tensor[0][2].sub(0.406).div(0.225);

    }

    predictions PyTorchNode::predict(const std::vector<torch::jit::IValue>& inputs)
    {
        auto outputs = module_.forward(inputs).toTuple();
        predictions preds = { };
        preds.segm_out = outputs->elements()[0].toTensor().cpu();
        preds.depth_out = outputs->elements()[1].toTensor().squeeze().cpu();
        return preds;
    }

    void PyTorchNode::print_output(const predictions& preds)
    {
        std::ostringstream stream;
        stream << "Segmentation Tensor Size is"<< ' ' <<preds.segm_out.sizes() << '\n' << "Depth Tensor Size is" << ' ' << preds.depth_out.sizes() ;
        //"and of type" << ' ' << preds.depth_out.type();
        std::string tensor_string = stream.str();

        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", tensor_string.c_str());
    }

    void PyTorchNode::publish_message(cv::Mat& depth)
    {   
        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::TYPE_32FC1, depth).toImageMsg();
        publisher_->publish(*msg.get());

    }

    cv::Mat PyTorchNode::depth_to_cv(at::Tensor& depth, const cv::Mat& msg)
    { 
        int64_t height = depth.sizes()[0];
        int64_t width = depth.sizes()[1];
        cv::Mat output_mat(height, width, CV_32FC1, depth.data_ptr<float>());
        cv::Mat output_cv_ ;
        cv::Size original_size = cv::Size(msg.cols, msg.rows);
        cv::resize(output_mat, output_cv_, original_size,cv::INTER_CUBIC);
        return output_cv_.clone();
    }

    
