
//
//  ros2_pytorch.cpp
//
//  Created by Andreas Klintberg on 11/17/18.
//  Updated, modified, and maintained by Samer Labban on 07/13/2022 
//  Copyright © 2022 Samer Labban. All rights reserved.
//

#define DEBUG 0
#include "ros2_pytorch.hpp"

using std::placeholders::_1;



PyTorchNode::PyTorchNode() : Node("pytorch_node")
    {
        // /front_camera/image_decompressed
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/front_camera/image_decompressed", 1, std::bind(&PyTorchNode::topic_callback, this, _1));
        segmentation_img_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("segm_image", 1);
        depth_img_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 1);

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
        input_img_cv_ = ros_to_cv(msg);
        at::Tensor image_tensor = cv_to_tensor();
        auto inputs = prepare_input(image_tensor);
        predictions preds = predict(inputs);
        
        segm_img_cv_ = segm_to_cv(preds.segm_out);
        depth_img_cv_ = depth_to_cv(preds.depth_out);
        
        publish_segmentation_image(segm_img_cv_);
        publish_depth_image(depth_img_cv_);

        #if DEBUG
        print_output(preds);
        #endif

    }

    cv::Mat PyTorchNode::ros_to_cv(const sensor_msgs::msg::Image::SharedPtr& msg)
    {
        std::shared_ptr<cv_bridge::CvImage> image_ = cv_bridge::toCvCopy(msg, "bgr8");
        cv::Mat cv_img = image_->image;
        cv::cvtColor(cv_img, cv_img, CV_BGR2RGB);
        cv::resize(cv_img, cv_img, cv::Size(msg->width, msg->height));
        cv_img.convertTo(cv_img, CV_32FC3, 1.0f/255.0f);
        return cv_img;
    }

    std::vector<torch::jit::IValue> PyTorchNode::prepare_input(at::Tensor& input_tensor)
    {
        input_tensor = input_tensor.to(device);

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input_tensor);
        return inputs;
    }


    at::Tensor PyTorchNode::cv_to_tensor()
    {
        auto n_channels = input_img_cv_.channels();
        at::Tensor input_tensor = torch::from_blob(input_img_cv_.data, {1, input_img_cv_.rows, input_img_cv_.cols, n_channels});
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        input_tensor[0][0].sub(0.485).div(0.229);
        input_tensor[0][1].sub(0.456).div(0.224);
        input_tensor[0][2].sub(0.406).div(0.225);
        return input_tensor;

    }

    predictions PyTorchNode::predict(const std::vector<torch::jit::IValue>& inputs)
    {
        auto outputs = module_.forward(inputs).toTuple();
        predictions preds = {};
        preds.segm_out = outputs->elements()[0].toTensor().squeeze().permute({1,2,0}).argmax(2).toType(torch::kUInt8).detach().cpu();
        preds.depth_out = outputs->elements()[1].toTensor().squeeze().detach().cpu();
        return preds;
    }

    void PyTorchNode::print_output(const predictions& preds)
    {
        std::ostringstream stream;
        stream << "Segmentation Tensor Size is"<< ' '  <<preds.segm_out.sizes() << '\n' << "Depth Tensor Size is" << ' ' << preds.depth_out.sizes() ;
        std::string tensor_string = stream.str();

        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", tensor_string.c_str());

        // Popup images for debugging
        cv::namedWindow("DEPTH", cv::WINDOW_AUTOSIZE);
        cv::imshow("DEPTH", depth_img_cv_);
        cv::namedWindow("Segmentation", cv::WINDOW_AUTOSIZE);
        cv::imshow("Segmentation", segm_img_cv_);
        cv::waitKey(1);
    }

    void PyTorchNode::publish_segmentation_image(cv::Mat& segm){
        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::TYPE_8UC3, segm).toImageMsg();
        segmentation_img_publisher_->publish(*msg.get());
    }

    void PyTorchNode::publish_depth_image(cv::Mat& depth)
    {   
        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::TYPE_8UC3, depth).toImageMsg();
        depth_img_publisher_->publish(*msg.get());

    }

    cv::Mat PyTorchNode::segm_to_cv(at::Tensor& segm)
    {   
        int64_t height = segm.sizes()[0];
        int64_t width = segm.sizes()[1];
        cv::Mat output_mat(height, width, CV_8U, segm.data_ptr<uint8_t>());
        cv::Size original_size = cv::Size(input_img_cv_.cols, input_img_cv_.rows);
        cv::resize(output_mat, output_mat, original_size,cv::INTER_CUBIC);
        cv::Mat mask(output_mat.size(), CV_8UC3);

        for(int j=0; j<mask.rows; j++)
        {
            for(int i=0; i<mask.cols; i++)
            {   
                uint8_t color_source= output_mat.at<uint8_t>(cv::Point(i,j));
                mask.at<cv::Vec3b>(j,i)[0] = cmap[color_source][0];
                mask.at<cv::Vec3b>(j,i)[1] = cmap[color_source][1];
                mask.at<cv::Vec3b>(j,i)[2] = cmap[color_source][2];
            }
        }
        return mask;
    }

    cv::Mat PyTorchNode::depth_to_cv(at::Tensor& depth)
    { 
        int64_t height = depth.sizes()[0];
        int64_t width = depth.sizes()[1];
        cv::Mat output_mat(height, width, CV_32FC1, depth.data_ptr<float>());
        cv::Size original_size = cv::Size(input_img_cv_.cols, input_img_cv_.rows);
        cv::resize(output_mat, output_mat, original_size,cv::INTER_CUBIC);
        double min_val, max_val;
        cv::Mat depth_visual;
        cv::minMaxLoc(output_mat, &min_val, &max_val);
        output_mat = 255 * (output_mat - min_val) / (max_val - min_val);
        output_mat.convertTo(depth_visual, CV_8U);
        cv::applyColorMap(depth_visual, depth_visual, cv::COLORMAP_MAGMA); 
        return depth_visual;
    }

    
