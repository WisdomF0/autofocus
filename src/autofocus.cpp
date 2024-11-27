#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <cmath>

class ImageFocusEvaluator {
public:
    ImageFocusEvaluator() {
        // 初始化订阅器，订阅图像话题
        nh_.param<std::string>("test_method", test_method_, "laplacian");
        nh_.param<std::string>("sub_name", sub_name_, "/camera/image_raw");
        nh_.param<int>("chessboard_rows", chessboard_rows_, 8);  // 默认棋盘格6行
        nh_.param<int>("chessboard_cols", chessboard_cols_, 11);  // 默认棋盘格9列

        // 打印节点名称
        ROS_INFO("Subscribe Name: %s", sub_name_.c_str());
        ROS_INFO("Method: %s", test_method_.c_str());

        image_sub_ = nh_.subscribe(sub_name_.c_str(), 1, &ImageFocusEvaluator::imageCallback, this);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    std::string test_method_;  // 存储传入的测试方法
    std::string sub_name_;    // 存储节点名称
    int chessboard_rows_;     // 棋盘格的行数
    int chessboard_cols_;     // 棋盘格的列数

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            // 转换 ROS 图像消息为 OpenCV 图像
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat image = cv_ptr->image;

            // 根据传入的测试方法选择计算方式
            double score = 0.0;
            if (test_method_ == "laplacian") {
                score = calculateLaplacian(image);
                displayResult(image, score);
            } else if (test_method_ == "sfr") {
                cv::Rect roi;
                score = calculateSFR(image, roi);
                displayResult(image, score, roi);
            } else {
                ROS_WARN("Unknown test method: %s. Defaulting to Laplacian.", test_method_.c_str());
                score = calculateLaplacian(image);
            }

            // 打印结果
            ROS_INFO("Image Quality Score (%s): %.4f", test_method_.c_str(), score);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    // 使用拉普拉斯算子计算清晰度
    double calculateLaplacian(const cv::Mat& image) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // 使用拉普拉斯算子计算图像的方差，方差越大，图像越清晰
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Mat laplacian_abs = cv::abs(laplacian);

        // 计算拉普拉斯方差作为图像清晰度的度量
        return cv::mean(laplacian_abs)[0];
    }

    // 使用 SFR 计算清晰度
    double calculateSFR(const cv::Mat& image, cv::Rect& roi) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // 检测棋盘格角点
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, cv::Size(chessboard_cols_, chessboard_rows_), corners, 
                                                cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

        if (!found) {
            // 如果无法识别棋盘格，选择图像的中心区域作为 ROI
            ROS_WARN("Chessboard corners not found. Using center region as ROI.");
            roi = cv::Rect(gray.cols / 4, gray.rows / 4, gray.cols / 2, gray.rows / 2);
            return calculateSFRWithROI(gray, roi);
        }

        // 精确化角点位置
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), 
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

        // 获取棋盘格区域的边界框作为 ROI
        roi = cv::boundingRect(corners);

        // 计算 SFR
        return calculateSFRWithROI(gray, roi);
    }

     // 在给定 ROI 区域内计算 SFR
    double calculateSFRWithROI(const cv::Mat& gray, const cv::Rect& roi) {
        // 选择 ROI 区域
        cv::Mat cropped = gray(roi);

        // 计算傅里叶变换
        cv::Mat fft_image;
        cv::dft(cropped, fft_image, cv::DFT_COMPLEX_OUTPUT);

        // 计算幅值谱
        cv::Mat magnitude;
        std::vector<cv::Mat> planes;
        cv::split(fft_image, planes);
        cv::magnitude(planes[0], planes[1], magnitude);

        // 计算 SFR
        return cv::sum(magnitude)[0] / (cropped.rows * cropped.cols);
    }

    void displayResult(const cv::Mat& image, double score) {
        cv::Mat display_image = image.clone();
        cv::putText(display_image, 
                    cv::format("Score: %.4f", score), 
                    cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    1.0, 
                    cv::Scalar(0, 255, 0), 
                    2);
        cv::imshow("Image Quality Evaluation", display_image);
        cv::waitKey(1);
    }

    void displayResult(const cv::Mat& image, double score, const cv::Rect& roi) {
        cv::Mat display_image = image.clone();
        cv::putText(display_image, 
                    cv::format("Score: %.4f", score), 
                    cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    1.0, 
                    cv::Scalar(0, 255, 0), 
                    2);
        cv::rectangle(display_image, roi, cv::Scalar(0, 255, 0), 2);
        cv::putText(display_image, "ROI", cv::Point(roi.x + 10, roi.y + 20), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
        cv::imshow("Image Quality Evaluation", display_image);
        cv::waitKey(1);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_focus_evaluator");
    ImageFocusEvaluator evaluator;
    ros::spin();
    return 0;
}
