#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include <tuple>
#include <memory>

class ArgParser
{
public:
    enum ArgType {
        Int,
        Str,
        Float,
        None
    };

    class Arg {
    public:

        Arg(const std::string& name = "", ArgType t = ArgType::None) :
            name(name),
            type(t)
        {}

        std::string name;
        std::string value;
        std::string description;
        ArgType     type;
    };

    template<typename T>
    T get_argument(const std::string& n) const {
        //const Arg& a = args[n];
        auto arg = args.find(n)->second;

        switch (arg.type)
        {
            //case ArgType::Float: return static_cast<T>(std::stof(arg.value));
        
            //case ArgType::Int : return static_cast<T>(std::stoi(arg.value));
        
            case ArgType::Str : return arg.value;
        
            case ArgType::None : return T();;
        }
        return T();
    }
    ArgParser() = default;

    void add_argument(std::string name, ArgType t, std::string desc = "") {
        args[name] = Arg(name, t);
        args[name].description = desc;
    }

    void parse(int argc, char** argv) {
        if (argc == 1)
        {

        }
        for (size_t i = 1; i < argc; i++)
        {
            if (args.find(argv[i]) != args.end())
            {
                args[argv[i]].value = argv[++i];
            }
        }

    }

    ~ArgParser() = default;

private:
    std::map <std::string, Arg> args;
    bool parsed;
    bool ToB(std::string value) {
        return value == "true" ||
            value == "True" ||
            value == "T";
    }
};

std::shared_ptr<ArgParser> create_arg_parser(int argc, char** argv) {
    std::shared_ptr<ArgParser> p(new ArgParser());

    p->add_argument("-i", ArgParser::ArgType::Str);
    p->add_argument("-o", ArgParser::ArgType::Str);
    p->parse(argc, argv);
    return p;
}

//void format(std::string& str, std::string& arg1) {
//
//}
#define DEBUG

//only for 1 channels
std::tuple<float, float> estimate_linear_contrast_params(const cv::Mat& img) {
    if (img.channels() != 1)
    {
        throw std::invalid_argument("channels != 1");
    }
    //cv::calcHist(img);
    return std::make_tuple<float, float>(1.f, 50.f);
}

std::tuple<float, float> estimate_canny_params(const cv::Mat& img) {
    if (img.channels() != 1)
    {
        throw std::invalid_argument("channels != 1");
    }
}


template<typename T>
void __linear_contrast(cv::Mat& img, float alpha, float beta) {
    float t;
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            t = (alpha * img.at<T>(y, x) + T(beta));
            img.at<T>(y, x) = t < std::numeric_limits<T>::max() ? t : std::numeric_limits<T>::max();
        }
    }
}

void linear_contrast(cv::Mat& img, float alpha, float beta) {
    int t = img.type();
    switch (t)
    {
    case CV_8UC1: __linear_contrast<unsigned char>(img, alpha, beta); break;
    default:
        break;
    }
}

int main(int argc, char** argv) {

    std::shared_ptr<ArgParser> p = create_arg_parser(argc, argv);
    std::string file_path = p->get_argument<std::string>("-i");
    cv::Mat image;
    try
    {
        image = cv::imread(file_path);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Can not read file "<< file_path <<std::endl;
        throw std::exception(e);
    }
    if (image.empty())
    {
        throw std::exception("image empty");
    }

#ifdef DEBUG
    cv::Size debug_img_size(450, 300);
#endif // DEBUG

#ifdef DEBUG
    {
        cv::Mat resized_img;
        cv::resize(image, resized_img, debug_img_size);
        cv::imshow("image after read", resized_img);
        cv::waitKey(10);
    }
#endif // DEBUG



    
    // Convert to grayscale 
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
#ifdef DEBUG
    {
        cv::Mat resized_img;
        cv::resize(gray_image, resized_img, debug_img_size);
        cv::imshow("gray scale image", resized_img);
        cv::waitKey(10);
    }
#endif // DEBUG

    auto params = estimate_linear_contrast_params(gray_image);
    linear_contrast(gray_image, std::get<0>(params), std::get<1>(params));

#ifdef DEBUG
    {
        cv::Mat resized_img;
        cv::resize(gray_image, resized_img, debug_img_size);
        cv::imshow("Contrast gray scale image", resized_img);
        cv::waitKey(10);
    }
  
#endif // DEBUG

    cv::Mat edges_image;
    cv::Canny(gray_image, edges_image, 70, 50);

#ifdef DEBUG
    {
        cv::Mat resized_img;
        cv::resize(edges_image, resized_img, debug_img_size);
        cv::imshow("Edges  image", resized_img);
        cv::waitKey(10);
    }
#endif // DEBUG

    cv::Mat corners, corners_norm, corners_norm_scaled;
    corners = cv::Mat::zeros(gray_image.size(), CV_32FC1);
    //corners = cv::Mat::zeros(gray_image.size(), CV_8UC1);
    int blockSize = 10;
    int apertureSize = 7;
    double k = 0.04;
    cv::cornerHarris(gray_image, corners, blockSize, apertureSize, k);
    cv::normalize(corners, corners_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

#ifdef DEBUG
    {
        cv::Mat corners_norm_;
        cv::Mat resized_img;
        cv::normalize(corners, corners_norm_, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat());
        cv::resize(corners_norm_, resized_img, debug_img_size);
        cv::imshow("Harris features", resized_img);
        cv::waitKey(10);
    }

#endif // DEBUG

    for (int j = 0; j < corners_norm.rows; j++)
    {
        for (int i = 0; i < corners_norm.cols; i++)
        {
            if (static_cast<int>(corners_norm.at<float>(j, i)) > 50)
            {
                cv::circle(image, cv::Point(i, j), 10, cv::Scalar(0), 2, 8, 0);
            }
        }
    }

    {
        cv::namedWindow("Corners");
        cv::Mat resized_img;
        cv::resize(image, resized_img, cv::Size(1350, 900));
        cv::imshow("Corners", resized_img);
        cv::waitKey(10);
        //cv::resize(image, resized_img, debug_img_size);
    }
    
    cv::Mat corners_norm_bin = cv::Mat::zeros(gray_image.size(), CV_8UC1);
    cv::threshold(corners_norm, corners_norm_bin,50, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat distance_map;
    cv::distanceTransform(corners_norm_bin, distance_map, cv::DIST_L2, 3);

#ifdef DEBUG
    {
        cv::Mat resized_img;
        cv::resize(edges_image, resized_img, debug_img_size);
        cv::imshow("Corners norm bin", resized_img);
        cv::waitKey(10);
    }
#endif // DEBUG
    
    cv::Mat edges_image_bin;
    cv::waitKey();
    return 0;
}