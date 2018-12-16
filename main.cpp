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
            //case ArgType::Float: return std::stof(arg.value);
        
            //case ArgType::Int : return std::stoi(arg.value);
        
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
        for (size_t i = 0; i < argc; i++)
        {
            if (args.find(argv[i]) != args.end())
            {
                args[argv[i]].value = argv[i++];
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

std::shared_ptr<ArgParser> create_arg_parser() {
    std::shared_ptr<ArgParser> p(new ArgParser());

    p->add_argument("-i", ArgParser::ArgType::Str);
    p->add_argument("-o", ArgParser::ArgType::Str);

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
    return std::make_tuple<float, float>(1.f, 0.f);
}

std::tuple<float, float> estimate_canny_params(const cv::Mat& img) {
    if (img.channels() != 1)
    {
        throw std::invalid_argument("channels != 1");
    }
}


template<typename T>
void __linear_contrast(cv::Mat& img,float alpha, float beta) {
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            img.at<T>(y, x) = alpha*(img.at<T>(y, x)) + T(beta);
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

    std::shared_ptr<ArgParser> p = create_arg_parser();
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
    cv::imshow("image after read", image);
#endif // DEBUG

    
    // Convert to grayscale 
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
#ifdef DEBUG
    cv::imshow("gray scale image", gray_image);
    
#endif // DEBUG

    auto params = estimate_linear_contrast_params(gray_image);
    linear_contrast(gray_image, std::get<0>(params), std::get<1>(params));

#ifdef DEBUG
    cv::imshow("Contrast gray scale image", gray_image);
#endif // DEBUG

    cv::Mat edges_image;
    cv::Canny(gray_image, edges_image, 100, 1);

#ifdef DEBUG
    cv::imshow("Edges  image", edges_image);
#endif // DEBUG

    cv::Mat corners;//, dst_norm, dst_norm_scaled;
    corners = cv::Mat::zeros(gray_image.size(), CV_8UC1);
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    cv::cornerHarris(gray_image, corners, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    return 0;
}