#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include <tuple>
#include <memory>

#define DEBUG

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
void __linear_contrast(const cv::Mat& img, cv::Mat& out, float alpha, float beta) {
    T t;
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            t = (alpha * img.at<T>(y, x) + T(beta));
            out.at<T>(y, x) = t < std::numeric_limits<T>::max() ? t : std::numeric_limits<T>::max();
        }
    }
}

void linear_contrast(cv::Mat& img, cv::Mat& out, float alpha, float beta) {
    int t = img.type();
    switch (t)
    {
    case CV_8UC1:
        out = cv::Mat::zeros(img.size(), CV_8UC1);
        __linear_contrast<unsigned char>(img, out, alpha, beta); 
        break;
    default:
        throw std::exception("uncorrect image type"); 
        break;
    }
}

void show_debug_image(cv::Mat image, std::string name, cv::Size debug_img_size = cv::Size(450, 300),
    int draw_time = 10) {
    cv::Mat resized_img;
    cv::resize(image, resized_img, debug_img_size);
    cv::imshow(name, resized_img);
    cv::waitKey(draw_time);
}

void read_image(int argc, char** argv, cv::Mat& image) {
    std::shared_ptr<ArgParser> p = create_arg_parser(argc, argv);
    std::string file_path = p->get_argument<std::string>("-i");
    
    try
    {
        image = cv::imread(file_path);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Can not read file " << file_path << std::endl;
        throw std::exception(e);
    }
    if (image.empty())
    {
        throw std::exception("image empty");
    }
}

void draw_corners(cv::Mat& image, const cv::Mat& corners, int threshold) {
    for (int j = 0; j < corners.rows; j++)
    {
        for (int i = 0; i < corners.cols; i++)
        {
            if (static_cast<int>(corners.at<uchar>(j, i)) > threshold)
            {
                cv::circle(image, cv::Point(i, j), 10, cv::Scalar(0), 2, 8, 0);
            }
        }
    }

}

void invers_bin_imgage(cv::Mat image) {
    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            auto& e = image.at<uchar>(r, c);
            e = e ^ static_cast<unsigned char>(255);
        }
    }
}

template<typename img_T, typename integ_T>
void filter_avg_dist(const cv::Mat& inverse_edges_image_bin, const cv::Mat& image, cv::Mat& processed_image, 
    float coeff = 1.f) {
    cv::Mat edge_distance_map;
    cv::distanceTransform(inverse_edges_image_bin, edge_distance_map, cv::DIST_L2, 3);

    cv::Mat integral_img;
    cv::integral(image, integral_img);

    image.copyTo(processed_image);
    unsigned int filter_size;
    for (int r = 1; r < image.rows - 1; r++)
    {
        for (int c = 1; c < image.cols - 1; c++)
        {
            filter_size = static_cast<int>(edge_distance_map.at<float>(r, c) * coeff);
            filter_size = 3 / 2;
            cv::Point pA(c - filter_size, r - filter_size);
            cv::Point pB(c - filter_size, r + filter_size);
            cv::Point pC(c + filter_size, r + filter_size);
            cv::Point pD(c + filter_size, r - filter_size);
            int A = integral_img.at<integ_T>(pA);
            int B = integral_img.at<integ_T>(pB);
            int C = integral_img.at<integ_T>(pC);
            int D = integral_img.at<integ_T>(pD);
            int rect_sum = C - B - D + A;
            processed_image.at<img_T>(r, c) = rect_sum / 3 * 3;
        }
    }
}
int main(int argc, char** argv) {

    // Read image
    cv::Mat image;
    read_image(argc, argv, image);
    show_debug_image(image, "Original image");

    // Convert to grayscale 
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    show_debug_image(gray_image, "Grayscale image");

    // Contrast correction
    cv::Mat contrast_gray_image;
    std::tuple<float, float> params = estimate_linear_contrast_params(gray_image);
    linear_contrast(gray_image, contrast_gray_image, std::get<0>(params), std::get<1>(params));
    show_debug_image(contrast_gray_image, "Grayscale image");
    
    // Bluring image for canny
    cv::Mat blured_gray_image;
    cv::blur(contrast_gray_image, blured_gray_image, cv::Size(3, 3));
    show_debug_image(contrast_gray_image, "Grayscale image");

    // Canny edgw detector
    double canny_threshold1 = 10;
    double canny_threshold2 = 200;
    cv::Mat canny_edges_image;
    cv::Canny(blured_gray_image, canny_edges_image, 10, 200);
    show_debug_image(canny_edges_image, "Canny edge image");

    //Harris corner detecting
    int block_size = 10;
    int aperture_size = 7;
    double k = 0.04;
    cv::Mat corners, corners_norm = cv::Mat::zeros(gray_image.size(), CV_8UC1), corners_norm_scaled;
    //corners = cv::Mat::zeros(gray_image.size(), CV_32FC1);
    cv::cornerHarris(gray_image, corners, block_size, aperture_size, k);
    cv::normalize(corners, corners_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat());
    show_debug_image(corners_norm, "Harris corners");

    // Draw corners
    cv::Mat image_w_corners;
    image.copyTo(image_w_corners);
    draw_corners(image_w_corners, corners_norm, 50);
    show_debug_image(image_w_corners, "Image with corners");

    //
    cv::Mat corners_norm_bin = cv::Mat::zeros(gray_image.size(), CV_8UC1);
    cv::threshold(corners_norm, corners_norm_bin, 50, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat corners_distance_map;
    cv::distanceTransform(corners_norm_bin, corners_distance_map, cv::DIST_L2, 3);
    //cv::normalize(resized_img, resized_img, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat());

    
    // Convert to binary 
    cv::Mat edges_image_bin = cv::Mat::zeros(gray_image.size(), CV_8UC1);
    cv::threshold(canny_edges_image, edges_image_bin, 50, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    invers_bin_imgage(edges_image_bin);
    cv::Mat debug_image;
    cv::normalize(edges_image_bin, debug_image, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat());
    show_debug_image(debug_image, "Invers binary image");

    // Filter image
    cv::Mat processed_image;
    filter_avg_dist<uchar, int>(edges_image_bin, gray_image, processed_image);
    show_debug_image(processed_image, "Processed image");

    cv::waitKey();
    return 0;
}