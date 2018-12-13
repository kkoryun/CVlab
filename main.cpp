#include <opencv2/opencv.hpp>
#include <string>

#include <map>
#include <tuple>
class ArgParser
{
public:
    class Type {

    };

    ArgParser(int argc, char** argv) {

    }
    ~ArgParser();

private:
    using valueType = std::tuple<std::string, Type>;
    std::map<std::string, valueType> args_map;
};


int main(int argc, char** argv) {
    cv::Mat image;
    std::string image_path;
    cv::imread(image_path);
    return 0;
}