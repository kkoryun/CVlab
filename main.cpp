#include <opencv2/opencv.hpp>
#include <string>

#include <map>
#include <tuple>


class ArgParser
{
public:
    enum ArgType {
        Int,
        Str,
        Float
    };

    class Arg {
    public:
        Arg(const std::string& name, ArgType t) : 
            name(name),
            type(t)
        {}

        std::string name;
        std::string value;
        std::string description;
        ArgType     type;
    };

    ArgParser() = default;
    ArgParser(int argc, char** argv) {
        parse(argc, argv);
    }

    void add(std::string name, ArgType t) {
        args[name] = Arg(name, t);
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
    
    bool ToB(std::string value) {
        return value == "true" ||
            value == "True" ||
            value == "T";
    }
};


int main(int argc, char** argv) {
    cv::Mat image;
    std::string image_path;
    cv::imread(image_path);
    return 0;
}