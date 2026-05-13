#include <cstdio>
#include <string>
#include "parser.h"

ArgParser::ArgParser(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // --key=value 
        if (arg.substr(0, 2) == "--") {
            size_t pos = arg.find('=');
            if (pos != std::string::npos) {
                std::string key = arg.substr(2, pos - 2);
                std::string value = arg.substr(pos + 1);
                args[key] = value;
            } else {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    args[arg.substr(2)] = argv[++i];
                } else {
                    args[arg.substr(2)] = "true";
                }
            }
        }
        // -k value 
        else if (arg[0] == '-' && arg.length() == 2) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args[std::string(1, arg[1])] = argv[++i];
            } else {
                args[std::string(1, arg[1])] = "true";
            }
        }
        // postional
        else {
            positional_args.push_back(arg);
        }
    }
}

bool ArgParser::has(const std::string& key) const {
    return args.find(key) != args.end();
}

std::string ArgParser::get(const std::string& key, const std::string& default_val) const {
    auto it = args.find(key);
    return it != args.end() ? it->second : default_val;
}

const std::vector<std::string>& ArgParser::positionals() const {
    return positional_args;
}

void ArgParser::print() const {
    for (const auto& [key, value] : args) {
        printf("%s = %s\n", key.c_str(), value.c_str());
    }
}