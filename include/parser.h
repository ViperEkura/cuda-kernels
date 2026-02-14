#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <unordered_map>
#include <vector>

class ArgParser {
private:
    std::unordered_map<std::string, std::string> args;
    std::vector<std::string> positional_args;
    
public:
    ArgParser(int argc, char* argv[]);
    
    bool has(const std::string& key) const;
    std::string get(const std::string& key, const std::string& default_val = "") const;
    const std::vector<std::string>& positionals() const;
    void print() const;
};

#endif
