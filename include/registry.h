#ifndef REGISTRY_H
#define REGISTRY_H

#include <map>
#include <string>
#include <cstdio>

template <typename ParamType>
struct KernelRegistry {
    using FuncType = void(*)(ParamType);
    using Map = std::map<std::string, FuncType>;

    static Map& get() {
        static Map map;
        return map;
    }

    static bool add(const std::string& name, FuncType func) {
        get()[name] = func;
        return true;
    }

    static FuncType lookup(const std::string& name) {
        auto& map = get();
        auto it = map.find(name);
        if (it != map.end()) return it->second;
        fprintf(stderr, "Unknown kernel '%s'. Available: ", name.c_str());
        for (auto& p : map) fprintf(stderr, "%s ", p.first.c_str());
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
    }
};

template <typename T>
struct func_traits;

template <typename Ret, typename Arg>
struct func_traits<Ret(*)(Arg)> {
    using param_type = Arg;
};

#define REGISTER_KERNEL(Name, Func)                                                     \
    namespace {                                                                         \
        static const bool _kreg_##Func =                                                \
            KernelRegistry<func_traits<decltype(&Func)>::param_type>::add(#Name, Func); \
    }

#endif
