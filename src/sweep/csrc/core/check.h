#pragma once
// core/check.h -- the core's argument check, torch-free.
//
// SWEEP_CHECK(cond, msg...) is TORCH_CHECK with the c10 taken out: the same
// spelling at every site, the same "message parts streamed together" contract,
// the same message text -- so a test that matched an error string before
// matches it after.  It throws sweep::Error (a std::runtime_error), which
// pybind11 already maps to Python's RuntimeError, the class c10::Error mapped
// to; the binding needs no translator.
//
// Message parts are anything with operator<<.  Vectors print as "[a, b, c]"
// (the shape lists in the pool/output checks) -- c10::str's spelling, kept.
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sweep {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

namespace detail {

template <class T>
inline void put_one(std::ostream& o, const T& v) { o << v; }

template <class T>
inline void put_one(std::ostream& o, const std::vector<T>& v)
{
    o << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) o << ", ";
        o << v[i];
    }
    o << "]";
}

inline void put(std::ostream&) {}

template <class T, class... R>
inline void put(std::ostream& o, const T& t, const R&... r)
{
    put_one(o, t);
    put(o, r...);
}

template <class... A>
inline std::string str(const A&... a)
{
    std::ostringstream o;
    put(o, a...);
    return o.str();
}

[[noreturn]] inline void fail(const char* cond, const std::string& msg)
{
    // TORCH_CHECK's wording for a bare condition, so a message-less check
    // reads the same as before.
    throw Error(msg.empty() ? std::string("Expected ") + cond + " to be true, but got false."
                            : msg);
}

}  // namespace detail
}  // namespace sweep

#define SWEEP_CHECK(cond, ...)                                                        \
    do {                                                                              \
        if (!(cond))                                                                  \
            ::sweep::detail::fail(#cond, ::sweep::detail::str(__VA_ARGS__));          \
    } while (0)
