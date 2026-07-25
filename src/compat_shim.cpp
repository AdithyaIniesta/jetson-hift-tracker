#include <new>

namespace std {
void __throw_bad_array_new_length() { throw bad_alloc(); }
} // namespace std
