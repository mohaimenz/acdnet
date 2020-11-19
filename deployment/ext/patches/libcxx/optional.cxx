//===------------------------ optional.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "optional"
#include "experimental/optional"

namespace std
{

bad_optional_access::~bad_optional_access() _NOEXCEPT = default;

} // std

} // FIXME:: There is a bug in any headers, but it works well with this "}". It should be found the root cause.

// _LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL

// bad_optional_access::~bad_optional_access() _NOEXCEPT = default;

// _LIBCPP_END_NAMESPACE_EXPERIMENTAL
