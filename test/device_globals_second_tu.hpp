/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

// Shared declarations between the main test translation unit and the second
// translation unit.
//
// The second translation unit must include the real ISAAC header that declares
// the device globals it exercises, so the linker enforces that every
// ALPAKA_DEVICE_GLOBAL(inline, ...) definition is ODR-safe and that both TUs
// address the same host/device instance.
//
// Only isaac_functor_chain.hpp is used here. Its include graph
// (isaac_functors.hpp -> isaac_types.hpp, alpaka, Boost) does not reach
// isaac_texture.hpp/isaac_helper.hpp. isaac_common_kernel.hpp and
// isaac_ssao_kernel.hpp do reach isaac_helper.hpp, which defines six
// non-inline host helpers; including those headers from a second TU duplicates
// those host symbols. See test/CMakeLists.txt for the rationale.

#include <isaac/isaac_functor_chain.hpp>

#include <utility>

namespace isaac_test
{
    using TestDevice = isaac::AccDevice;
    using TestQueue = decltype(std::declval<TestDevice&>().makeQueue(alpaka::queueKind::nonBlocking));

    // Write the whole FunctorParameter device global from the second TU. The
    // fact that this compiles and links, and that the main TU observes the
    // values, proves the inline definition is shared across TUs instead of
    // being a TU-local copy.
    void secondTUWriteFunctorParameter(TestQueue const& queue, isaac::isaac_float4 const* values);

    // Enqueue a kernel defined in the second TU that reads FunctorParameter and
    // writes the first four entries to the device pointer `out`.
    void secondTUEnqueueFunctorParameterRead(TestQueue const& queue, isaac::isaac_float4* out);
} // namespace isaac_test
