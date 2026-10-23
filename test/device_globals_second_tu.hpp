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
// translation unit. Both TUs include the real ISAAC headers, so the linker
// enforces that every ALPAKA_DEVICE_GLOBAL(inline, ...) definition is ODR-safe
// and that both TUs address the same host/device instance.

// isaac_helper.hpp (included through isaac_texture.hpp) uses jansson types
// without including jansson.h itself. ISAAC's own translation units get the
// header via isaac_communicator.hpp, so every test TU includes it first.
#include <jansson.h>

#include <isaac/isaac_common_kernel.hpp>

#include <utility>

namespace isaac_test
{
    using TestDevice = isaac::AccDevice;
    using TestQueue = decltype(std::declval<TestDevice&>().makeQueue(alpaka::queueKind::nonBlocking));

    // Write the SimulationSize device global from the second TU. The fact that
    // this compiles and links proves the inline definition is shared across
    // TUs instead of being a TU-local copy.
    void secondTUWriteSimulationSize(TestQueue const& queue, isaac::SimulationSizeStruct const& value);

    // Enqueue a kernel defined in the second TU that reads ProjectionMatrix and
    // writes its 16 entries (column-major) to the device pointer `out`.
    void secondTUEnqueueProjectionMatrixRead(TestQueue const& queue, isaac::isaac_float* out);
} // namespace isaac_test
