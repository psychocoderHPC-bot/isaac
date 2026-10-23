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

#include "device_globals_second_tu.hpp"

namespace
{
    // Reads a device global declared in isaac_common_kernel.hpp from a second
    // translation unit than the writer. A missing `inline` on the declaration
    // would either fail to link or produce two distinct instances.
    struct SecondTUProjectionMatrixReader
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(T_Acc const&, isaac::isaac_float* out) const
        {
            auto const& matrix = isaac::ProjectionMatrix.get();
            for(int column = 0; column < 4; ++column)
                for(int row = 0; row < 4; ++row)
                    out[column * 4 + row] = matrix[column][row];
        }
    };
} // namespace

namespace isaac_test
{
    void secondTUWriteSimulationSize(TestQueue const& queue, isaac::SimulationSizeStruct const& value)
    {
        alpaka::onHost::memcpy(queue, isaac::SimulationSize, &value);
    }

    void secondTUEnqueueProjectionMatrixRead(TestQueue const& queue, isaac::isaac_float* out)
    {
        const alpaka::Vec<ISAAC_IDX_TYPE, 3u> one(ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1));
        queue.enqueue(alpaka::onHost::ThreadSpec{one, one}, SecondTUProjectionMatrixReader{}, out);
    }
} // namespace isaac_test
