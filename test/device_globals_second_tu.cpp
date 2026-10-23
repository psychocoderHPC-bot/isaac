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

// This TU deliberately includes only device_globals_second_tu.hpp, which pulls
// isaac_functor_chain.hpp. That header declares FunctorParameter and
// FunctionChain without reaching the non-inline host helpers in isaac_helper.hpp.

#include "device_globals_second_tu.hpp"

namespace
{
    constexpr int kSecondTUReadCount = 4;

    // Reads a device global declared in isaac_functor_chain.hpp from a second
    // translation unit than the writer. A missing `inline` on the declaration
    // would either fail to link or produce two distinct instances.
    struct SecondTUFunctorParameterReader
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(T_Acc const&, isaac::isaac_float4* out) const
        {
            for(int i = 0; i < kSecondTUReadCount; ++i)
                out[i] = isaac::FunctorParameter.get()[i];
        }
    };
} // namespace

namespace isaac_test
{
    void secondTUWriteFunctorParameter(TestQueue const& queue, isaac::isaac_float4 const* values)
    {
        alpaka::onHost::memcpy(queue, isaac::FunctorParameter, values);
    }

    void secondTUEnqueueFunctorParameterRead(TestQueue const& queue, isaac::isaac_float4* out)
    {
        const alpaka::Vec<ISAAC_IDX_TYPE, 3u> one(ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1));
        queue.enqueue(alpaka::onHost::ThreadSpec{one, one}, SecondTUFunctorParameterReader{}, out);
    }
} // namespace isaac_test
