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


#include "isaac/isaac_iso_kernel.hpp"
#include "isaac/isaac_min_max_kernel.hpp"
#include "isaac/isaac_particle_kernel.hpp"
#include "isaac/isaac_ssao_kernel.hpp"
#include "isaac/isaac_volume_kernel.hpp"

namespace isaac
{
    template<typename T_Queue, typename T_KernelFnObj, typename... T_Args>
    inline void executeKernelOnVolume(
        isaac_size3 volumeSize,
        T_Queue& queue,
        const T_KernelFnObj& kernelFnObj,
        T_Args&&... args)
    {
        if(volumeSize.x == 0 || volumeSize.y == 0 || volumeSize.z == 0)
        {
            return;
        }

#ifdef ISAAC_MORTON_CODE
        isaac_size3 blockSize(
            glm::min(volumeSize.x, ISAAC_IDX_TYPE(128)),
            glm::min(volumeSize.y, ISAAC_IDX_TYPE(4)),
            1);
#else

        isaac_size3 blockSize;
        blockSize.x = glm::min(volumeSize.x, ISAAC_IDX_TYPE(512));
        blockSize.y = 512 / blockSize.x;
        blockSize.z = 512 / (blockSize.x * blockSize.y);
#endif
        isaac_size3 gridSize = (volumeSize + blockSize - ISAAC_IDX_TYPE(1)) / blockSize;

        if(queue.getApi() == alpaka::api::host)
        {
            blockSize = isaac_size3(1);
            gridSize = volumeSize;
        }
        // Coordinates need to be swapped for alpaka
        const alpaka::Vec<ISAAC_IDX_TYPE, 3u> blocks(blockSize.z, blockSize.y, blockSize.x);
        const alpaka::Vec<ISAAC_IDX_TYPE, 3u> grid(gridSize.z, gridSize.y, gridSize.x);
        queue.enqueue(alpaka::onHost::ThreadSpec{grid, blocks}, kernelFnObj, args...);
    }

    template<typename T_Queue, typename T_AccDev, typename T_Type, IndexType T_indexType>
    void syncOwnGuardTextures(
        T_Queue& queue,
        SyncedTexture3DAllocator<T_AccDev, T_Type, T_indexType>& texture,
        Neighbours<isaac_int>& neighbourIds)
    {
        const SyncToOwnGuard kernel;
        for(isaac_uint i = 1; i < 27; ++i)
        {
            if(neighbourIds.array[i] != -1)
            {
                Texture<T_Type, 3>& dstTexture = texture.getOwnGuardTexture(i);
                executeKernelOnVolume(
                    dstTexture.getSize(),
                    queue,
                    kernel,
                    indexToDirection(i),
                    texture.getTexture(),
                    dstTexture);
            }
        }
        alpaka::onHost::wait(queue);
    }

    template<typename T_Queue, typename T_AccDev, typename T_Type, IndexType T_indexType>
    void syncNeighbourGuardTextures(
        T_Queue& queue,
        SyncedTexture3DAllocator<T_AccDev, T_Type, T_indexType>& texture,
        Neighbours<isaac_int>& neighbourIds)
    {
        const SyncFromNeighbourGuard kernel;
        for(isaac_uint i = 1; i < 27; ++i)
        {
            if(neighbourIds.array[i] != -1)
            {
                Texture<T_Type, 3>& guardTexture = texture.getNeighbourGuardTexture(i);
                executeKernelOnVolume(
                    guardTexture.getSize(),
                    queue,
                    kernel,
                    indexToDirection(i),
                    guardTexture,
                    texture.getTexture());
            }
        }
        alpaka::onHost::wait(queue);
    }
} // namespace isaac