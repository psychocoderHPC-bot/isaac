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

// Regression coverage for the ALPAKA_DEVICE_GLOBAL variables that the alpaka3
// migration introduced. The real ISAAC headers are included, not a copy, so
// this test tracks the declarations ISAAC actually ships.

#include "device_globals_second_tu.hpp"

#include <isaac/isaac_common_kernel.hpp>
#include <isaac/isaac_ssao_kernel.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace
{
    constexpr int kFunctorParameterCount = ISAAC_MAX_SOURCES * ISAAC_MAX_FUNCTORS;
    constexpr int kSsaoKernelSize = 64;
    constexpr int kSsaoNoiseSize = 16;
    constexpr int kInitializerCheckCount = 8;
    constexpr int kFunctionChainSources = 4;
    constexpr unsigned char kSimulationPaddingByte = 0x5A;

    // ---------------------------------------------------------------------
    // Device side
    // ---------------------------------------------------------------------

    ISAAC_DEVICE_INLINE bool allZero(isaac::isaac_float3 const& value)
    {
        return value.x == isaac::isaac_float(0) && value.y == isaac::isaac_float(0)
               && value.z == isaac::isaac_float(0);
    }

    ISAAC_DEVICE_INLINE bool allZero(isaac::isaac_float4 const& value)
    {
        return value.x == isaac::isaac_float(0) && value.y == isaac::isaac_float(0)
               && value.z == isaac::isaac_float(0) && value.w == isaac::isaac_float(0);
    }

    ISAAC_DEVICE_INLINE bool allZero(isaac::isaac_mat4 const& value)
    {
        for(int column = 0; column < 4; ++column)
            for(int row = 0; row < 4; ++row)
                if(value[column][row] != isaac::isaac_float(0))
                    return false;
        return true;
    }

    ISAAC_DEVICE_INLINE bool allZero(isaac::SimulationSizeStruct const& value)
    {
        for(int axis = 0; axis < 3; ++axis)
        {
            if(value.globalSize[axis] != 0 || value.localSize[axis] != 0
               || value.localParticleSize[axis] != 0 || value.globalSizeScaled[axis] != 0
               || value.localSizeScaled[axis] != 0)
                return false;
            if(value.position[axis] != 0 || value.positionScaled[axis] != 0)
                return false;
        }
        return value.maxGlobalSize == 0 && value.maxGlobalSizeScaled == 0;
    }

    // Checks the declared initializer of every migrated global. Every ALPAKA_DEVICE_GLOBAL
    // in the ISAAC headers is declared without an initializer argument, so a fresh
    // device must observe all-zero storage (and null function pointers).
    struct InitializerCheckKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(T_Acc const&, isaac::isaac_int* failures) const
        {
            for(int i = 0; i < kInitializerCheckCount; ++i)
                failures[i] = 0;

            if(!allZero(isaac::InverseMVPMatrix.get()))
                failures[0] = 1;
            if(!allZero(isaac::ModelViewMatrix.get()))
                failures[1] = 1;
            if(!allZero(isaac::ProjectionMatrix.get()))
                failures[2] = 1;
            if(!allZero(isaac::SimulationSize.get()))
                failures[3] = 1;

            for(int i = 0; i < kFunctorParameterCount; ++i)
                if(!allZero(isaac::FunctorParameter.get()[i]))
                    failures[4] = 1;

            for(int i = 0; i < ISAAC_MAX_SOURCES; ++i)
                if(isaac::FunctionChain.get()[i] != nullptr)
                    failures[5] = 1;

            for(int i = 0; i < kSsaoKernelSize; ++i)
                if(!allZero(isaac::SSAOKernelArray.get()[i]))
                    failures[6] = 1;

            for(int i = 0; i < kSsaoNoiseSize; ++i)
                if(!allZero(isaac::SSAONoiseArray.get()[i]))
                    failures[7] = 1;
        }
    };

    struct ReadOutput
    {
        isaac::isaac_float4 matvec[3];
        isaac::SimulationSizeStruct simulationSize;
        isaac::isaac_float4 functorParameter[kFunctorParameterCount];
        isaac::isaac_float3 ssaoKernel[kSsaoKernelSize];
        isaac::isaac_float3 ssaoNoise[kSsaoNoiseSize];
    };

    // Reads every migrated global from a kernel. The matrices are consumed via
    // a matrix-vector product so the whole matrix content takes part in the
    // result instead of a single element being inspected.
    struct ReadAllKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(T_Acc const&, ReadOutput* out) const
        {
            const isaac::isaac_float4 vector(
                isaac::isaac_float(1),
                isaac::isaac_float(2),
                isaac::isaac_float(4),
                isaac::isaac_float(8));
            out->matvec[0] = isaac::InverseMVPMatrix.get() * vector;
            out->matvec[1] = isaac::ModelViewMatrix.get() * vector;
            out->matvec[2] = isaac::ProjectionMatrix.get() * vector;

            out->simulationSize = isaac::SimulationSize.get();

            for(int i = 0; i < kFunctorParameterCount; ++i)
                out->functorParameter[i] = isaac::FunctorParameter.get()[i];
            for(int i = 0; i < kSsaoKernelSize; ++i)
                out->ssaoKernel[i] = isaac::SSAOKernelArray.get()[i];
            for(int i = 0; i < kSsaoNoiseSize; ++i)
                out->ssaoNoise[i] = isaac::SSAONoiseArray.get()[i];
        }
    };

    // Device function stored into the FunctionChain device global. Its signature
    // matches isaac::FunctorChainPointer exactly, the same way ISAAC stores
    // pointers to generateFunctorChain<N>.
    ISAAC_DEVICE isaac::isaac_float testChainFunctor(
        void const*,
        void const* data,
        isaac::isaac_int sourceIndex)
    {
        const auto value = *static_cast<isaac::isaac_float const*>(data);
        const auto parameter = isaac::FunctorParameter.get()[sourceIndex * ISAAC_MAX_FUNCTORS];
        return value + parameter.x + isaac::isaac_float(2) * parameter.y
               + isaac::isaac_float(3) * parameter.z + isaac::isaac_float(4) * parameter.w;
    }

    // NV2 regression: write device function pointers into the FunctionChain
    // device global from inside a kernel.
    struct PopulateFunctionChainKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(T_Acc const&) const
        {
            for(int i = 0; i < ISAAC_MAX_SOURCES; ++i)
                isaac::FunctionChain.get()[i] = &testChainFunctor;
        }
    };

    // Call the pointers indirectly from a second kernel, exactly like
    // isaac::applyFunctorChain() does.
    struct CallFunctionChainKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(T_Acc const& acc, isaac::isaac_float const* input, isaac::isaac_float* output) const
        {
            for(int i = 0; i < kFunctionChainSources; ++i)
                output[i] = isaac::FunctionChain.get()[i](&acc, &input[i], i);
        }
    };

    // ---------------------------------------------------------------------
    // Host side sentinels and comparisons
    // ---------------------------------------------------------------------

    struct Sentinels
    {
        isaac::isaac_mat4 inverse;
        isaac::isaac_mat4 modelview;
        isaac::isaac_mat4 projection;
        isaac::SimulationSizeStruct simulationSize;
        isaac::isaac_float4 functorParameter[kFunctorParameterCount];
        isaac::isaac_float3 ssaoKernel[kSsaoKernelSize];
        isaac::isaac_float3 ssaoNoise[kSsaoNoiseSize];
    };

    isaac::isaac_mat4 makeMatrix(isaac::isaac_float base)
    {
        isaac::isaac_mat4 matrix;
        auto* values = glm::value_ptr(matrix);
        for(int i = 0; i < 16; ++i)
            values[i] = base + isaac::isaac_float(i);
        return matrix;
    }

    isaac::SimulationSizeStruct makeSimulationSize(int iteration)
    {
        isaac::SimulationSizeStruct value;
        // Fill padding with a non-zero pattern first. The fields below only
        // overwrite their own bytes, so the padding keeps this byte value and
        // the readback proves the copy covers sizeof(SimulationSizeStruct).
        std::memset(&value, kSimulationPaddingByte, sizeof(value));
        const ISAAC_IDX_TYPE base = ISAAC_IDX_TYPE(1000 * (iteration + 1));
        value.globalSize = isaac::isaac_size3(base + 1, base + 2, base + 3);
        value.maxGlobalSize = base + 4;
        value.position = isaac::isaac_int3(-int(base), int(base) + 1, -int(base) - 1);
        value.localSize = isaac::isaac_size3(base + 5, base + 6, base + 7);
        value.localParticleSize = isaac::isaac_size3(base + 8, base + 9, base + 10);
        value.globalSizeScaled = isaac::isaac_size3(base + 11, base + 12, base + 13);
        value.maxGlobalSizeScaled = base + 14;
        value.positionScaled = isaac::isaac_int3(int(base) + 2, -int(base) - 2, int(base) + 3);
        value.localSizeScaled = isaac::isaac_size3(base + 15, base + 16, base + 17);
        return value;
    }

    void fillFunctorParameters(isaac::isaac_float4* parameters, int iteration)
    {
        const auto base = isaac::isaac_float(100 * (iteration + 1));
        for(int i = 0; i < kFunctorParameterCount; ++i)
        {
            const auto offset = isaac::isaac_float(i);
            parameters[i] = isaac::isaac_float4(
                base + offset,
                base + offset + isaac::isaac_float(0.5),
                base + offset + isaac::isaac_float(0.25),
                base + offset - isaac::isaac_float(0.75));
        }
    }

    void fillSsaoKernel(isaac::isaac_float3* samples, int iteration)
    {
        const auto base = isaac::isaac_float(10 * (iteration + 1));
        for(int i = 0; i < kSsaoKernelSize; ++i)
            samples[i] = isaac::isaac_float3(base + isaac::isaac_float(i), base - isaac::isaac_float(i), base * isaac::isaac_float(0.5));
    }

    void fillSsaoNoise(isaac::isaac_float3* samples, int iteration)
    {
        const auto base = isaac::isaac_float(7 * (iteration + 1));
        for(int i = 0; i < kSsaoNoiseSize; ++i)
            samples[i] = isaac::isaac_float3(base + isaac::isaac_float(i), base + isaac::isaac_float(2 * i), isaac::isaac_float(0));
    }

    Sentinels makeSentinels(int iteration)
    {
        Sentinels sentinels{};
        const auto matrixBase = isaac::isaac_float(25 * iteration);
        sentinels.inverse = makeMatrix(matrixBase + isaac::isaac_float(1));
        sentinels.modelview = makeMatrix(matrixBase + isaac::isaac_float(101));
        sentinels.projection = makeMatrix(matrixBase + isaac::isaac_float(201));
        sentinels.simulationSize = makeSimulationSize(iteration);
        fillFunctorParameters(sentinels.functorParameter, iteration);
        fillSsaoKernel(sentinels.ssaoKernel, iteration);
        fillSsaoNoise(sentinels.ssaoNoise, iteration);
        return sentinels;
    }

    bool close(isaac::isaac_float lhs, isaac::isaac_float rhs)
    {
        return std::fabs(lhs - rhs) <= isaac::isaac_float(1e-4);
    }

    bool sameVec(isaac::isaac_float4 const& lhs, isaac::isaac_float4 const& rhs)
    {
        return close(lhs.x, rhs.x) && close(lhs.y, rhs.y) && close(lhs.z, rhs.z) && close(lhs.w, rhs.w);
    }

    bool sameVec(isaac::isaac_float3 const& lhs, isaac::isaac_float3 const& rhs)
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }

    bool sameSimulationSize(isaac::SimulationSizeStruct const& lhs, isaac::SimulationSizeStruct const& rhs)
    {
        for(int axis = 0; axis < 3; ++axis)
        {
            if(lhs.globalSize[axis] != rhs.globalSize[axis] || lhs.localSize[axis] != rhs.localSize[axis]
               || lhs.localParticleSize[axis] != rhs.localParticleSize[axis]
               || lhs.globalSizeScaled[axis] != rhs.globalSizeScaled[axis]
               || lhs.localSizeScaled[axis] != rhs.localSizeScaled[axis]
               || lhs.position[axis] != rhs.position[axis] || lhs.positionScaled[axis] != rhs.positionScaled[axis])
                return false;
        }
        return lhs.maxGlobalSize == rhs.maxGlobalSize && lhs.maxGlobalSizeScaled == rhs.maxGlobalSizeScaled;
    }

    bool report(char const* name, bool ok, std::string const& detail)
    {
        if(ok)
        {
            std::printf("PASS %s\n", name);
            return true;
        }
        std::printf("FAIL %s: %s\n", name, detail.c_str());
        return false;
    }

    void copyToHost(isaac_test::TestQueue const& queue, auto& hostBuffer, auto const& deviceBuffer)
    {
        alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
        alpaka::onHost::wait(queue);
    }

    bool checkInitializers(isaac_test::TestQueue const& queue, isaac_test::TestDevice const& device)
    {
        auto failures = alpaka::onHost::alloc<isaac::isaac_int>(device, ISAAC_IDX_TYPE(kInitializerCheckCount));
        auto failuresHost = alpaka::onHost::alloc<isaac::isaac_int>(
            alpaka::onHost::makeHostDevice(),
            ISAAC_IDX_TYPE(kInitializerCheckCount));

        const alpaka::Vec<ISAAC_IDX_TYPE, 3u> one(ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1));
        queue.enqueue(
            alpaka::onHost::ThreadSpec{one, one},
            InitializerCheckKernel{},
            alpaka::onHost::data(failures));
        copyToHost(queue, failuresHost, failures);

        static char const* const names[kInitializerCheckCount] = {
            "InverseMVPMatrix",
            "ModelViewMatrix",
            "ProjectionMatrix",
            "SimulationSize",
            "FunctorParameter",
            "FunctionChain (null function pointers)",
            "SSAOKernelArray",
            "SSAONoiseArray"};
        std::string detail;
        for(int i = 0; i < kInitializerCheckCount; ++i)
        {
            if(alpaka::onHost::data(failuresHost)[i] != 0)
            {
                if(!detail.empty())
                    detail += ", ";
                detail += names[i];
                detail += " not zero-initialized";
            }
        }

        // Read the SimulationSize device global back as raw bytes to also cover
        // its padding bytes, which a field-by-field kernel check cannot see.
        isaac::SimulationSizeStruct readback;
        std::memset(&readback, 0, sizeof(readback));
        alpaka::onHost::memcpy(queue, &readback, isaac::SimulationSize);
        alpaka::onHost::wait(queue);
        isaac::SimulationSizeStruct zeroSize;
        std::memset(&zeroSize, 0, sizeof(zeroSize));
        if(std::memcmp(&readback, &zeroSize, sizeof(readback)) != 0 && detail.empty())
            detail = "SimulationSize initializer bytes are not all zero (padding included)";

        return report("initializers", detail.empty(), detail);
    }

    bool checkNoInitReadback(isaac_test::TestQueue const& queue, isaac_test::TestDevice const& device)
    {
        auto output = alpaka::onHost::alloc<ReadOutput>(device, ISAAC_IDX_TYPE(1));
        auto outputHost = alpaka::onHost::alloc<ReadOutput>(alpaka::onHost::makeHostDevice(), ISAAC_IDX_TYPE(1));

        const alpaka::Vec<ISAAC_IDX_TYPE, 3u> one(ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1));
        queue.enqueue(alpaka::onHost::ThreadSpec{one, one}, ReadAllKernel{}, alpaka::onHost::data(output));
        copyToHost(queue, outputHost, output);

        ReadOutput const& got = *alpaka::onHost::data(outputHost);
        std::string detail;
        for(int matrix = 0; matrix < 3; ++matrix)
            if(!sameVec(got.matvec[matrix], isaac::isaac_float4(isaac::isaac_float(0))))
            {
                detail = "matrix-vector result is not zero";
                break;
            }
        if(detail.empty())
        {
            isaac::SimulationSizeStruct zeroSize;
            std::memset(&zeroSize, 0, sizeof(zeroSize));
            if(!sameSimulationSize(got.simulationSize, zeroSize))
                detail = "SimulationSize is not zero";
        }
        if(detail.empty())
            for(int i = 0; i < kFunctorParameterCount; ++i)
                if(!sameVec(got.functorParameter[i], isaac::isaac_float4(isaac::isaac_float(0))))
                {
                    detail = "FunctorParameter is not zero";
                    break;
                }
        if(detail.empty())
            for(int i = 0; i < kSsaoKernelSize; ++i)
                if(!sameVec(got.ssaoKernel[i], isaac::isaac_float3(isaac::isaac_float(0))))
                {
                    detail = "SSAOKernelArray is not zero";
                    break;
                }
        if(detail.empty())
            for(int i = 0; i < kSsaoNoiseSize; ++i)
                if(!sameVec(got.ssaoNoise[i], isaac::isaac_float3(isaac::isaac_float(0))))
                {
                    detail = "SSAONoiseArray is not zero";
                    break;
                }
        return report("negative_control_no_init", detail.empty(), detail);
    }

    bool runRoundTrip(
        isaac_test::TestQueue const& queue,
        isaac_test::TestDevice const& device,
        int iteration)
    {
        const Sentinels sentinels = makeSentinels(iteration);

        auto output = alpaka::onHost::alloc<ReadOutput>(device, ISAAC_IDX_TYPE(1));
        auto outputHost = alpaka::onHost::alloc<ReadOutput>(alpaka::onHost::makeHostDevice(), ISAAC_IDX_TYPE(1));
        auto projectionOutput = alpaka::onHost::alloc<isaac::isaac_float>(device, ISAAC_IDX_TYPE(16));
        auto projectionOutputHost
            = alpaka::onHost::alloc<isaac::isaac_float>(alpaka::onHost::makeHostDevice(), ISAAC_IDX_TYPE(16));

        // Ordering: all initialisation copies and the reading kernel are
        // enqueued on the same non-blocking queue without an intermediate wait.
        alpaka::onHost::memcpy(queue, isaac::InverseMVPMatrix, &sentinels.inverse);
        alpaka::onHost::memcpy(queue, isaac::ModelViewMatrix, &sentinels.modelview);
        alpaka::onHost::memcpy(queue, isaac::ProjectionMatrix, &sentinels.projection);
        // Written from the second translation unit to prove the inline device
        // global has one shared instance across TUs.
        isaac_test::secondTUWriteSimulationSize(queue, sentinels.simulationSize);
        alpaka::onHost::memcpy(queue, isaac::FunctorParameter, &sentinels.functorParameter[0]);
        alpaka::onHost::memcpy(queue, isaac::SSAOKernelArray, &sentinels.ssaoKernel[0]);
        alpaka::onHost::memcpy(queue, isaac::SSAONoiseArray, &sentinels.ssaoNoise[0]);

        const alpaka::Vec<ISAAC_IDX_TYPE, 3u> one(ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1));
        queue.enqueue(alpaka::onHost::ThreadSpec{one, one}, ReadAllKernel{}, alpaka::onHost::data(output));
        if(iteration == 0)
            isaac_test::secondTUEnqueueProjectionMatrixRead(queue, alpaka::onHost::data(projectionOutput));

        copyToHost(queue, outputHost, output);
        if(iteration == 0)
            copyToHost(queue, projectionOutputHost, projectionOutput);

        ReadOutput const& got = *alpaka::onHost::data(outputHost);
        std::string detail;

        const isaac::isaac_float4 vector(
            isaac::isaac_float(1),
            isaac::isaac_float(2),
            isaac::isaac_float(4),
            isaac::isaac_float(8));
        const isaac::isaac_mat4* expectedMatrices[3]
            = {&sentinels.inverse, &sentinels.modelview, &sentinels.projection};
        static char const* const matrixNames[3]
            = {"InverseMVPMatrix", "ModelViewMatrix", "ProjectionMatrix"};
        for(int matrix = 0; matrix < 3; ++matrix)
        {
            if(!sameVec(got.matvec[matrix], *expectedMatrices[matrix] * vector))
            {
                detail = std::string(matrixNames[matrix]) + " matrix-vector product mismatch";
                break;
            }
        }
        if(detail.empty() && !sameSimulationSize(got.simulationSize, sentinels.simulationSize))
            detail = "SimulationSize content mismatch";
        if(detail.empty())
            for(int i = 0; i < kFunctorParameterCount; ++i)
                if(!sameVec(got.functorParameter[i], sentinels.functorParameter[i]))
                {
                    detail = "FunctorParameter content mismatch";
                    break;
                }
        if(detail.empty())
            for(int i = 0; i < kSsaoKernelSize; ++i)
                if(!sameVec(got.ssaoKernel[i], sentinels.ssaoKernel[i]))
                {
                    detail = "SSAOKernelArray content mismatch";
                    break;
                }
        if(detail.empty())
            for(int i = 0; i < kSsaoNoiseSize; ++i)
                if(!sameVec(got.ssaoNoise[i], sentinels.ssaoNoise[i]))
                {
                    detail = "SSAONoiseArray content mismatch";
                    break;
                }

        // SimulationSize must round-trip byte for byte, including padding.
        isaac::SimulationSizeStruct simulationReadback;
        std::memset(&simulationReadback, 0, sizeof(simulationReadback));
        alpaka::onHost::memcpy(queue, &simulationReadback, isaac::SimulationSize);
        alpaka::onHost::wait(queue);
        if(detail.empty()
           && std::memcmp(&simulationReadback, &sentinels.simulationSize, sizeof(simulationReadback)) != 0)
            detail = "SimulationSize bytes (padding included) do not round-trip";

        // Cross-TU read of ProjectionMatrix.
        if(iteration == 0 && detail.empty())
        {
            const auto* projectionValues = glm::value_ptr(sentinels.projection);
            for(int i = 0; i < 16; ++i)
                if(!close(alpaka::onHost::data(projectionOutputHost)[i], projectionValues[i]))
                {
                    detail = "ProjectionMatrix read from the second TU mismatch";
                    break;
                }
        }

        char name[64];
        std::snprintf(name, sizeof(name), "roundtrip_iteration_%d", iteration);
        return report(name, detail.empty(), detail);
    }

    bool runFunctionChain(isaac_test::TestQueue const& queue, isaac_test::TestDevice const& device)
    {
        isaac::isaac_float4 parameters[kFunctorParameterCount];
        fillFunctorParameters(parameters, 7);

        auto input = alpaka::onHost::alloc<isaac::isaac_float>(device, ISAAC_IDX_TYPE(kFunctionChainSources));
        auto inputHost
            = alpaka::onHost::alloc<isaac::isaac_float>(alpaka::onHost::makeHostDevice(), ISAAC_IDX_TYPE(kFunctionChainSources));
        auto output = alpaka::onHost::alloc<isaac::isaac_float>(device, ISAAC_IDX_TYPE(kFunctionChainSources));
        auto outputHost
            = alpaka::onHost::alloc<isaac::isaac_float>(alpaka::onHost::makeHostDevice(), ISAAC_IDX_TYPE(kFunctionChainSources));

        for(int i = 0; i < kFunctionChainSources; ++i)
            alpaka::onHost::data(inputHost)[i] = isaac::isaac_float(i) + isaac::isaac_float(0.25);

        // Parameter buffer and input data are ready before the kernel chain runs.
        alpaka::onHost::memcpy(queue, isaac::FunctorParameter, &parameters[0]);
        alpaka::onHost::memcpy(queue, input, inputHost);
        alpaka::onHost::wait(queue);

        // NV2 regression: populate the FunctionChain device global from inside
        // one kernel, then call the stored pointers from a second kernel. Both
        // are on the same non-blocking queue, so the second kernel must observe
        // the pointers written by the first.
        const alpaka::Vec<ISAAC_IDX_TYPE, 3u> one(ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1), ISAAC_IDX_TYPE(1));
        queue.enqueue(alpaka::onHost::ThreadSpec{one, one}, PopulateFunctionChainKernel{});
        queue.enqueue(
            alpaka::onHost::ThreadSpec{one, one},
            CallFunctionChainKernel{},
            alpaka::onHost::data(input),
            alpaka::onHost::data(output));
        copyToHost(queue, outputHost, output);

        std::string detail;
        for(int i = 0; i < kFunctionChainSources; ++i)
        {
            const auto& parameter = parameters[i * ISAAC_MAX_FUNCTORS];
            const auto expected = alpaka::onHost::data(inputHost)[i] + parameter.x + isaac::isaac_float(2) * parameter.y
                                  + isaac::isaac_float(3) * parameter.z + isaac::isaac_float(4) * parameter.w;
            if(!close(alpaka::onHost::data(outputHost)[i], expected))
            {
                detail = "indirect function-pointer call returned the wrong value";
                break;
            }
        }
        return report("function_chain_nv2", detail.empty(), detail);
    }
} // namespace

int main(int argc, char** argv)
{
    const bool negativeControl = argc > 1 && std::string(argv[1]) == "--negative-control";

    auto deviceSelector = alpaka::onHost::makeDeviceSelector(isaac::accApi, isaac::accDeviceKind);
    if(deviceSelector.getDeviceCount() == 0)
    {
        std::fprintf(stderr, "FAIL no accelerator device available for the selected ISAAC_BACKEND\n");
        return 2;
    }

    isaac_test::TestDevice device = deviceSelector.makeDevice(0);
    isaac_test::TestQueue queue = device.makeQueue(alpaka::queueKind::nonBlocking);

    int failures = 0;
    if(!checkInitializers(queue, device))
        ++failures;

    if(negativeControl)
    {
        if(!checkNoInitReadback(queue, device))
            ++failures;
    }
    else
    {
        for(int iteration = 0; iteration < 2; ++iteration)
            if(!runRoundTrip(queue, device, iteration))
                ++failures;
        if(!runFunctionChain(queue, device))
            ++failures;
    }

    if(failures != 0)
    {
        std::printf("%d test case(s) failed\n", failures);
        return 1;
    }
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
