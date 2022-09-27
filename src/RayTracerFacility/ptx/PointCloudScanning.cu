#include <RayTracerUtilities.cuh>

namespace RayTracerFacility {
    extern "C" __constant__ PointCloudScanningLaunchParams
            pointCloudScanningLaunchParams;
#pragma region Closest hit functions
    extern "C" __global__ void __closesthit__PCS_R() {
        const auto &sbtData = *(const SBT *) optixGetSbtDataPointer();
        const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
        const int primitiveId = optixGetPrimitiveIndex();
        const float3 rayDirectionInternal = optixGetWorldRayDirection();
        glm::vec3 rayDirection = glm::vec3(
                rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
        glm::vec2 texCoord;
        glm::vec3 hitPoint;
        glm::vec3 normal;
        glm::vec3 tangent;
        glm::vec3 vertexColor;
        sbtData.GetGeometricInfo(rayDirection, texCoord, hitPoint, normal, tangent, vertexColor);

        PerRayData <uint64_t> &prd = *GetRayDataPointer < PerRayData < uint64_t >> ();
        prd.m_hitCount = 1;
        prd.m_energy = sbtData.m_handle;
        /*
        switch (sbtData.m_materialType) {
            case MaterialType::VertexColor: {
                prd.m_albedo = sbtData.m_mesh.GetColor(triangleBarycentricsInternal,
        indices);
            }
                break;
            case MaterialType::Default: {
                glm::vec3 albedoColor = static_cast<DefaultMaterial
        *>(sbtData.m_material)->GetAlbedo(texCoord); prd.m_albedo = albedoColor;
            }
                break;
            case MaterialType::MLVQ: {
                glm::vec3 btfColor = glm::vec3(0);
                prd.m_albedo = btfColor;
            }
                break;
        }*/
        prd.m_albedo = vertexColor;
        prd.m_normal = hitPoint;
    }
    extern "C" __global__ void __closesthit__PCS_SS() {}
#pragma endregion
#pragma region Any hit functions
    extern "C" __global__ void __anyhit__PCS_R() {}
    extern "C" __global__ void __anyhit__PCS_SS() {}
#pragma endregion
#pragma region Miss functions
    extern "C" __global__ void __miss__PCS_R() {
        PerRayData <uint64_t> &prd = *GetRayDataPointer < PerRayData < uint64_t >> ();
        prd.m_hitCount = 0;
        prd.m_energy = 0;
    }
    extern "C" __global__ void __miss__PCS_SS() {}
#pragma endregion
#pragma region Main ray generation
    extern "C" __global__ void __raygen__PCS() {
        unsigned ix = optixGetLaunchIndex().x;
        auto &samples = pointCloudScanningLaunchParams.m_samples[ix];
        auto start = samples.m_start;
        auto direction = samples.m_direction;
        float3 rayOrigin = make_float3(start.x, start.y, start.z);
        float3 rayDirection = make_float3(direction.x, direction.y, direction.z);

        PerRayData <uint64_t> perRayData;
        perRayData.m_random.Init(ix, 0);
        perRayData.m_hitCount = 0;
        perRayData.m_energy = 0;
        perRayData.m_normal = glm::vec3(0);
        perRayData.m_albedo = glm::vec3(0);
        uint32_t u0, u1;
        PackRayDataPointer(&perRayData, u0, u1);
        optixTrace(
                pointCloudScanningLaunchParams.m_traversable, rayOrigin, rayDirection,
                1e-3f, // tmin
                1e20f, // tmax
                0.0f,  // rayTime
                static_cast<OptixVisibilityMask>(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
                static_cast<int>(
                        RayType::Radiance), // SBT offset
                static_cast<int>(
                        RayType::RayTypeCount), // SBT stride
                static_cast<int>(
                        RayType::Radiance), // missSBTIndex
                u0, u1);
        samples.m_handle = perRayData.m_energy;
        samples.m_hit = perRayData.m_hitCount != 0;
        samples.m_albedo = perRayData.m_albedo;
        samples.m_end = perRayData.m_normal;
    }
#pragma endregion
} // namespace RayTracerFacility
