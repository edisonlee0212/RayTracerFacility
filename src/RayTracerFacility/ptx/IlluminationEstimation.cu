#include <RayTracerUtilities.cuh>

namespace RayTracerFacility {
    extern "C" __constant__ IlluminationEstimationLaunchParams
            illuminationEstimationLaunchParams;
#pragma region Closest hit functions
    extern "C" __global__ void __closesthit__IE_R() {
#pragma region Retrive information
        const auto &sbtData = *(const DefaultSbtData *) optixGetSbtDataPointer();
        const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
        const int primitiveId = optixGetPrimitiveIndex();
        const float3 rayDirectionInternal = optixGetWorldRayDirection();
        glm::vec3 rayDirection = glm::vec3(
                rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
        auto indices = sbtData.m_mesh.GetIndices(primitiveId);
        auto texCoord =
                sbtData.m_mesh.GetTexCoord(triangleBarycentricsInternal, indices);
        auto normal = sbtData.m_mesh.GetNormal(triangleBarycentricsInternal, indices);
        if (glm::dot(rayDirection, normal) > 0.0f) {
            normal = -normal;
        }
        auto tangent =
                sbtData.m_mesh.GetTangent(triangleBarycentricsInternal, indices);
        auto hitPoint =
                sbtData.m_mesh.GetPosition(triangleBarycentricsInternal, indices);
#pragma endregion
        PerRayData<float> &perRayData = *GetRayDataPointer < PerRayData < float >> ();
        unsigned hitCount = perRayData.m_hitCount + 1;
        auto energy = 0.0f;
        uint32_t u0, u1;
        PackRayDataPointer(&perRayData, u0, u1);
        perRayData.m_hitCount = hitCount;
        perRayData.m_energy = 0.0f;
        switch (sbtData.m_materialType) {
            case MaterialType::VertexColor: {
                perRayData.m_energy = glm::length(glm::vec3(
                        sbtData.m_mesh.GetColor(triangleBarycentricsInternal, indices)));
            }
                break;
            case MaterialType::Default: {
                static_cast<DefaultMaterial *>(sbtData.m_material)
                        ->ApplyNormalTexture(normal, texCoord, tangent);
                float metallic =
                        static_cast<DefaultMaterial *>(sbtData.m_material)->GetMetallic(texCoord);
                float roughness =
                        static_cast<DefaultMaterial *>(sbtData.m_material)->GetRoughness(texCoord);
                glm::vec3 albedoColor =
                        static_cast<DefaultMaterial *>(sbtData.m_material)->GetAlbedo(texCoord);
                if (perRayData.m_hitCount <=
                    illuminationEstimationLaunchParams.m_rayTracerProperties.m_rayProperties
                            .m_bounces) {
                    energy = 0.0f;
                    float f = 1.0f;
                    if (metallic >= 0.0f)
                        f = (metallic + 2) / (metallic + 1);
                    float3 incidentRayOrigin;
                    float3 newRayDirectionInternal;
                    BRDF(metallic, perRayData.m_random, normal, hitPoint, rayDirection,
                         incidentRayOrigin, newRayDirectionInternal);
                    optixTrace(
                            illuminationEstimationLaunchParams.m_traversable, incidentRayOrigin,
                            newRayDirectionInternal,
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
                    energy +=
                            glm::clamp(
                                    glm::abs(glm::dot(normal, glm::vec3(newRayDirectionInternal.x,
                                                                        newRayDirectionInternal.y,
                                                                        newRayDirectionInternal.z))) *
                                    roughness +
                                    (1.0f - roughness) * f,
                                    0.0f, 1.0f) *
                            perRayData.m_energy;
                }
                perRayData.m_energy =
                        energy +
                        static_cast<DefaultMaterial *>(sbtData.m_material)->m_diffuseIntensity;
            }
                break;
            case MaterialType::MLVQ: {
                glm::vec3 btfColor;
                if (perRayData.m_hitCount <=
                    illuminationEstimationLaunchParams.m_rayTracerProperties.m_rayProperties
                            .m_bounces) {
                    energy = 0.0f;
                    float f = 1.0f;
                    glm::vec3 reflected = Reflect(rayDirection, normal);
                    glm::vec3 newRayDirection =
                            RandomSampleHemisphere(perRayData.m_random, reflected, 1.0f);
                    static_cast<MLVQMaterial *>(sbtData.m_material)
                            ->GetValue(texCoord, rayDirection, newRayDirection, normal, tangent,
                                       btfColor,
                                       false /*(perRayData.m_printInfo && sampleID == 0)*/);
                    auto origin = hitPoint;
                    origin += normal * 1e-3f;
                    float3 incidentRayOrigin = make_float3(origin.x, origin.y, origin.z);
                    float3 newRayDirectionInternal =
                            make_float3(newRayDirection.x, newRayDirection.y, newRayDirection.z);
                    optixTrace(
                            illuminationEstimationLaunchParams.m_traversable, incidentRayOrigin,
                            newRayDirectionInternal,
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
                    energy += glm::length(btfColor) * perRayData.m_energy;
                }
                perRayData.m_energy = energy;
            }
                break;
        }
    }
    extern "C" __global__ void __closesthit__IE_SS() {}
#pragma endregion
#pragma region Any hit functions
    extern "C" __global__ void __anyhit__IE_R() {}
    extern "C" __global__ void __anyhit__IE_SS() {}
#pragma endregion
#pragma region Miss functions
    extern "C" __global__ void __miss__IE_R() {
        PerRayData<float> &prd = *GetRayDataPointer < PerRayData < float >> ();
        const float3 rayDir = optixGetWorldRayDirection();
        float3 rayOrigin = optixGetWorldRayOrigin();
        glm::vec3 rayOrig = glm::vec3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
        glm::vec3 rayDirection = glm::vec3(rayDir.x, rayDir.y, rayDir.z);
        glm::vec3 environmentalLightColor = CalculateEnvironmentalLight(
                rayOrig, rayDirection,
                illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment);
        prd.m_energy = glm::length(environmentalLightColor);
    }
    extern "C" __global__ void __miss__IE_SS() {}
#pragma endregion
#pragma region Main ray generation
    extern "C" __global__ void __raygen__IE() {
        unsigned ix = optixGetLaunchIndex().x;
        const auto numPointSamples =
                illuminationEstimationLaunchParams.m_rayTracerProperties.m_rayProperties
                        .m_samples;
        const auto position =
                illuminationEstimationLaunchParams.m_lightProbes[ix].m_position;
        const auto surfaceNormal =
                illuminationEstimationLaunchParams.m_lightProbes[ix].m_surfaceNormal;
        const auto pushDistance =
                illuminationEstimationLaunchParams.m_pushNormalDistance;
        const auto doubleFace =
                illuminationEstimationLaunchParams.m_lightProbes[ix].m_doubleFace;
        float pointEnergy = 0.0f;
        auto pointDirection = glm::vec3(0.0f);

        PerRayData<float> perRayData;
        perRayData.m_random.Init(ix, illuminationEstimationLaunchParams.m_seed);
        uint32_t u0, u1;
        PackRayDataPointer(&perRayData, u0, u1);
        for (int sampleID = 0; sampleID < numPointSamples; sampleID++) {
            perRayData.m_energy = 0.0f;
            perRayData.m_hitCount = 0;
            glm::vec3 rayDir, rayOrigin;
            if (doubleFace) {
                rayDir = RandomSampleSphere(perRayData.m_random);
                if (glm::dot(rayDir, surfaceNormal) > 0)
                    rayOrigin = position + surfaceNormal * pushDistance;
                else
                    rayOrigin = position - surfaceNormal * pushDistance;
            } else {
                rayDir = RandomSampleHemisphere(perRayData.m_random, surfaceNormal);
                rayOrigin = position + surfaceNormal * pushDistance;
            }
            float3 rayOriginInternal =
                    make_float3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
            float3 rayDirection = make_float3(rayDir.x, rayDir.y, rayDir.z);
            optixTrace(
                    illuminationEstimationLaunchParams.m_traversable, rayOriginInternal,
                    rayDirection,
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
            auto energy =
                    perRayData.m_energy * glm::abs(glm::dot(surfaceNormal, rayDir));
            pointEnergy += energy;
            pointDirection += rayDir * energy;
        }
        if (pointEnergy != 0) {
            illuminationEstimationLaunchParams.m_lightProbes[ix].m_energy =
                    pointEnergy / numPointSamples;
            illuminationEstimationLaunchParams.m_lightProbes[ix].m_direction =
                    glm::normalize(pointDirection);
        }
    }
#pragma endregion
} // namespace RayTracerFacility
