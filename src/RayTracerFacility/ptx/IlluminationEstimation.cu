#include <RayTracerUtilities.cuh>

namespace RayTracerFacility {
    extern "C" __constant__ IlluminationEstimationLaunchParams
            illuminationEstimationLaunchParams;
#pragma region Closest hit functions
    extern "C" __global__ void __closesthit__IE_R() {
#pragma region Retrive information
        const auto &sbtData = *(const SBT *) optixGetSbtDataPointer();
        const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
        const int primitiveId = optixGetPrimitiveIndex();
        const float3 rayDirectionInternal = optixGetWorldRayDirection();
        glm::vec3 rayDirection = glm::vec3(
                rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
        auto hitInfo = sbtData.GetHitInfo(rayDirection);
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
                perRayData.m_energy = glm::length(hitInfo.m_color);
            }
                break;
            case MaterialType::Default: {
                auto *material = static_cast<SurfaceMaterial *>(sbtData.m_material);
                material->ApplyNormalTexture(hitInfo.m_normal, hitInfo.m_texCoord, hitInfo.m_tangent);
                float metallic =
                        material->GetMetallic(hitInfo.m_texCoord);
                float roughness =
                        material->GetRoughness(hitInfo.m_texCoord);
                glm::vec3 albedoColor =
                        material->GetAlbedo(hitInfo.m_texCoord);
                energy = 0.0;
                float f = 1.0f;
                if (metallic >= 0.0f)
                    f = (metallic + 2) / (metallic + 1);
                if (perRayData.m_hitCount <=
                        illuminationEstimationLaunchParams.m_rayTracerProperties.m_rayProperties
                            .m_bounces) {
                    bool needSample = false;
                    if (hitCount <= 1 && material->m_materialProperties.m_subsurfaceFactor > 0.0f && material->m_materialProperties.m_subsurfaceRadius > 0.0f) {
                        float3 incidentRayOrigin;
                        float3 newRayDirectionInternal;
                        glm::vec3 outNormal;
                        needSample = BSSRDF(metallic, perRayData.m_random,
                                                 material->m_materialProperties.m_subsurfaceRadius, sbtData.m_handle,
                                                 illuminationEstimationLaunchParams.m_traversable,
                                            hitInfo.m_position, rayDirection, hitInfo.m_normal,
                                                 incidentRayOrigin, newRayDirectionInternal, outNormal);
                        if (needSample) {
                            optixTrace(
                                    illuminationEstimationLaunchParams.m_traversable, incidentRayOrigin,
                                    newRayDirectionInternal,
                                    1e-3f, // tmin
                                    1e20f, // tmax
                                    0.0f,  // rayTime
                                    static_cast<OptixVisibilityMask>(255), OPTIX_RAY_FLAG_NONE,
                                    static_cast<int>(
                                            RayType::Radiance), // SBT offset
                                    static_cast<int>(
                                            RayType::RayTypeCount), // SBT stride
                                    static_cast<int>(
                                            RayType::Radiance), // missSBTIndex
                                    u0, u1);
                            energy += material->m_materialProperties.m_subsurfaceFactor *
                                      glm::clamp(
                                              glm::abs(glm::dot(outNormal, glm::vec3(newRayDirectionInternal.x,
                                                                                     newRayDirectionInternal.y,
                                                                                     newRayDirectionInternal.z))) *
                                              roughness +
                                              (1.0f - roughness) * f,
                                              0.0f, 1.0f) *
                                      perRayData.m_energy;
                        }
                    }
                    if(!needSample) {
                        float3 newRayDirectionInternal;
                        BRDF(metallic, perRayData.m_random, rayDirection, hitInfo.m_normal, newRayDirectionInternal);
                        optixTrace(
                                illuminationEstimationLaunchParams.m_traversable,
                                make_float3(hitInfo.m_position.x, hitInfo.m_position.y, hitInfo.m_position.z),
                                newRayDirectionInternal,
                                1e-3f, // tmin
                                1e20f, // tmax
                                0.0f,  // rayTime
                                static_cast<OptixVisibilityMask>(255),
                                OPTIX_RAY_FLAG_NONE,
                                static_cast<int>(
                                        RayType::Radiance), // SBT offset
                                static_cast<int>(
                                        RayType::RayTypeCount), // SBT stride
                                static_cast<int>(
                                        RayType::Radiance), // missSBTIndex
                                u0, u1);
                        energy += (1.0f - material->m_materialProperties.m_subsurfaceFactor) *
                                  glm::clamp(glm::abs(glm::dot(
                                                     hitInfo.m_normal, glm::vec3(newRayDirectionInternal.x,
                                                                       newRayDirectionInternal.y,
                                                                       newRayDirectionInternal.z))) *
                                             roughness +
                                             (1.0f - roughness) * f,
                                             0.0f, 1.0f) *
                                  perRayData.m_energy;
                    }
                }
                if (hitCount == 1) {
                    perRayData.m_normal = hitInfo.m_normal;
                    perRayData.m_albedo = albedoColor;
                }
                perRayData.m_energy =
                        energy +
                        static_cast<SurfaceMaterial *>(sbtData.m_material)->m_materialProperties.m_emission;

            }
                break;
            case MaterialType::MLVQ: {
                glm::vec3 btfColor;
                if (perRayData.m_hitCount <=
                    illuminationEstimationLaunchParams.m_rayTracerProperties.m_rayProperties
                            .m_bounces) {
                    energy = 0.0f;
                    float f = 1.0f;
                    glm::vec3 reflected = Reflect(rayDirection, hitInfo.m_normal);
                    glm::vec3 newRayDirection =
                            RandomSampleHemisphere(perRayData.m_random, reflected, 1.0f);
                    static_cast<MLVQMaterial *>(sbtData.m_material)
                            ->GetValue(hitInfo.m_texCoord, rayDirection, newRayDirection, hitInfo.m_normal, hitInfo.m_tangent,
                                       btfColor,
                                       false /*(perRayData.m_printInfo && sampleID == 0)*/);
                    auto origin = hitInfo.m_position;
                    origin += hitInfo.m_normal * 1e-3f;
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
    extern "C" __global__ void __closesthit__IE_SS() {
        SSHit();
    }
#pragma endregion
#pragma region Any hit functions
    extern "C" __global__ void __anyhit__IE_R() {
        const float3 rayDirectionInternal = optixGetWorldRayDirection();
        glm::vec3 rayDirection = glm::vec3(
                rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
#pragma region Retrive information
        const auto &sbtData = *(const SBT *) optixGetSbtDataPointer();
        auto hitInfo = sbtData.GetHitInfo(rayDirection);
#pragma endregion
        switch (sbtData.m_materialType) {
            case MaterialType::Default: {
                glm::vec4 albedoColor =
                        static_cast<SurfaceMaterial *>(sbtData.m_material)->GetAlbedo(hitInfo.m_texCoord);
                if (albedoColor.w <= 0.05f) optixIgnoreIntersection();
            }
                break;
        }
    }
    extern "C" __global__ void __anyhit__IE_SS() {
        SSAnyHit();
    }
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
                    OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_NONE,
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
