#include <RayTracerUtilities.cuh>

namespace RayTracerFacility {
    extern "C" __constant__ CameraRenderingLaunchParams cameraRenderingLaunchParams;

#pragma region Closest hit functions
    extern "C" __global__ void __closesthit__CR_R() {
        const float3 rayDirectionInternal = optixGetWorldRayDirection();
        glm::vec3 rayDirection = glm::vec3(
                rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
#pragma region Retrive information
        const auto &sbtData = *(const SBT *) optixGetSbtDataPointer();
        auto hitInfo = sbtData.GetHitInfo(rayDirection);
#pragma endregion
        PerRayData <glm::vec3> &perRayData =
                *GetRayDataPointer < PerRayData < glm::vec3 >> ();
        unsigned hitCount = perRayData.m_hitCount + 1;

        // start with some ambient term
        auto energy = glm::vec3(0.0f);
        uint32_t u0, u1;
        PackRayDataPointer(&perRayData, u0, u1);
        perRayData.m_hitCount = hitCount;
        perRayData.m_energy = glm::vec3(0.0f);
        auto &environment = cameraRenderingLaunchParams.m_rayTracerProperties.m_environment;
        if (sbtData.m_materialType != MaterialType::MLVQ) {
            auto *material = static_cast<SurfaceMaterial *>(sbtData.m_material);
            material->ApplyNormalTexture(hitInfo.m_normal, hitInfo.m_texCoord, hitInfo.m_tangent);
            float metallic =
                    material->GetMetallic(hitInfo.m_texCoord);
            float roughness =
                    material->GetRoughness(hitInfo.m_texCoord);
            glm::vec3 albedoColor;
            if (sbtData.m_materialType == MaterialType::Default) { albedoColor = material->GetAlbedo(hitInfo.m_texCoord); }
            else {
                albedoColor = hitInfo.m_color;
            }
            energy = glm::vec3(0.0f);
            float f = 1.0f;
            if (metallic >= 0.0f)
                f = (metallic + 2) / (metallic + 1);
            if (environment.m_environmentalLightingType == EnvironmentalLightingType::SingleLightSource) {
                glm::vec3 newRayDirection = RandomSampleHemisphere(perRayData.m_random,
                                                                   environment.m_sunDirection,
                                                                   1.0f -
                                                                   environment.m_lightSize);
                energy += glm::vec3(environment.m_color) * environment.m_ambientLightIntensity * albedoColor;
                const float NdotL = glm::dot(
                        hitInfo.m_normal, newRayDirection);
                if (NdotL > 0.0f) {
                    uint32_t u0, u1;
                    PackRayDataPointer(&perRayData, u0, u1);
                    perRayData.m_energy = glm::vec3(0.0f);
                    optixTrace(
                            cameraRenderingLaunchParams.m_traversable,
                            make_float3(hitInfo.m_position.x, hitInfo.m_position.y, hitInfo.m_position.z),
                            make_float3(newRayDirection.x, newRayDirection.y, newRayDirection.z),
                            1e-3f, // tmin
                            1e20f, // tmax
                            0.0f,  // rayTime
                            static_cast<OptixVisibilityMask>(255),
                            OPTIX_RAY_FLAG_DISABLE_ANYHIT
                            | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                            | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                            static_cast<int>(
                                    RayType::Radiance), // SBT offset
                            static_cast<int>(
                                    RayType::RayTypeCount), // SBT stride
                            static_cast<int>(
                                    RayType::Radiance), // missSBTIndex
                            u0, u1);
                    energy += perRayData.m_energy * NdotL * albedoColor;
                }
            } else {
                if (perRayData.m_hitCount <=
                    cameraRenderingLaunchParams.m_rayTracerProperties.m_rayProperties
                            .m_bounces) {
                    bool needSample = false;
                    if (hitCount <= 1 && material->m_materialProperties.m_subsurfaceFactor > 0.0f &&
                        material->m_materialProperties.m_subsurfaceRadius > 0.0f) {
                        float3 incidentRayOrigin;
                        float3 newRayDirectionInternal;
                        glm::vec3 outNormal;
                        needSample = BSSRDF(metallic, perRayData.m_random,
                                            material->m_materialProperties.m_subsurfaceRadius, sbtData.m_handle,
                                            cameraRenderingLaunchParams.m_traversable,
                                            hitInfo.m_position, rayDirection, hitInfo.m_normal,
                                            incidentRayOrigin, newRayDirectionInternal, outNormal);
                        if (needSample) {
                            optixTrace(
                                    cameraRenderingLaunchParams.m_traversable, incidentRayOrigin,
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
                                      material->m_materialProperties.m_subsurfaceColor *
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
                    float3 newRayDirectionInternal;
                    BRDF(metallic, perRayData.m_random, rayDirection, hitInfo.m_normal, newRayDirectionInternal);
                    optixTrace(
                            cameraRenderingLaunchParams.m_traversable,
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
                              albedoColor *
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
                perRayData.m_position = hitInfo.m_position;
            }
            perRayData.m_energy =
                    energy +
                    static_cast<SurfaceMaterial *>(sbtData.m_material)->m_materialProperties.m_emission *
                    albedoColor;

        } else {
            glm::vec3 btfColor;
            if (perRayData.m_hitCount <=
                cameraRenderingLaunchParams.m_rayTracerProperties.m_rayProperties
                        .m_bounces) {
                energy = glm::vec3(0.0f);
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
                        cameraRenderingLaunchParams.m_traversable, incidentRayOrigin,
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
                energy += btfColor * perRayData.m_energy;
            }
            if (hitCount == 1) {
                perRayData.m_normal = hitInfo.m_normal;
                perRayData.m_albedo = btfColor;
                perRayData.m_position = hitInfo.m_position;
            }
            perRayData.m_energy = energy;
        }
    }

    extern "C" __global__ void __closesthit__CR_SS() {
        SSHit();
    }
#pragma endregion
#pragma region Any hit functions
    extern "C" __global__ void __anyhit__CR_R() {
        const float3 rayDirectionInternal = optixGetWorldRayDirection();
        glm::vec3 rayDirection = glm::vec3(
                rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
#pragma region Retrive information
        const auto &sbtData = *(const SBT *) optixGetSbtDataPointer();
        auto hitInfo = sbtData.GetHitInfo(rayDirection);
#pragma endregion
        switch (sbtData.m_materialType) {
            case MaterialType::Default: {
                PerRayData <glm::vec3> &perRayData =
                        *GetRayDataPointer < PerRayData < glm::vec3 >> ();
                glm::vec4 albedoColor =
                        static_cast<SurfaceMaterial *>(sbtData.m_material)->GetAlbedo(hitInfo.m_texCoord);
                if (albedoColor.w <= perRayData.m_random()) optixIgnoreIntersection();
            }
                break;
        }
    }
    extern "C" __global__ void __anyhit__CR_SS() {
        SSAnyHit();
    }
#pragma endregion
#pragma region Miss functions
    extern "C" __global__ void __miss__CR_R() {
        PerRayData <glm::vec3> &perRayData =
                *GetRayDataPointer < PerRayData < glm::vec3 >> ();
        const float3 rayDir = optixGetWorldRayDirection();
        float3 rayOrigin = optixGetWorldRayOrigin();
        glm::vec3 rayOrig = glm::vec3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
        glm::vec3 rayDirection = glm::vec3(rayDir.x, rayDir.y, rayDir.z);
        auto &environment = cameraRenderingLaunchParams.m_rayTracerProperties.m_environment;

        glm::vec3 environmentalLightColor = CalculateEnvironmentalLight(
                rayOrig, rayDirection,
                environment);
        perRayData.m_albedo = perRayData.m_energy = environmentalLightColor;

    }
    extern "C" __global__ void __miss__CR_SS() {

    }
#pragma endregion
#pragma region Main ray generation
    extern "C" __global__ void __raygen__CR() {
        float ix = optixGetLaunchIndex().x;
        float iy = optixGetLaunchIndex().y;
        const uint32_t fbIndex =
                ix + iy * cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x;

        // compute a test pattern based on pixel ID

        PerRayData <glm::vec3> cameraRayData;
        cameraRayData.m_hitCount = 0;
        cameraRayData.m_random.Init(
                ix + cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x * iy,
                cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId);
        cameraRayData.m_energy = glm::vec3(0);
        cameraRayData.m_normal = glm::vec3(0);
        cameraRayData.m_albedo = glm::vec3(0);
        cameraRayData.m_position = glm::vec3(999999.0f);
        // the values we store the PRD pointer in:
        uint32_t u0, u1;
        PackRayDataPointer(&cameraRayData, u0, u1);

        const auto numPixelSamples = cameraRenderingLaunchParams.m_rayTracerProperties
                .m_rayProperties.m_samples;
        auto pixelColor = glm::vec3(0.f);
        auto pixelNormal = glm::vec3(0.f);
        auto pixelAlbedo = glm::vec3(0.f);
        auto pixelPosition = glm::vec3(0.0f);
        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
            // normalized screen plane position, in [0,1]^2
            // iw: note for de-noising that's not actually correct - if we
            // assume that the camera should only(!) cover the de-noised
            // screen then the actual screen plane we should be using during
            // rendering is slightly larger than [0,1]^2
            glm::vec2 screen;
            screen = glm::vec2(ix + cameraRayData.m_random(),
                               iy + cameraRayData.m_random()) /
                     glm::vec2(
                             cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size);
            glm::vec3 primaryRayDir = glm::normalize(
                    cameraRenderingLaunchParams.m_cameraProperties.m_direction +
                    (screen.x - 0.5f) *
                    cameraRenderingLaunchParams.m_cameraProperties.m_horizontal +
                    (screen.y - 0.5f) *
                    cameraRenderingLaunchParams.m_cameraProperties.m_vertical);

            glm::vec3 convergence = cameraRenderingLaunchParams.m_cameraProperties.m_from +
                                    primaryRayDir * cameraRenderingLaunchParams.m_cameraProperties.m_focalLength;
            float angle = cameraRayData.m_random() * 3.1415927f * 2.0f;
            glm::vec3 aperturePoint = cameraRenderingLaunchParams.m_cameraProperties.m_from +
                                      cameraRenderingLaunchParams.m_cameraProperties.m_aperture *
                                      (cameraRenderingLaunchParams.m_cameraProperties.m_horizontal * glm::sin(angle) +
                                       cameraRenderingLaunchParams.m_cameraProperties.m_vertical * glm::cos(angle));
            glm::vec3 rayDir = glm::normalize(convergence - aperturePoint);
            float3 rayOrigin =
                    make_float3(aperturePoint.x,
                                aperturePoint.y,
                                aperturePoint.z);
            float3 rayDirection = make_float3(rayDir.x, rayDir.y, rayDir.z);

            optixTrace(
                    cameraRenderingLaunchParams.m_traversable, rayOrigin, rayDirection,
                    0.f,   // tmin
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
            pixelColor += cameraRayData.m_energy / static_cast<float>(numPixelSamples);
            pixelNormal += cameraRayData.m_normal / static_cast<float>(numPixelSamples);
            pixelAlbedo += cameraRayData.m_albedo / static_cast<float>(numPixelSamples);
            pixelPosition += cameraRayData.m_position / static_cast<float>(numPixelSamples);
            cameraRayData.m_energy = glm::vec3(0.0f);
            cameraRayData.m_normal = glm::vec3(0.0f);
            cameraRayData.m_albedo = glm::vec3(0.0f);
            cameraRayData.m_position = glm::vec3(0.0f);
            cameraRayData.m_hitCount = 0;
        }

        // and write/accumulate to frame buffer ...
        if (cameraRenderingLaunchParams.m_cameraProperties.m_accumulate) {
            if (cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId > 1) {
                glm::vec3 currentGammaCorrectedColor =
                        cameraRenderingLaunchParams.m_cameraProperties.m_frame
                                .m_colorBuffer[fbIndex];
                glm::vec3 accumulatedColor = glm::vec3(glm::pow(
                        currentGammaCorrectedColor,
                        glm::vec3(cameraRenderingLaunchParams.m_cameraProperties.m_gamma)));
                pixelColor +=
                        static_cast<float>(cameraRenderingLaunchParams.m_cameraProperties
                                .m_frame.m_frameId) *
                        accumulatedColor;
                pixelColor /= static_cast<float>(
                        cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId + 1);
            }
        }
        auto gammaCorrectedColor = glm::pow(
                pixelColor,
                glm::vec3(1.0 / cameraRenderingLaunchParams.m_cameraProperties.m_gamma));
        // and write to frame buffer ...
        cameraRenderingLaunchParams.m_cameraProperties.m_frame
                .m_colorBuffer[fbIndex] = glm::vec4(gammaCorrectedColor, 1.0f);
        if (cameraRenderingLaunchParams.m_cameraProperties.m_outputType == OutputType::Depth) {
            float distance = glm::distance(cameraRenderingLaunchParams.m_cameraProperties.m_from, pixelPosition);
            cameraRenderingLaunchParams.m_cameraProperties.m_frame
                    .m_albedoBuffer[fbIndex] = glm::vec4(glm::vec3(
                                                                 glm::clamp(distance / cameraRenderingLaunchParams.m_cameraProperties.m_maxDistance, 0.0f, 1.0f)),
                                                         1.0f);
        } else {
            cameraRenderingLaunchParams.m_cameraProperties.m_frame
                    .m_albedoBuffer[fbIndex] = glm::vec4(pixelAlbedo, 1.0f);
        }
        cameraRenderingLaunchParams.m_cameraProperties.m_frame
                .m_normalBuffer[fbIndex] = glm::vec4(pixelNormal, 1.0f);
    }
#pragma endregion
} // namespace RayTracerFacility
