﻿#pragma once

#include <LinearCongruenceGenerator.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <optix_device.h>

#include <fstream>
#include <random>
#include <sstream>
#include <vector>

#include <CUDAModule.hpp>
#include <Optix7.hpp>
#include <RayDataDefinations.hpp>
#include "DisneyBssrdf.hpp"

namespace RayTracerFacility {
    typedef LinearCongruenceGenerator<16> Random;
#pragma region Data
    template <typename T>
    struct PerRayData {
        unsigned m_hitCount;
        Random m_random;
        T m_energy;
        glm::vec3 m_normal;
        glm::vec3 m_albedo;
    };
#pragma endregion
#pragma region Helpers

    static __forceinline__ __device__ void *
    UnpackRayDataPointer(const uint32_t &i0, const uint32_t &i1) {
        const uint64_t uPointer = static_cast<uint64_t>(i0) << 32 | i1;
        void *pointer = reinterpret_cast<void *>(uPointer);
        return pointer;
    }

    static __forceinline__ __device__ void
    PackRayDataPointer(void *ptr, uint32_t &i0, uint32_t &i1) {
        const auto uPointer = reinterpret_cast<uint64_t>(ptr);
        i0 = uPointer >> 32;
        i1 = uPointer & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T
    SampleCubeMap(const cudaTextureObject_t cubeMap[], const glm::vec3 &direction) {
        const float absX = abs(direction.x);
        const float absY = abs(direction.y);
        const float absZ = abs(direction.z);
        float ma;
        int faceIndex;
        glm::vec2 uv;
        if (absZ >= absX && absZ >= absY) {
            faceIndex = direction.z < 0.0 ? 5 : 4;
            ma = 0.5f / absZ;
            uv =
                    glm::vec2(direction.z < 0.0 ? -direction.x : direction.x, -direction.y);
        } else if (absY >= absX) {
            faceIndex = direction.y < 0.0 ? 3 : 2;
            ma = 0.5f / absY;
            uv = glm::vec2(direction.x, direction.y > 0.0 ? direction.z : -direction.z);
        } else {
            faceIndex = direction.x < 0.0 ? 1 : 0;
            ma = 0.5f / absX;
            uv =
                    glm::vec2(direction.x < 0.0 ? direction.z : -direction.z, -direction.y);
        }
        uv = uv * ma + glm::vec2(0.5);
        return tex2D<T>(cubeMap[faceIndex], uv.x, uv.y);
    }

    template<typename T>
    static __forceinline__ __device__ T *GetRayDataPointer() {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return static_cast<T *>(UnpackRayDataPointer(u0, u1));
    }

    static __forceinline__ __device__ glm::vec3 Reflect(const glm::vec3 &incident,
                                                        const glm::vec3 &normal) {
        return incident - 2 * glm::dot(incident, normal) * normal;
    }

    static __forceinline__ __device__ glm::vec3
    Refract(const glm::vec3 &incident, const glm::vec3 &normal, const float &ior) {
        float cosI = glm::clamp(glm::dot(incident, normal), -1.0f, 1.0f);
        float etai = 1, etat = ior;
        glm::vec3 n = normal;
        if (cosI < 0) {
            cosI = -cosI;
        } else {
            std::swap(etai, etat);
            n = -normal;
        }
        const float eta = etai / etat;
        const float k = 1 - eta * eta * (1 - cosI * cosI);
        return k < 0 ? glm::vec3(0.0f) : incident * eta + (eta * cosI - sqrtf(k)) * n;
    }

    static __forceinline__ __device__ glm::mat3x3
    GetTangentSpace(const glm::vec3 &normal) {
        // Choose a helper vector for the cross product
        glm::vec3 helper = glm::vec3(1.0f, 0.0f, 0.0f);
        if (abs(normal.x) > 0.99f)
            helper = glm::vec3(0.0f, 0.0f, 1.0f);
        // Generate vectors
        const auto tangent = glm::normalize(cross(normal, helper));
        const auto binormal = glm::normalize(cross(normal, tangent));
        return glm::mat3x3(tangent, binormal, normal);
    }

    static __forceinline__ __device__ glm::vec3
    RandomSampleHemisphere(Random &random, const glm::vec3 &normal,
                           const float &alpha) {
        // Uniformly sample hemisphere direction
        auto cosTheta = 1.0f;
        if (alpha >= 0.0f)
            cosTheta = pow(random(), 1.0f / (alpha + 1.0f));
        const auto sinTheta = sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
        const auto phi = 2.0f * glm::pi<float>() * random();
        const auto tangentSpaceDir =
                glm::vec3(glm::cos(phi) * sinTheta, glm::sin(phi) * sinTheta, cosTheta);
        // Transform direction to world space
        return GetTangentSpace(normal) * tangentSpaceDir;
    }

    static __forceinline__ __device__ glm::vec3
    RandomSampleHemisphere(Random &random, const glm::vec3 &normal) {
        // Uniformly sample hemisphere direction
        auto cosTheta = random();
        const auto sinTheta = sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
        const auto phi = 2.0f * glm::pi<float>() * random();
        const auto tangentSpaceDir =
                glm::vec3(glm::cos(phi) * sinTheta, glm::sin(phi) * sinTheta, cosTheta);
        // Transform direction to world space
        return GetTangentSpace(normal) * tangentSpaceDir;
    }

    static __forceinline__ __device__ glm::vec3 RandomSampleSphere(Random &random) {
        const float theta = 2 * glm::pi<float>() * random();
        const float phi = glm::acos(1.0f - 2.0f * random());
        return glm::vec3(glm::sin(phi) * glm::cos(theta),
                         glm::sin(phi) * glm::sin(theta), glm::cos(phi));
    }

    static __forceinline__ __device__ void
    BRDF(const float &metallic, Random &random, const glm::vec3 &normal,
         const glm::vec3 &hitPoint, const glm::vec3 &in, float3 &origin,
         float3 &out) {
        const glm::vec3 reflected = Reflect(in, normal);
        const glm::vec3 newRayDirection =
                RandomSampleHemisphere(random, reflected, metallic);
        origin =
                make_float3(hitPoint.x + normal.x * 1e-3f, hitPoint.y + normal.y * 1e-3f,
                            hitPoint.z + normal.z * 1e-3f);
        out = make_float3(newRayDirection.x, newRayDirection.y, newRayDirection.z);
    }

    static __forceinline__ __device__ float
    CIESkyIntensity(glm::vec3 rayDir, const glm::vec3 &sunDir, const glm::vec3 &zenith) {
        if (rayDir.y <= 0) {
            rayDir = glm::normalize(glm::vec3(rayDir.x, 0.01f, rayDir.z));
        } else {
            rayDir = glm::normalize(rayDir);
        }
        const float gamma = glm::angle(sunDir, rayDir);
        const float cosGamma = glm::cos(gamma);
        const float cos2Gamma = cosGamma * cosGamma;
        const float theta = glm::angle(zenith, rayDir);
        const float cosTheta = glm::cos(theta);
        const float z0 = glm::angle(zenith, sunDir);
        const float cosz0 = glm::cos(z0);
        const float cos2z0 = cosz0 * cosz0;
        return (0.91f + 10.0f * glm::pow(2.7182818f, -3.0f * gamma) + 0.45f * cos2Gamma)
               * (1.0f - glm::pow(2.7182818f, -0.32f / cosTheta))
               / 0.27f / (0.91f + 10.0f * glm::pow(2.7182818f, -3.0f * z0) + 0.45f * cos2z0);
    }

    static __forceinline__ __device__ bool
    SolveQuadratic(float a, float b, float c, float &x1, float &x2) {
        if (b == 0) {
            // Handle special case where the the two vector ray.dir and V are perpendicular
            // with V = ray.orig - sphere.centre
            if (a == 0) return false;
            x1 = 0;
            x2 = glm::sqrt(-c / a);
            return true;
        }
        float discr = b * b - 4 * a * c;

        if (discr < 0) return false;

        float q = (b < 0.f) ? -0.5f * (b - glm::sqrt(discr)) : -0.5f * (b + glm::sqrt(discr));
        x1 = q / a;
        x2 = c / q;

        return true;
    }

    static __forceinline__ __device__ bool
    RaySphereIntersect(const glm::vec3 &orig, const glm::vec3 &dir, const float &radius, float &t0, float &t1) {
        // They ray dir is normalized so A = 1
        float A = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
        float B = 2.0f * (dir.x * orig.x + dir.y * orig.y + dir.z * orig.z);
        float C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius;

        if (!SolveQuadratic(A, B, C, t0, t1)) return false;

        if (t0 > t1) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        return true;
    }

    /**
     * From https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky/simulating-colors-of-the-sky
     * @param position
     * @param rayDir
     * @param sunDirection
     * @return
     */
    static __forceinline__ __device__ glm::vec3
    NishitaSkyIncidentLight(const glm::vec3 &position, const glm::vec3 &rayDir, const glm::vec3 &sunDirection) {
        float earthRadius = 6360e3;      // In the paper this is usually Rg or Re (radius ground, eart)
        float atmosphereRadius = 6420e3; // In the paper this is usually R or Ra (radius atmosphere)
        float Hr = 7994;               // Thickness of the atmosphere if density was uniform (Hr)
        float Hm = 1200;               // Same as above but for Mie scattering (Hm)

        glm::vec3 betaR = glm::vec3(3.8e-6f, 13.5e-6f, 33.1e-6f);
        glm::vec3 betaM = glm::vec3(21e-6f);
        float tmin = 0;
        float tmax = 999999999999;
        glm::vec3 orig = position + glm::vec3(0.0f, earthRadius, 0.0f);
        float t0, t1;
        if (!RaySphereIntersect(orig, rayDir, atmosphereRadius, t0, t1) || t1 < 0.0f)
            return glm::vec3(0.0f, 0.0f, 0.0f);
        if (t0 > tmin && t0 > 0.0f) tmin = t0;
        if (t1 < tmax) tmax = t1;
        unsigned numSamples = 16;
        unsigned numSamplesLight = 8;
        float segmentLength = (tmax - tmin) / numSamples;
        float tCurrent = tmin;
        glm::vec3 sumR = glm::vec3(0.0f);
        glm::vec3 sumM = glm::vec3(0.0f); // mie and rayleigh contribution
        float opticalDepthR = 0, opticalDepthM = 0;
        float mu = glm::dot(rayDir,
                            sunDirection); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction
        float phaseR = 3.f / (16.f * 3.1415926f) * (1.0f + mu * mu);
        float g = 0.76f;
        float phaseM = 3.f / (8.f * 3.1415926f) * ((1.f - g * g) * (1.f + mu * mu)) /
                       ((2.f + g * g) * glm::pow(1.f + g * g - 2.f * g * mu, 1.5f));
        for (unsigned i = 0; i < numSamples; ++i) {
            glm::vec3 samplePosition = orig + (tCurrent + segmentLength * 0.5f) * rayDir;
            float height = glm::length(samplePosition) - earthRadius;
            // compute optical depth for light
            float hr = glm::exp(-height / Hr) * segmentLength;
            float hm = glm::exp(-height / Hm) * segmentLength;
            opticalDepthR += hr;
            opticalDepthM += hm;
            // light optical depth
            float t0Light, t1Light;
            RaySphereIntersect(samplePosition, sunDirection, atmosphereRadius, t0Light, t1Light);
            float segmentLengthLight = t1Light / numSamplesLight, tCurrentLight = 0;
            float opticalDepthLightR = 0, opticalDepthLightM = 0;
            unsigned j;
            for (j = 0; j < numSamplesLight; ++j) {
                glm::vec3 samplePositionLight =
                        samplePosition + (tCurrentLight + segmentLengthLight * 0.5f) * sunDirection;
                float heightLight = glm::length(samplePositionLight) - earthRadius;
                if (heightLight < 0) break;
                opticalDepthLightR += glm::exp(-heightLight / Hr) * segmentLengthLight;
                opticalDepthLightM += glm::exp(-heightLight / Hm) * segmentLengthLight;
                tCurrentLight += segmentLengthLight;
            }
            if (j == numSamplesLight) {
                glm::vec3 tau = betaR * (opticalDepthR + opticalDepthLightR) +
                                betaM * 1.1f * (opticalDepthM + opticalDepthLightM);
                glm::vec3 attenuation(glm::exp(-tau.x), glm::exp(-tau.y), glm::exp(-tau.z));
                sumR += attenuation * hr;
                sumM += attenuation * hm;
            }
            tCurrent += segmentLength;
        }
        // We use a magic number here for the intensity of the sun (20). We will make it more
        // scientific in a future revision of this lesson/code
        glm::vec3 result = (glm::vec3(sumR.x * betaR.x, sumR.y * betaR.y, sumR.z * betaR.z) * phaseR +
                            glm::vec3(sumM.x * betaM.x, sumM.y * betaM.y, sumM.z * betaM.z) * phaseM) * 20.0f;
        return result;
    }

    static __forceinline__ __device__ glm::vec3
    CalculateEnvironmentalLight(const glm::vec3 &position, const glm::vec3 &rayDir, const Environment &environment) {
        glm::vec3 environmentalLightColor;
        switch (environment.m_environmentalLightingType) {
            case EnvironmentalLightingType::Color:
                environmentalLightColor = environment.m_sunColor;
                break;
            case EnvironmentalLightingType::EnvironmentalMap:
                if (environment.m_environmentalMapId != 0) {
                    float4 color = SampleCubeMap<float4>(
                            environment.m_environmentalMaps,
                            rayDir);
                    environmentalLightColor = glm::vec3(color.x, color.y, color.z);
                }
                break;
            case EnvironmentalLightingType::Skydome:
                environmentalLightColor = NishitaSkyIncidentLight(position, rayDir,
                                                                  environment.m_sunDirection);
                break;
        }
        return environmentalLightColor * environment.m_skylightIntensity;
    }
#pragma endregion
} // namespace RayTracerFacility