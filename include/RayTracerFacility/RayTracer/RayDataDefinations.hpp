#pragma once

#include <Optix7.hpp>

#include <Vertex.hpp>

#include "MaterialProperties.hpp"
#include <optix_device.h>
#include "Curve.hpp"

namespace RayTracerFacility {
    static __forceinline__ __device__ float3 GetHitPoint() {
        const float t = optixGetRayTmax();
        const float3 rayOrigin = optixGetWorldRayOrigin();
        const float3 rayDirection = optixGetWorldRayDirection();

        return rayOrigin + rayDirection * t;
    }

    // Compute surface normal of quadratic pimitive in world space.
    static __forceinline__ __device__ float3 NormalLinear(const int primitiveIndex) {
        const OptixTraversableHandle gas = optixGetGASTraversableHandle();
        const unsigned int gasSbtIndex = optixGetSbtGASIndex();
        float4 controlPoints[2];

        optixGetLinearCurveVertexData(gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints);

        LinearBSplineSegment interpolator(controlPoints);
        float3 hitPoint = GetHitPoint();
        // interpolators work in object space
        hitPoint = optixTransformPointFromWorldToObjectSpace(hitPoint);
        const float3 normal = surfaceNormal(interpolator, optixGetCurveParameter(), hitPoint);
        return optixTransformNormalFromObjectToWorldSpace(normal);
    }

    // Compute surface normal of quadratic pimitive in world space.
    static __forceinline__ __device__ float3 NormalQuadratic(const int primitiveIndex) {
        const OptixTraversableHandle gas = optixGetGASTraversableHandle();
        const unsigned int gasSbtIndex = optixGetSbtGASIndex();
        float4 controlPoints[3];

        optixGetQuadraticBSplineVertexData(gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints);

        QuadraticBSplineSegment interpolator(controlPoints);
        float3 hitPoint = GetHitPoint();
        // interpolators work in object space
        hitPoint = optixTransformPointFromWorldToObjectSpace(hitPoint);
        const float3 normal = surfaceNormal(interpolator, optixGetCurveParameter(), hitPoint);
        return optixTransformNormalFromObjectToWorldSpace(normal);
    }

    // Compute surface normal of cubic pimitive in world space.
    static __forceinline__ __device__ float3 NormalCubic(const int primitiveIndex) {
        const OptixTraversableHandle gas = optixGetGASTraversableHandle();
        const unsigned int gasSbtIndex = optixGetSbtGASIndex();
        float4 controlPoints[4];

        optixGetCubicBSplineVertexData(gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints);

        CubicBSplineSegment interpolator(controlPoints);
        float3 hitPoint = GetHitPoint();
        // interpolators work in object space
        hitPoint = optixTransformPointFromWorldToObjectSpace(hitPoint);
        const float3 normal = surfaceNormal(interpolator, optixGetCurveParameter(), hitPoint);
        return optixTransformNormalFromObjectToWorldSpace(normal);
    }

    // Compute normal
    static __forceinline__ __device__ float3 ComputeNormal(OptixPrimitiveType type, const int primitiveIndex) {
        switch (type) {
            case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
                return NormalLinear(primitiveIndex);
            case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
                return NormalQuadratic(primitiveIndex);
            case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
                return NormalCubic(primitiveIndex);
        }
        return make_float3(0.0f);
    }

    struct HitInfo{
        glm::vec3 m_position = glm::vec3(0.0f);
        glm::vec3 m_normal = glm::vec3(0.0f);
        glm::vec3 m_tangent = glm::vec3(0.0f);
        glm::vec3 m_color = glm::vec3(1.0f);
        glm::vec2 m_texCoord = glm::vec2(0.0f);
    };

    struct Curves {
        glm::vec2 *m_strandU = nullptr;
        int *m_strandIndices = nullptr;
        glm::uvec2 *m_strandInfos = nullptr;

        // Get curve hit-point in world coordinates.
        __device__ HitInfo GetHitInfo() const {
            HitInfo hitInfo;
            const unsigned int primitiveIndex = optixGetPrimitiveIndex();
            auto normal = ComputeNormal(optixGetPrimitiveType(), primitiveIndex);
            hitInfo.m_normal = glm::vec3(normal.x, normal.y, normal.z);
            auto hitPoint = GetHitPoint();
            hitInfo.m_position = glm::vec3(hitPoint.x, hitPoint.y, hitPoint.z);
            hitInfo.m_tangent = glm::cross(hitInfo.m_normal, glm::vec3(hitInfo.m_normal.y, hitInfo.m_normal.z, hitInfo.m_normal.x));
            return hitInfo;
        }
    };


    struct TriangularMesh {
        UniEngine::Vertex *m_vertices = nullptr;
        glm::uvec3 *m_triangles = nullptr;

        __device__ HitInfo GetHitInfo() const {
            HitInfo hitInfo;
            const auto triangleBarycentrics = optixGetTriangleBarycentrics();
            const auto primitiveId = optixGetPrimitiveIndex();
            const auto triangleIndices = m_triangles[primitiveId];
            const auto &vx = m_vertices[triangleIndices.x];
            const auto &vy = m_vertices[triangleIndices.y];
            const auto &vz = m_vertices[triangleIndices.z];
            hitInfo.m_texCoord = (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                                 vx.m_texCoord +
                                 triangleBarycentrics.x * vy.m_texCoord +
                                 triangleBarycentrics.y * vz.m_texCoord;
            hitInfo.m_position = (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                                vx.m_position +
                                triangleBarycentrics.x * vy.m_position +
                                triangleBarycentrics.y * vz.m_position;
            hitInfo.m_normal = (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                              vx.m_normal +
                              triangleBarycentrics.x * vy.m_normal +
                              triangleBarycentrics.y * vz.m_normal;
            hitInfo.m_tangent = (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                               vx.m_tangent +
                               triangleBarycentrics.x * vy.m_tangent +
                               triangleBarycentrics.y * vz.m_tangent;

            auto z = 1.f - triangleBarycentrics.x - triangleBarycentrics.y;
            if (triangleBarycentrics.x > z && triangleBarycentrics.x > triangleBarycentrics.y) {
                hitInfo.m_color = vx.m_color;
            } else if (triangleBarycentrics.y > z) {
                hitInfo.m_color = vy.m_color;
            } else {
                hitInfo.m_color = vz.m_color;
            }
            return hitInfo;
        }

        __device__ glm::uvec3 GetIndices(const int &primitiveId) const {
            return m_triangles[primitiveId];
        }

        __device__ glm::vec2 GetTexCoord(const float2 &triangleBarycentrics,
                                         const glm::uvec3 &triangleIndices) const {
            return (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                   m_vertices[triangleIndices.x].m_texCoord +
                   triangleBarycentrics.x * m_vertices[triangleIndices.y].m_texCoord +
                   triangleBarycentrics.y * m_vertices[triangleIndices.z].m_texCoord;
        }

        __device__ glm::vec3
        GetTransformedPosition(const glm::mat4 &globalTransform, const float2 &triangleBarycentrics,
                               const glm::uvec3 &triangleIndices) const {

            return globalTransform * glm::vec4((1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                                               m_vertices[triangleIndices.x].m_position +
                                               triangleBarycentrics.x * m_vertices[triangleIndices.y].m_position +
                                               triangleBarycentrics.y * m_vertices[triangleIndices.z].m_position, 1.0f);
        }

        __device__ glm::vec3 GetPosition(const float2 &triangleBarycentrics,
                                         const glm::uvec3 &triangleIndices) const {
            return (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                   m_vertices[triangleIndices.x].m_position +
                   triangleBarycentrics.x * m_vertices[triangleIndices.y].m_position +
                   triangleBarycentrics.y * m_vertices[triangleIndices.z].m_position;
        }

        __device__ glm::vec3 GetColor(const float2 &triangleBarycentrics,
                                      const glm::uvec3 &triangleIndices) const {
            auto z = 1.f - triangleBarycentrics.x - triangleBarycentrics.y;
            if (triangleBarycentrics.x > z && triangleBarycentrics.x > triangleBarycentrics.y) {
                return m_vertices[triangleIndices.x].m_color;
            } else if (triangleBarycentrics.y > z) {
                return m_vertices[triangleIndices.y].m_color;
            }
            return m_vertices[triangleIndices.z].m_color;
        }

        __device__ glm::vec3 GetTransformedNormal(const glm::mat4 &globalTransform, const float2 &triangleBarycentrics,
                                                  const glm::uvec3 &triangleIndices) const {
            return globalTransform * glm::vec4((1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                                               m_vertices[triangleIndices.x].m_normal +
                                               triangleBarycentrics.x * m_vertices[triangleIndices.y].m_normal +
                                               triangleBarycentrics.y * m_vertices[triangleIndices.z].m_normal, 0.0f);
        }

        __device__ glm::vec3 GetTransformedTangent(const glm::mat4 &globalTransform, const float2 &triangleBarycentrics,
                                                   const glm::uvec3 &triangleIndices) const {
            return globalTransform * glm::vec4((1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                                               m_vertices[triangleIndices.x].m_tangent +
                                               triangleBarycentrics.x * m_vertices[triangleIndices.y].m_tangent +
                                               triangleBarycentrics.y * m_vertices[triangleIndices.z].m_tangent, 0.0f);
        }

        __device__ glm::vec3 GetNormal(const float2 &triangleBarycentrics,
                                       const glm::uvec3 &triangleIndices) const {
            return (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                   m_vertices[triangleIndices.x].m_normal +
                   triangleBarycentrics.x * m_vertices[triangleIndices.y].m_normal +
                   triangleBarycentrics.y * m_vertices[triangleIndices.z].m_normal;
        }

        __device__ glm::vec3 GetTangent(const float2 &triangleBarycentrics,
                                        const glm::uvec3 &triangleIndices) const {
            return (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                   m_vertices[triangleIndices.x].m_tangent +
                   triangleBarycentrics.x * m_vertices[triangleIndices.y].m_tangent +
                   triangleBarycentrics.y * m_vertices[triangleIndices.z].m_tangent;
        }
    };

    struct SurfaceMaterial {
        MaterialProperties m_materialProperties;

        cudaTextureObject_t m_albedoTexture;
        cudaTextureObject_t m_normalTexture;
        cudaTextureObject_t m_metallicTexture;
        cudaTextureObject_t m_roughnessTexture;

        __device__ glm::vec4 GetAlbedo(const glm::vec2 &texCoord) const {
            if (!m_albedoTexture)
                return glm::vec4(m_materialProperties.m_surfaceColor, 1.0f);
            float4 textureAlbedo =
                    tex2D<float4>(m_albedoTexture, texCoord.x, texCoord.y);
            return glm::vec4(textureAlbedo.x, textureAlbedo.y, textureAlbedo.z, textureAlbedo.w);
        }

        __device__ float GetRoughness(const glm::vec2 &texCoord) const {
            if (!m_roughnessTexture)
                return m_materialProperties.m_roughness;
            return tex2D<float4>(m_roughnessTexture, texCoord.x, texCoord.y).x;
        }

        __device__ float GetMetallic(const glm::vec2 &texCoord) const {
            if (!m_metallicTexture)
                return m_materialProperties.m_metallic;
            return tex2D<float4>(m_metallicTexture, texCoord.x, texCoord.y).x;
        }

        __device__ void ApplyNormalTexture(glm::vec3 &normal,
                                           const glm::vec2 &texCoord,
                                           const glm::vec3 &tangent) const {
            if (!m_normalTexture)
                return;
            float4 textureNormal =
                    tex2D<float4>(m_normalTexture, texCoord.x, texCoord.y);
            glm::vec3 B = glm::cross(normal, tangent);
            glm::mat3 TBN = glm::mat3(tangent, B, normal);
            normal =
                    glm::vec3(textureNormal.x, textureNormal.y, textureNormal.z) * 2.0f -
                    glm::vec3(1.0f);
            normal = glm::normalize(TBN * normal);
        }

        __device__ float GetRadiusMax() const { return 0.5f; }
    };

    struct MLVQMaterial {
        BtfBase m_btf;
#pragma region Device functions

        __device__ void ComputeAngles(const glm::vec3 &direction,
                                      const glm::vec3 &normal,
                                      const glm::vec3 &tangent, float &theta,
                                      float &phi) const {
            // transform view & illum vectors into local texture coordinates, i.e.
            // tangent space
            glm::vec3 transformedDir;
            glm::vec3 B = glm::cross(normal, tangent);
            transformedDir[0] = glm::dot(
                    tangent, direction); // T[0]*view[0] + T[1]*view[1] + T[2]*view[2];
            transformedDir[1] =
                    glm::dot(B, direction); // B[0]*view[0] + B[1]*view[1] + B[2]*view[2];
            transformedDir[2] = glm::dot(
                    normal, direction); // N[0]*view[0] + N[1]*view[1] + N[2]*view[2];
            if (isnan(transformedDir[0])) {
                theta = 0.f;
                phi = 0.f;
                return;
            }

            assert(fabs(transformedDir[2]) <= 1.01f);

            if (transformedDir[2] < 0.0) {
                phi = 0.0;
                theta = 90.0;
                return;
            }

            theta = glm::degrees(acosf(transformedDir[2]));

            phi = glm::degrees(atan2(transformedDir[1], transformedDir[0])) + 360.0f;

            if (phi > 360.f)
                phi -= 360.f;
        }

        __device__ void GetValue(const glm::vec2 &texCoord, const glm::vec3 &viewDir,
                                 const glm::vec3 &illuminationDir,
                                 const glm::vec3 &normal, const glm::vec3 tangent,
                                 glm::vec3 &out, const bool &print) const {
            out = glm::vec3(1.0f);
            float illuminationTheta, illuminationPhi, viewTheta, viewPhi;
            ComputeAngles(-viewDir, normal, tangent, viewTheta, viewPhi);
            ComputeAngles(illuminationDir, normal, tangent, illuminationTheta,
                          illuminationPhi);

            if (print) {
                printf("TexCoord[%.2f, %.2f]\n", texCoord.x, texCoord.y);
                printf("Angles[%.1f, %.1f, %.1f, %.1f]\n", illuminationTheta,
                       illuminationPhi, viewTheta, viewPhi);
                printf("Normal[%.2f, %.2f, %.2f]\n", normal.x, normal.y, normal.z);
                printf("View[%.2f, %.2f, %.2f]\n", viewDir.x, viewDir.y, viewDir.z);
                printf("Illumination[%.2f, %.2f, %.2f]\n", illuminationDir.x,
                       illuminationDir.y, illuminationDir.z);
            }
            m_btf.GetValueDeg(texCoord, illuminationTheta, illuminationPhi, viewTheta,
                              viewPhi, out, print);
            out /= 256.0f;
            if (print) {
                printf("ColBase[%.2f, %.2f, %.2f]\n", out.x, out.y, out.z);
            }
        }

#pragma endregion
    };

    struct SBT {
        unsigned long long m_handle;
        glm::mat4 m_globalTransform;
        RendererType m_geometryType;
        void *m_geometry;
        MaterialType m_materialType;
        void *m_material;

        __device__ HitInfo
        GetHitInfo(glm::vec3 &rayDirection) const {
            HitInfo retVal;
            if (m_geometryType != RendererType::Curve) {
                auto *mesh = (TriangularMesh *) m_geometry;
                retVal = mesh->GetHitInfo();
                retVal.m_normal = m_globalTransform * glm::vec4(retVal.m_normal, 0.0f);
                if (glm::dot(rayDirection, retVal.m_normal) > 0.0f) {
                    retVal.m_normal = -retVal.m_normal;
                }
                retVal.m_tangent = m_globalTransform * glm::vec4(retVal.m_tangent, 0.0f);
                retVal.m_position = m_globalTransform * glm::vec4(retVal.m_position, 1.0f);
            }else {
                auto *curves = (Curves *) m_geometry;
                retVal = curves->GetHitInfo();
            }
            return retVal;
        }
    };

/*! SBT record for a raygen program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) CameraRenderingRayGenRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *m_data;
    };

/*! SBT record for a miss program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) CameraRenderingRayMissRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *m_data;
    };

/*! SBT record for a hitgroup program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) CameraRenderingRayHitRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        SBT m_data;
    };

/*! SBT record for a raygen program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    IlluminationEstimationRayGenRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *m_data;
    };

/*! SBT record for a miss program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    IlluminationEstimationRayMissRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *m_data;
    };

/*! SBT record for a hitgroup program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    IlluminationEstimationRayHitRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        SBT m_data;
    };
    /*! SBT record for a raygen program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    PointCloudScanningRayGenRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *m_data;
    };

/*! SBT record for a miss program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    PointCloudScanningRayMissRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *m_data;
    };

/*! SBT record for a hitgroup program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    PointCloudScanningRayHitRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        SBT m_data;
    };
} // namespace RayTracerFacility
