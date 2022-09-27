#pragma once

#include <Optix7.hpp>

#include <Vertex.hpp>

#include "MaterialProperties.hpp"
#include <optix_device.h>

namespace RayTracerFacility {
    struct Mesh {
        UniEngine::Vertex *m_vertices;
        glm::uvec3 *m_triangles;

        __device__ void GetVertex(UniEngine::Vertex &target, const float2 &triangleBarycentrics,
                                  const glm::uvec3 &triangleIndices) const {
            const auto &vx = m_vertices[triangleIndices.x];
            const auto &vy = m_vertices[triangleIndices.y];
            const auto &vz = m_vertices[triangleIndices.z];
            target.m_texCoords = (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                                 vx.m_texCoords +
                                 triangleBarycentrics.x * vy.m_texCoords +
                                 triangleBarycentrics.y * vz.m_texCoords;
            target.m_position = (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                                vx.m_position +
                                triangleBarycentrics.x * vy.m_position +
                                triangleBarycentrics.y * vz.m_position;
            target.m_normal = (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                              vx.m_normal +
                              triangleBarycentrics.x * vy.m_normal +
                              triangleBarycentrics.y * vz.m_normal;
            target.m_tangent = (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                               vx.m_tangent +
                               triangleBarycentrics.x * vy.m_tangent +
                               triangleBarycentrics.y * vz.m_tangent;

            auto z = 1.f - triangleBarycentrics.x - triangleBarycentrics.y;
            if (triangleBarycentrics.x > z && triangleBarycentrics.x > triangleBarycentrics.y) {
                target.m_color = vx.m_color;
            } else if (triangleBarycentrics.y > z) {
                target.m_color = vy.m_color;
            } else {
                target.m_color = vz.m_color;
            }
        }

        __device__ glm::uvec3 GetIndices(const int &primitiveId) const {
            return m_triangles[primitiveId];
        }

        __device__ glm::vec2 GetTexCoord(const float2 &triangleBarycentrics,
                                         const glm::uvec3 &triangleIndices) const {
            return (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                   m_vertices[triangleIndices.x].m_texCoords +
                   triangleBarycentrics.x * m_vertices[triangleIndices.y].m_texCoords +
                   triangleBarycentrics.y * m_vertices[triangleIndices.z].m_texCoords;
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

    struct DefaultMaterialTexture {
        cudaTextureObject_t m_texture;
        int m_channel;
    };

    struct DefaultMaterial {
        MaterialProperties m_materialProperties;

        DefaultMaterialTexture m_albedoTexture;
        DefaultMaterialTexture m_normalTexture;
        DefaultMaterialTexture m_metallicTexture;
        DefaultMaterialTexture m_roughnessTexture;

        __device__ glm::vec4 GetAlbedo(const glm::vec2 &texCoord) const {
            if (!m_albedoTexture.m_texture)
                return glm::vec4(m_materialProperties.m_surfaceColor, 1.0f);
            float4 textureAlbedo =
                    tex2D<float4>(m_albedoTexture.m_texture, texCoord.x, texCoord.y);
            return glm::vec4(textureAlbedo.x, textureAlbedo.y, textureAlbedo.z, textureAlbedo.w);
        }

        __device__ float GetRoughness(const glm::vec2 &texCoord) const {
            if (!m_roughnessTexture.m_texture)
                return m_materialProperties.m_roughness;
            return tex2D<float4>(m_roughnessTexture.m_texture, texCoord.x, texCoord.y).x;
        }

        __device__ float GetMetallic(const glm::vec2 &texCoord) const {
            if (!m_metallicTexture.m_texture)
                return m_materialProperties.m_metallic;
            return tex2D<float4>(m_metallicTexture.m_texture, texCoord.x, texCoord.y).x;
        }

        __device__ void ApplyNormalTexture(glm::vec3 &normal,
                                           const glm::vec2 &texCoord,
                                           const glm::vec3 &tangent) const {
            if (!m_normalTexture.m_texture)
                return;
            float4 textureNormal =
                    tex2D<float4>(m_normalTexture.m_texture, texCoord.x, texCoord.y);
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
        GeometryType m_geometryType;
        void *m_geometry;
        MaterialType m_materialType;
        void *m_material;

        __device__ void
        GetGeometricInfo(glm::vec3 &rayDirection, glm::vec2 &texCoord, glm::vec3 &hitPoint, glm::vec3 &normal,
                         glm::vec3 &tangent, glm::vec3 &color) const {
            if (m_geometryType != GeometryType::Curve) {
                const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
                const int primitiveId = optixGetPrimitiveIndex();
                auto *mesh = (Mesh *) m_geometry;
                auto indices = mesh->GetIndices(primitiveId);
                UniEngine::Vertex weightedVertex;
                mesh->GetVertex(weightedVertex, triangleBarycentricsInternal, indices);
                texCoord = weightedVertex.m_texCoords;
                normal = m_globalTransform * glm::vec4(weightedVertex.m_normal, 0.0f);
                if (glm::dot(rayDirection, normal) > 0.0f) {
                    normal = -normal;
                }
                tangent = m_globalTransform * glm::vec4(weightedVertex.m_tangent, 0.0f);
                hitPoint = m_globalTransform * glm::vec4(weightedVertex.m_position, 1.0f);
                color = weightedVertex.m_color;
            }
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
