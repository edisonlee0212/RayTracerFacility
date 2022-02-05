#pragma once
#include <glm/glm.hpp>
namespace RayTracerFacility {
    struct MaterialProperties{
        glm::vec3 m_surfaceColor;
        glm::vec3 m_subsurfaceColor;
        float m_subsurfaceRadius;
        float m_subsurfaceFactor;
        float m_roughness = 15.0f;
        float m_metallic = 0.5f;
        float m_emission = 0.0f;
    };

    enum class MaterialType {
        Default,
        VertexColor,
        MLVQ
    };
}