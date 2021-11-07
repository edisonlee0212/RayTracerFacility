
#include <TriangleIlluminationEstimator.hpp>
#include "RayTracerManager.hpp"
using namespace RayTracerFacility;

void TriangleIlluminationEstimator::OnInspect() {
    m_lightSensorsGroup.OnInspect();

    if (ImGui::Button("Calculate illumination")) {
        CalculateIllumination();
    }
    ImGui::Text(("Surface area: " + std::to_string(m_totalArea)).c_str());
    ImGui::Text(("Total energy: " + std::to_string(m_totalEnergy)).c_str());
    ImGui::Text(("Radiant flux: " + std::to_string(m_radiantFlux)).c_str());
}

void TriangleIlluminationEstimator::CalculateIllumination() {
    m_triangleAreas.clear();
    m_totalArea = 0.0f;
#pragma region Illumination estimation
    m_lightSensorsGroup.CalculateIllumination();
    m_probeTransforms.clear();
    m_probeColors.clear();
    m_totalEnergy = 0.0f;
    for (const auto &probe: m_lightSensorsGroup.m_lightProbes) {
        m_probeTransforms.push_back(glm::translate(probe.m_position) * glm::scale(glm::vec3(1.0f)));
        m_totalEnergy += probe.m_energy;
        const float energy = glm::pow(probe.m_energy, 1.0f);
        m_probeColors.emplace_back(energy, energy, energy, 1.0f);
    }
    m_radiantFlux = m_totalEnergy / m_totalArea;
#pragma endregion
}

void TriangleIlluminationEstimator::CollectLightSensorsFromDescendents() {
    m_lightSensorsGroup.m_lightProbes.clear();
    m_entities = GetOwner().GetDescendants();
    m_entities.push_back(GetOwner());
    for (const auto &entity: m_entities) {
        if (entity.HasPrivateComponent<MeshRenderer>()) {
            auto globalTransform = entity.GetDataComponent<GlobalTransform>();
            auto meshRenderer = entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
            auto mesh = meshRenderer->m_mesh.Get<Mesh>();
            for (const auto &triangle: mesh->UnsafeGetTriangles()) {
                auto &vertices = mesh->UnsafeGetVertices();
                const auto position = (vertices[triangle.x].m_position + vertices[triangle.y].m_position +
                                       vertices[triangle.z].m_position) / 3.0f;
                const float a = glm::distance(vertices[triangle.x].m_position, vertices[triangle.y].m_position);
                const float b = glm::distance(vertices[triangle.y].m_position, vertices[triangle.z].m_position);
                const float c = glm::distance(vertices[triangle.z].m_position, vertices[triangle.x].m_position);
                const float p = (a + b + c) * 0.5f;
                const float area = glm::sqrt(p * (p - a) * (p - b) * (p - c));
                m_triangleAreas.push_back(area);
                m_totalArea += area;
                RayTracerFacility::LightProbe<float> lightProbe;
                lightProbe.m_direction = glm::vec3(0.0f);
                lightProbe.m_energy = 0.0f;
                lightProbe.m_surfaceNormal = glm::cross(
                        vertices[triangle.x].m_position - vertices[triangle.y].m_position,
                        vertices[triangle.y].m_position - vertices[triangle.z].m_position);
                lightProbe.m_position = globalTransform.m_value * glm::vec4(position, 1.0f);
                m_lightSensorsGroup.m_lightProbes.push_back(lightProbe);
            }
        }
    }
}

void TriangleIlluminationEstimator::CalculateIlluminationForDescendents() {
    CollectLightSensorsFromDescendents();
    CalculateIllumination();
    size_t i = 0;
    for (const auto &entity: m_entities) {
        if (entity.HasPrivateComponent<MeshRenderer>()) {
            auto meshRenderer = entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
            auto mesh = meshRenderer->m_mesh.Get<Mesh>();
            std::vector<std::pair<size_t, glm::vec4>> colors;
            colors.resize(mesh->GetVerticesAmount());
            for (auto &i: colors) {
                i.first = 0;
                i.second = glm::vec4(0.0f);
            }
            size_t ti = 0;
            for (const auto &triangle: mesh->UnsafeGetTriangles()) {
                const auto color = m_probeColors[i];
                colors[triangle.x].first++;
                colors[triangle.y].first++;
                colors[triangle.z].first++;
                colors[triangle.x].second += color;
                colors[triangle.y].second += color;
                colors[triangle.z].second += color;
                ti++;
                i++;
            }
            ti = 0;
            for (auto &vertices: mesh->UnsafeGetVertices()) {
                vertices.m_color = colors[ti].second / static_cast<float>(colors[ti].first);
                ti++;
            }
        }
    }
}
