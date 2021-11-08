
#include <TriangleIlluminationEstimator.hpp>
#include "RayTracerManager.hpp"
using namespace RayTracerFacility;

void TriangleIlluminationEstimator::OnInspect() {
    m_lightSensorsGroup.OnInspect();
    if (ImGui::Button("Calculate illumination")) {
        CalculateIlluminationForDescendents();
    }
    static bool renderProbes = true;
    static float probeSize = 0.02f;
    ImGui::Checkbox("Render Probes", &renderProbes);
    if(renderProbes) {
        ImGui::DragFloat("Size", &probeSize, 0.0001f, 0.0001f, 0.2f, "%.5f");
        RenderManager::DrawGizmoMeshInstancedColored(
                DefaultResources::Primitives::Cube, m_probeColors, m_probeTransforms,
                glm::mat4(1.0f), probeSize);
    }
    ImGui::Text(("Entity Size: " + std::to_string(m_entities.size())).c_str());
    ImGui::Text(("Surface area: " + std::to_string(m_totalArea)).c_str());
    ImGui::Text(("Total energy: " + std::to_string(m_totalEnergy)).c_str());
    ImGui::Text(("Radiant flux: " + std::to_string(m_radiantFlux)).c_str());
}

void TriangleIlluminationEstimator::CalculateIllumination() {
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

void TriangleIlluminationEstimator::CalculateIlluminationForDescendents() {
    m_totalArea = 0.0f;
    m_lightSensorsGroup.m_lightProbes.clear();
    m_entities = GetOwner().GetDescendants();
    m_entities.push_back(GetOwner());
    for (const auto &entity: m_entities) {
        if (entity.HasPrivateComponent<MeshRenderer>()) {
            auto globalTransform = entity.GetDataComponent<GlobalTransform>();
            auto meshRenderer = entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
            auto mesh = meshRenderer->m_mesh.Get<Mesh>();
            auto material = meshRenderer->m_material.Get<Material>();
            for (const auto &triangle: mesh->UnsafeGetTriangles()) {
                auto &vertices = mesh->UnsafeGetVertices();
                const auto position = (vertices[triangle.x].m_position + vertices[triangle.y].m_position +
                                       vertices[triangle.z].m_position) / 3.0f;
                const float a = glm::distance(vertices[triangle.x].m_position, vertices[triangle.y].m_position);
                const float b = glm::distance(vertices[triangle.y].m_position, vertices[triangle.z].m_position);
                const float c = glm::distance(vertices[triangle.z].m_position, vertices[triangle.x].m_position);
                const float p = (a + b + c) * 0.5f;
                const float area = glm::sqrt(p * (p - a) * (p - b) * (p - c));
                m_totalArea += area;
                LightProbe<float> lightProbe;
                lightProbe.m_direction = glm::vec3(0.0f);
                lightProbe.m_energy = 0.0f;
                lightProbe.m_doubleFace = material->m_cullingMode == MaterialCullingMode::Off;
                lightProbe.m_surfaceNormal = glm::normalize(glm::cross(
                        vertices[triangle.x].m_position - vertices[triangle.y].m_position,
                        vertices[triangle.y].m_position - vertices[triangle.z].m_position));
                if(material->m_cullingMode == MaterialCullingMode::Front){
                    lightProbe.m_surfaceNormal = -lightProbe.m_surfaceNormal;
                }
                lightProbe.m_position = globalTransform.m_value * glm::vec4(position, 1.0f);
                m_lightSensorsGroup.m_lightProbes.push_back(lightProbe);
            }
        }else if (entity.HasPrivateComponent<SkinnedMeshRenderer>()) {
            auto globalTransform = entity.GetDataComponent<GlobalTransform>();
            auto skinnedMeshRenderer = entity.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
            auto skinnedMesh = skinnedMeshRenderer->m_skinnedMesh.Get<SkinnedMesh>();
            auto material = skinnedMeshRenderer->m_material.Get<Material>();
            for (const auto &triangle: skinnedMesh->UnsafeGetTriangles()) {
                auto &vertices = skinnedMesh->UnsafeGetSkinnedVertices();
                const auto position = (vertices[triangle.x].m_position + vertices[triangle.y].m_position +
                                       vertices[triangle.z].m_position) / 3.0f;
                const float a = glm::distance(vertices[triangle.x].m_position, vertices[triangle.y].m_position);
                const float b = glm::distance(vertices[triangle.y].m_position, vertices[triangle.z].m_position);
                const float c = glm::distance(vertices[triangle.z].m_position, vertices[triangle.x].m_position);
                const float p = (a + b + c) * 0.5f;
                const float area = glm::sqrt(p * (p - a) * (p - b) * (p - c));
                m_totalArea += area;
                RayTracerFacility::LightProbe<float> lightProbe;
                lightProbe.m_direction = glm::vec3(0.0f);
                lightProbe.m_energy = 0.0f;
                lightProbe.m_doubleFace = material->m_cullingMode == MaterialCullingMode::Off;
                lightProbe.m_surfaceNormal = glm::normalize(glm::cross(
                        vertices[triangle.x].m_position - vertices[triangle.y].m_position,
                        vertices[triangle.y].m_position - vertices[triangle.z].m_position));
                if(material->m_cullingMode == MaterialCullingMode::Front){
                    lightProbe.m_surfaceNormal = -lightProbe.m_surfaceNormal;
                }
                lightProbe.m_position = globalTransform.m_value * glm::vec4(position, 1.0f);
                m_lightSensorsGroup.m_lightProbes.push_back(lightProbe);
            }
        }
    }
    CalculateIllumination();
    size_t i = 0;
    for (const auto &entity: m_entities) {
        if (entity.HasPrivateComponent<MeshRenderer>()) {
            auto meshRenderer = entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
            auto mesh = meshRenderer->m_mesh.Get<Mesh>();
            std::vector<std::pair<size_t, glm::vec4>> colors;
            colors.resize(mesh->GetVerticesAmount());
            for (auto &color: colors) {
                color.first = 0;
                color.second = glm::vec4(0.0f);
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
        }else if (entity.HasPrivateComponent<SkinnedMeshRenderer>()) {
            auto skinnedMeshRenderer = entity.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
            auto skinnedMesh = skinnedMeshRenderer->m_skinnedMesh.Get<SkinnedMesh>();
            std::vector<std::pair<size_t, glm::vec4>> colors;
            colors.resize(skinnedMesh->GetSkinnedVerticesAmount());
            for (auto &i: colors) {
                i.first = 0;
                i.second = glm::vec4(0.0f);
            }
            size_t ti = 0;
            for (const auto &triangle: skinnedMesh->UnsafeGetTriangles()) {
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
            for (auto &vertices: skinnedMesh->UnsafeGetSkinnedVertices()) {
                vertices.m_color = colors[ti].second / static_cast<float>(colors[ti].first);
                ti++;
            }
        }
    }
}
