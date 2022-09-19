
#include <TriangleIlluminationEstimator.hpp>
#include "RayTracerLayer.hpp"
#include "Graphics.hpp"
using namespace RayTracerFacility;

void TriangleIlluminationEstimator::OnInspect() {
    m_lightSensorsGroup.OnInspect();
    static int seed = 0;
    static float pushNormalDistance = 0.001f;
    static RayProperties rayProperties;
    ImGui::DragInt("Seed", &seed);
    ImGui::DragFloat("Normal Distance", &pushNormalDistance, 0.0001f, -1.0f, 1.0f);
    ImGui::DragInt("Samples", &rayProperties.m_samples);
    ImGui::DragInt("Bounces", &rayProperties.m_bounces);
    if (ImGui::Button("Calculate illumination")) {
        CalculateIlluminationForDescendents(rayProperties, seed, pushNormalDistance);
    }
    static bool renderProbes = true;
    static float probeSize = 0.02f;
    ImGui::Checkbox("Render Probes", &renderProbes);
    if(renderProbes) {
        ImGui::DragFloat("Render Size", &probeSize, 0.0001f, 0.0001f, 0.2f, "%.5f");
        Gizmos::DrawGizmoMeshInstancedColored(
                DefaultResources::Primitives::Cube, m_probeColors, m_probeTransforms,
                glm::mat4(1.0f), probeSize);
    }
    ImGui::Text("%s", ("Surface area: " + std::to_string(m_totalArea)).c_str());
    ImGui::Text("%s", ("Total energy: " + std::to_string(m_totalEnergy)).c_str());
    ImGui::Text("%s", ("Radiant flux: " + std::to_string(m_radiantFlux)).c_str());
}

void TriangleIlluminationEstimator::CalculateIllumination(const RayProperties& rayProperties, int seed, float pushNormalDistance) {
#pragma region Illumination estimation
    m_lightSensorsGroup.CalculateIllumination(rayProperties, seed, pushNormalDistance);
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

void TriangleIlluminationEstimator::CalculateIlluminationForDescendents(const RayProperties& rayProperties, int seed, float pushNormalDistance) {
    m_totalArea = 0.0f;
    m_lightSensorsGroup.m_lightProbes.clear();
    auto scene = GetScene();
    auto entities = scene->GetDescendants(GetOwner());
    entities.push_back(GetOwner());
    for (const auto &entity: entities) {
        if (scene->HasPrivateComponent<MeshRenderer>(entity)) {
            auto globalTransform = scene->GetDataComponent<GlobalTransform>(entity);
            auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(entity).lock();
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
                IlluminationSampler<float> lightProbe;
                lightProbe.m_direction = glm::vec3(0.0f);
                lightProbe.m_energy = 0.0f;
                lightProbe.m_doubleFace = material->m_drawSettings.m_cullFace == false;
                lightProbe.m_surfaceNormal = glm::normalize(glm::cross(
                        vertices[triangle.x].m_position - vertices[triangle.y].m_position,
                        vertices[triangle.y].m_position - vertices[triangle.z].m_position));
                if(material->m_drawSettings.m_cullFaceMode == OpenGLCullFace::Front){
                    lightProbe.m_surfaceNormal = -lightProbe.m_surfaceNormal;
                }
                lightProbe.m_position = globalTransform.m_value * glm::vec4(position, 1.0f);
                m_lightSensorsGroup.m_lightProbes.push_back(lightProbe);
            }
        }else if (scene->HasPrivateComponent<SkinnedMeshRenderer>(entity)) {
            auto globalTransform = scene->GetDataComponent<GlobalTransform>(entity);
            auto skinnedMeshRenderer = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(entity).lock();
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
                RayTracerFacility::IlluminationSampler<float> lightProbe;
                lightProbe.m_direction = glm::vec3(0.0f);
                lightProbe.m_energy = 0.0f;
                lightProbe.m_doubleFace = material->m_drawSettings.m_cullFace == false;
                lightProbe.m_surfaceNormal = glm::normalize(glm::cross(
                        vertices[triangle.x].m_position - vertices[triangle.y].m_position,
                        vertices[triangle.y].m_position - vertices[triangle.z].m_position));
                if(material->m_drawSettings.m_cullFaceMode == OpenGLCullFace::Front){
                    lightProbe.m_surfaceNormal = -lightProbe.m_surfaceNormal;
                }
                lightProbe.m_position = globalTransform.m_value * glm::vec4(position, 1.0f);
                m_lightSensorsGroup.m_lightProbes.push_back(lightProbe);
            }
        }
    }
    CalculateIllumination(rayProperties, seed, pushNormalDistance);
    size_t i = 0;
    for (const auto &entity: entities) {
        if (scene->HasPrivateComponent<MeshRenderer>(entity)) {
            auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(entity).lock();
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
        }else if (scene->HasPrivateComponent<SkinnedMeshRenderer>(entity)) {
            auto skinnedMeshRenderer = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(entity).lock();
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

void TriangleIlluminationEstimator::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_totalArea" << YAML::Value << m_totalArea;
    out << YAML::Key << "m_totalEnergy" << YAML::Value << m_totalArea;
    out << YAML::Key << "m_radiantFlux" << YAML::Value << m_totalArea;
}

void TriangleIlluminationEstimator::Deserialize(const YAML::Node &in) {
    m_totalArea = in["m_totalArea"].as<float>();
    m_totalEnergy = in["m_totalEnergy"].as<float>();
    m_radiantFlux = in["m_radiantFlux"].as<float>();
}
