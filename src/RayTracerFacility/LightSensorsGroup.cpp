//
// Created by lllll on 11/6/2021.
//

#include "LightSensorsGroup.hpp"
#include "RayTracerManager.hpp"
using namespace RayTracerFacility;
void LightSensorsGroup::CalculateIllumination() {
#pragma region Illumination estimation
    if (m_lightProbes.empty()) return;
    CudaModule::EstimateIlluminationRayTracing(Application::GetLayer<RayTracerManager>()->m_defaultWindow.m_defaultRenderingProperties, m_lightProbes, m_seed, m_numPointSamples, m_pushNormalDistance);
}

void LightSensorsGroup::OnInspect() {
    ImGui::Text("Light probes size: %llu", m_lightProbes.size());
    ImGui::DragInt("Seed", &m_seed);
    ImGui::DragInt("Point sample", &m_numPointSamples);
    ImGui::DragFloat("Push Normal Distance", &m_pushNormalDistance, 0.0001f, 0.0f, 1.0f, "%.5f");
}
