//
// Created by lllll on 11/6/2021.
//

#include "LightSensorsGroup.hpp"
#include "RayTracerManager.hpp"
using namespace RayTracerFacility;
void LightSensorsGroup::CalculateIllumination(const RayProperties& rayProperties, int seed, float pushNormalDistance) {
    if (m_lightProbes.empty()) return;
    CudaModule::EstimateIlluminationRayTracing(Application::GetLayer<RayTracerManager>()->m_environmentProperties, rayProperties, m_lightProbes, seed, pushNormalDistance);
}

void LightSensorsGroup::OnInspect() {
    ImGui::Text("Light probes size: %llu", m_lightProbes.size());
}
