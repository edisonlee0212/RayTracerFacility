//
// Created by lllll on 11/6/2021.
//

#include "LightSensorsGroup.hpp"
#include "RayTracerManager.hpp"
using namespace RayTracerFacility;
void LightSensorsGroup::CalculateIllumination(int seed, float pushNormalDistance, int sampleAmount) {
    if (m_lightProbes.empty()) return;
    CudaModule::EstimateIlluminationRayTracing(Application::GetLayer<RayTracerManager>()->m_defaultRenderingProperties, m_lightProbes, seed, sampleAmount, pushNormalDistance);
}

void LightSensorsGroup::OnInspect() {
    ImGui::Text("Light probes size: %llu", m_lightProbes.size());
}
