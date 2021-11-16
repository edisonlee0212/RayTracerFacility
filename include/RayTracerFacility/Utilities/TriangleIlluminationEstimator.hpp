#pragma once
#include "LightSensorsGroup.hpp"
#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <ray_tracer_facility_export.h>
#include <CUDAModule.hpp>

using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API TriangleIlluminationEstimator : public IPrivateComponent {
        LightSensorsGroup m_lightSensorsGroup;
    public:
        std::vector<glm::mat4> m_probeTransforms;
        std::vector<glm::vec4> m_probeColors;
        void CalculateIlluminationForDescendents(const RayProperties& rayProperties, int seed, float pushNormalDistance);
        void CalculateIllumination(const RayProperties& rayProperties, int seed, float pushNormalDistance);
        float m_totalArea = 0.0f;
        float m_totalEnergy = 0.0f;
        float m_radiantFlux = 0.0f;
        void OnInspect() override;

        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
    };


} // namespace SorghumFactory
