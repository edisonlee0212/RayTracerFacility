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
        std::vector<Entity> m_entities;
        std::vector<glm::mat4> m_probeTransforms;
        std::vector<glm::vec4> m_probeColors;

        void CalculateIlluminationForDescendents();
        void CalculateIllumination();
        float m_totalArea = 0.0f;
        float m_totalEnergy = 0.0f;
        float m_radiantFlux = 0.0f;
        void OnInspect() override;
    };
} // namespace SorghumFactory
