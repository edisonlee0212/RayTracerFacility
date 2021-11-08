#pragma once

#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <ray_tracer_facility_export.h>
#include <CUDAModule.hpp>

using namespace UniEngine;
namespace RayTracerFacility {
    struct LightSensorsGroup {
        int m_seed = 0;
        int m_numPointSamples = 100;
        float m_pushNormalDistance = 0.001f;
        std::vector<RayTracerFacility::LightProbe<float>> m_lightProbes;
        void CalculateIllumination();
        void OnInspect();
    };
}