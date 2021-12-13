#pragma once

#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <CUDAModule.hpp>
#include <ray_tracer_facility_export.h>

using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API RayTracerCamera : public IPrivateComponent {
        friend class RayTracerLayer;
        friend class RayTracer;
        CameraProperties m_cameraProperties;
        bool m_rendered = false;
    public:
        bool m_allowAutoResize = true;
        std::shared_ptr<Texture2D> m_colorTexture;
        RayProperties m_rayProperties;
        glm::ivec2 m_frameSize;
        void Ready(const glm::vec3& position, const glm::quat& rotation);
        void OnInspect() override;
        void SetFov(float value);
        void SetOutputType(OutputType value);
        void SetAccumulate(bool value);
        void SetGamma(float value);
        void SetDenoiserStrength(float value);
        void OnCreate() override;
        void OnDestroy() override;
        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
        RayTracerCamera& operator=(const RayTracerCamera& source);
        void Render();
        void Render(const RayProperties& rayProperties);
        void Render(const RayProperties& rayProperties, const EnvironmentProperties& environmentProperties);
    };
}