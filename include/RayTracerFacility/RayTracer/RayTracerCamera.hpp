#pragma once

#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <CUDAModule.hpp>
#include <ray_tracer_facility_export.h>

using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API RayTracerCamera : public IPrivateComponent {
        friend class RayTracerManager;
        friend class RayTracer;
        CameraProperties m_cameraProperties;
        RayProperties m_rayProperties;
        bool m_rendered = false;
        std::shared_ptr<Texture2D> m_colorTexture;

        bool m_allowAutoResize = true;
    public:
        [[nodiscard]] std::shared_ptr<Texture2D> &UnsafeGetColorTexture();
        glm::ivec2 m_frameSize;
        void Ready(const glm::vec3& position, const glm::quat& rotation);

        void OnInspect() override;

        void OnCreate() override;

        void OnDestroy() override;
        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;

        void PostCloneAction(const std::shared_ptr<IPrivateComponent> &target) override;
        RayTracerCamera& operator=(const RayTracerCamera& source);

        void Render();
        void Render(const RayProperties& rayProperties);
        void Render(const RayProperties& rayProperties, const EnvironmentProperties& environmentProperties);
    };
}