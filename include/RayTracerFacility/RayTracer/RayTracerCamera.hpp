#pragma once

#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <CUDAModule.hpp>
#include <ray_tracer_facility_export.h>

using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API RayTracerCamera : public IPrivateComponent{
        friend class RayTracerManager;
        friend class RayTracer;
        glm::ivec2 m_frameSize;
        CameraProperties m_cameraSettings;
        void Resize(const glm::ivec2 &newSize);
        bool m_rendered = false;
    public:

        std::shared_ptr<Texture2D> m_colorTexture;

        void OnInspect() override;

        void OnCreate() override;

        void OnDestroy() override;

    };
}