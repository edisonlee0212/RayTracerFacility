#pragma once

#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <CUDAModule.hpp>
#include <Cubemap.hpp>
#include <Editor.hpp>
#include <Inputs.hpp>
#include <MeshRenderer.hpp>
#include <ProjectManager.hpp>
#include "ILayer.hpp"
#include "Windows.hpp"
#include <memory>
#include <ray_tracer_facility_export.h>

using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API RayTracerLayer : public ILayer {
    protected:
        void UpdateMeshesStorage(std::map<uint64_t, RayTracedGeometry> &geometryStorage, std::map<uint64_t, RayTracedInstance> &instanceStorage, bool &rebuildInstances,
                                 bool &updateShaderBindingTable) const;

        void SceneCameraWindow();

        void RayCameraWindow();

        friend class RayTracerCamera;

        static std::shared_ptr<RayTracerCamera> m_rayTracerCamera;

        bool CheckMaterial(RayTracerMaterial &rayTracerMaterial, const std::shared_ptr<Material> &material) const;

    public:
        bool m_renderMeshRenderer = true;
        bool m_renderParticles = true;
        bool m_renderMLVQRenderer = true;
        bool m_renderSkinnedMeshRenderer = true;

        bool m_enableMenus = true;
        EnvironmentProperties m_environmentProperties;

        bool m_enableSceneCamera = false;
        bool m_renderingEnabled = true;
        float m_lastX = 0;
        float m_lastY = 0;
        float m_lastScrollY = 0;
        bool m_startMouse = false;
        bool m_startScroll = false;
        bool m_rightMouseButtonHold = false;
        float m_resolutionMultiplier = 0.5f;
        std::shared_ptr<RayTracerCamera> m_sceneCamera;

        void UpdateScene();

        void Update() override;

        void OnCreate() override;

        void LateUpdate() override;

        void OnInspect() override;

        void OnDestroy() override;
    };
} // namespace RayTracerFacility