#pragma once

#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <CUDAModule.hpp>
#include <Cubemap.hpp>
#include <EditorManager.hpp>
#include <InputManager.hpp>
#include <MeshRenderer.hpp>
#include <AssetManager.hpp>
#include "ILayer.hpp"
#include <WindowManager.hpp>
#include <memory>
#include <ray_tracer_facility_export.h>

using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API SunlightCalculator {
      SunlightCalculator() = default;
      SunlightCalculator(SunlightCalculator &&) = default;
      SunlightCalculator(const SunlightCalculator &) = default;
      SunlightCalculator &operator=(SunlightCalculator &&) = default;
      SunlightCalculator &operator=(const SunlightCalculator &) = default;
    public:
      float m_intensityFactor = 1.0f;
      /*
       * The database of intensity and angle.
       */
      std::map<float, std::pair<float, float>> m_database;
      static SunlightCalculator &GetInstance();
      static void CalculateSunlightIntensity(int hour, int minute, float& intensity);
      static void CalculateSunlightAngle(int hour, int minute, float& angle);
    };

    class RAY_TRACER_FACILITY_API RayTracerManager : public ILayer {
    protected:
        void UpdateMeshesStorage(std::vector<RayTracerInstance>& meshesStorage, bool &rebuildAccelerationStructure,
                                 bool &updateShaderBindingTable) const;
        void
        UpdateSkinnedMeshesStorage(std::vector<SkinnedRayTracerInstance>& meshesStorage, bool &rebuildAccelerationStructure,
                                   bool &updateShaderBindingTable) const;

        void SceneCameraWindow();

    public:
        AssetRef m_environmentalMap;
        bool m_enableMenus = true;
        EnvironmentProperties m_environmentProperties;


        bool m_renderingEnabled = true;
        float m_lastX = 0;
        float m_lastY = 0;
        float m_lastScrollY = 0;
        bool m_startMouse = false;
        bool m_startScroll = false;
        bool m_rightMouseButtonHold = false;
        float m_resolutionMultiplier = 0.5f;
        std::shared_ptr<RayTracerCamera> m_sceneCamera;
        void UpdateScene() const;

        void OnCreate() override;

        void LateUpdate() override;

        void OnInspect() override;

        void OnDestroy() override;
    };
} // namespace RayTracerFacility