#pragma once

#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <ray_tracer_facility_export.h>
#include <CUDAModule.hpp>
#include <PointCloud.hpp>
using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API PointCloudScanner : public IPrivateComponent {
    public:
        glm::vec2 m_size;
        glm::vec2 m_distance;
        std::vector<PointCloudSample> m_samples;

        void ConstructPointCloud(std::shared_ptr<PointCloud> pointCloud);

        void Scan();

        void OnInspect() override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;
    };
}