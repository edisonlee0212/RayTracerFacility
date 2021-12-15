//
// Created by lllll on 12/15/2021.
//

#include "PointCloudScanner.hpp"
#include "RayTracerLayer.hpp"
using namespace RayTracerFacility;
void PointCloudScanner::OnInspect() {
    ImGui::DragFloat2("Size", &m_size.x, 0.1f);
    ImGui::DragFloat2("Distance", &m_distance.x, 0.001f);

    static glm::vec4 color = glm::vec4(0, 1, 0, 0.5);
    ImGui::ColorEdit4("Color", &color.x);
    static bool renderPlane = true;
    ImGui::Checkbox("Render plane", &renderPlane);
    auto gt = GetOwner().GetDataComponent<GlobalTransform>();
    if(renderPlane) RenderManager::DrawGizmoMesh(DefaultResources::Primitives::Quad, color, glm::translate(gt.GetPosition()) * glm::mat4_cast(gt.GetRotation()) * glm::scale(glm::vec3(m_size.x, 1.0f, m_size.y)), 1.0f);
}

void PointCloudScanner::CollectAssetRef(std::vector<AssetRef> &list) {

}

void PointCloudScanner::Serialize(YAML::Emitter &out) {

}

void PointCloudScanner::Deserialize(const YAML::Node &in) {

}

void PointCloudScanner::Scan() {
    CudaModule::SamplePointCloud(Application::GetLayer<RayTracerLayer>()->m_environmentProperties, m_samples);
}

void PointCloudScanner::ConstructPointCloud(std::shared_ptr<PointCloud> pointCloud) {

}
