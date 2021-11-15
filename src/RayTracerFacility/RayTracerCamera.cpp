//
// Created by lllll on 11/15/2021.
//

#include "RayTracerCamera.hpp"
#include "Optix7.hpp"

using namespace RayTracerFacility;

void RayTracerCamera::Ready(const glm::vec3& position, const glm::quat& rotation) {
    if(m_cameraSettings.m_frame.m_size != m_frameSize) {
        m_frameSize = glm::max(glm::ivec2(1, 1), m_frameSize);
        m_cameraSettings.Resize(m_frameSize);
        m_colorTexture->UnsafeGetGLTexture()->ReSize(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, m_frameSize.x, m_frameSize.y);
    }
    m_cameraSettings.Set(position, rotation);
}

const char *OutputTypes[]{"Color", "Normal", "Albedo", "DenoisedColor"};
void RayTracerCamera::OnInspect() {
    ImGui::Checkbox("Accumulate", &m_cameraSettings.m_accumulate);
    ImGui::DragFloat("Gamma", &m_cameraSettings.m_gamma,
                     0.01f, 0.1f, 3.0f);
    int outputType = (int)m_cameraSettings.m_outputType;
    if (ImGui::Combo("Output Type", &outputType, OutputTypes,
                     IM_ARRAYSIZE(OutputTypes))) {
        m_cameraSettings.m_outputType = static_cast<OutputType>(outputType);
    }
    ImGui::DragFloat("FOV", &m_cameraSettings.m_fov, 1.0f, 1, 359);
    if (ImGui::TreeNode("Debug"))
    {
        static float debugSacle = 0.25f;
        ImGui::DragFloat("Scale", &debugSacle, 0.01f, 0.1f, 1.0f);
        debugSacle = glm::clamp(debugSacle, 0.1f, 1.0f);
        ImGui::Image(
                (ImTextureID)m_colorTexture->UnsafeGetGLTexture()->Id(),
                ImVec2(m_cameraSettings.m_frame.m_size.x * debugSacle, m_cameraSettings.m_frame.m_size.y * debugSacle),
                ImVec2(0, 1),
                ImVec2(1, 0));
        ImGui::TreePop();
    }
    FileUtils::SaveFile("Screenshot", "Texture2D", {".png", ".jpg"}, [this](const std::filesystem::path &filePath) {
        m_colorTexture->SetPathAndSave(filePath);
    });
    ImGui::Checkbox("Allow auto resize", &m_allowAutoResize);
    if (!m_allowAutoResize)
    {
        ImGui::DragInt2("Resolution", &m_frameSize.x);
    }
}

void RayTracerCamera::OnCreate() {
    m_colorTexture = AssetManager::CreateAsset<Texture2D>();
    m_colorTexture->m_name = "CameraTexture";
    m_colorTexture->UnsafeGetGLTexture() =
            std::make_shared<OpenGLUtils::GLTexture2D>(0, GL_RGBA32F, 1, 1, false);
    m_colorTexture->UnsafeGetGLTexture()->SetData(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0);
    m_colorTexture->UnsafeGetGLTexture()->SetInt(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    m_colorTexture->UnsafeGetGLTexture()->SetInt(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    m_colorTexture->UnsafeGetGLTexture()->SetInt(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    m_colorTexture->UnsafeGetGLTexture()->SetInt(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    m_cameraSettings.m_frame.m_size = glm::ivec2(1, 1);
    m_cameraSettings.m_outputTextureId = m_colorTexture->UnsafeGetGLTexture()->Id();

    m_frameSize = glm::ivec2(2, 2);
    Ready(glm::vec3(0), glm::vec3(0));
}

void RayTracerCamera::OnDestroy() {
}

std::shared_ptr<Texture2D> &RayTracerCamera::UnsafeGetColorTexture() {
    return m_colorTexture;
}
