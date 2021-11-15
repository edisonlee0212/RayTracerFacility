//
// Created by lllll on 11/15/2021.
//

#include "RayTracerCamera.hpp"
#include "Optix7.hpp"

using namespace RayTracerFacility;

void RayTracerCamera::Resize(const glm::ivec2 &newSize) {
    m_cameraSettings.Resize(newSize);
    m_colorTexture->UnsafeGetGLTexture()->ReSize(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, newSize.x, newSize.y);
}

void RayTracerCamera::OnInspect() {

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
    Resize(glm::ivec2(2, 2));
}

void RayTracerCamera::OnDestroy() {
}
