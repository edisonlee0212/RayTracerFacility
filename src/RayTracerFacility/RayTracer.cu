#include <RayTracer.hpp>

#include <optix_function_table_definition.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#define GL_TEXTURE_CUBE_MAP 0x8513

#include <cuda_gl_interop.h>

#include <iostream>

#include <RayDataDefinations.hpp>

#include <functional>

#include <filesystem>

#include <imgui.h>
#include <CUDAModule.hpp>

using namespace RayTracerFacility;

void CameraProperties::Set(const glm::vec3 &position, const glm::quat &rotation) {
    auto newFront = glm::normalize(rotation * glm::vec3(0, 0, -1));
    auto newUp = glm::normalize(rotation * glm::vec3(0, 1, 0));
    if (m_from != position || newFront != m_direction || m_up != newUp) m_modified = true;
    m_from = position;
    m_direction = newFront;
    m_up = newUp;
    const float cosFovY = glm::radians(m_fov * 0.5f);
    const float aspect = static_cast<float>(m_frame.m_size.x) / static_cast<float>(m_frame.m_size.y);
    m_horizontal =
            cosFovY * aspect *
            glm::normalize(glm::cross(m_direction, newUp));
    m_vertical = cosFovY * glm::normalize(m_up);
}

void CameraProperties::Resize(const glm::ivec2 &newSize) {
    if (m_frame.m_size == newSize) return;
    m_frame.m_size = newSize;
    m_modified = true;
    if (m_denoiser) {
        OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
    };
    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiserOptions = {};
    OPTIX_CHECK(optixDenoiserCreate(CudaModule::GetRayTracer()->m_optixDeviceContext,
                                    OPTIX_DENOISER_MODEL_KIND_LDR,
                                    &denoiserOptions, &m_denoiser));
    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(
            m_denoiser, m_frame.m_size.x, m_frame.m_size.y, &denoiserReturnSizes));

    m_denoiserScratch.Resize(
            std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
                     denoiserReturnSizes.withoutOverlapScratchSizeInBytes));

    m_denoiserState.Resize(denoiserReturnSizes.stateSizeInBytes);
    // ------------------------------------------------------------------
    // resize our cuda frame buffer
    m_denoisedBuffer.Resize(m_frame.m_size.x * m_frame.m_size.y * sizeof(glm::vec4));
    m_frameBufferColor.Resize(m_frame.m_size.x * m_frame.m_size.y * sizeof(glm::vec4));
    m_frameBufferNormal.Resize(m_frame.m_size.x * m_frame.m_size.y * sizeof(glm::vec4));
    m_frameBufferAlbedo.Resize(m_frame.m_size.x * m_frame.m_size.y * sizeof(glm::vec4));

    // update the launch parameters that we'll pass to the optix
    // launch:
    m_frame.m_colorBuffer =
            (glm::vec4 *) m_frameBufferColor.DevicePointer();
    m_frame.m_normalBuffer =
            (glm::vec4 *) m_frameBufferNormal.DevicePointer();
    m_frame.m_albedoBuffer =
            (glm::vec4 *) m_frameBufferAlbedo.DevicePointer();

    // ------------------------------------------------------------------
    OPTIX_CHECK(optixDenoiserSetup(
            m_denoiser, 0, m_frame.m_size.x, m_frame.m_size.y, m_denoiserState.DevicePointer(),
            m_denoiserState.m_sizeInBytes, m_denoiserScratch.DevicePointer(),
            m_denoiserScratch.m_sizeInBytes));
}

void CameraProperties::SetFov(float value) {
    m_modified = true;
    m_fov = value;
}

const char *OutputTypes[]{"Color", "Normal", "Albedo", "Depth"};

void CameraProperties::OnInspect() {
    if (ImGui::TreeNode("Camera Properties")) {
        ImGui::Checkbox("Accumulate", &m_accumulate);
        ImGui::DragFloat("Gamma", &m_gamma,
                         0.01f, 0.1f, 3.0f);
        int outputType = (int) m_outputType;
        if (ImGui::Combo("Output Type", &outputType, OutputTypes,
                         IM_ARRAYSIZE(OutputTypes))) {
            m_outputType = static_cast<OutputType>(outputType);
        }
        if (ImGui::DragFloat("Max Distance", &m_maxDistance, 0.1f, 0.1f, 10000.0f)) {
            SetMaxDistance(m_maxDistance);
        }

        if (ImGui::DragFloat("FOV", &m_fov, 1.0f, 1, 359)) {
            SetFov(m_fov);
        }
        if (ImGui::DragFloat("Aperture", &m_aperture, 0.0001f, 0.0f, 99999.0f, "%.4f")) {
            SetAperture(m_aperture);
        }
        if (ImGui::DragFloat("Focal Length", &m_focalLength, 0.0001f, 0.0f, 99999.0f, "%.4f")) {
            SetFocalLength(m_focalLength);
        }
        if (ImGui::DragFloat("Denoiser Strength", &m_denoiserStrength, 0.01f, 0.0f, 1.0f)) {
            SetDenoiserStrength(m_denoiserStrength);
        }
        ImGui::TreePop();
    }
}

void CameraProperties::SetDenoiserStrength(float value) {
    m_denoiserStrength = glm::clamp(value, 0.0f, 1.0f);
}

void CameraProperties::SetGamma(float value) {
    m_modified = true;
    m_gamma = value;
}

void CameraProperties::SetOutputType(OutputType value) {
    m_modified = true;
    m_outputType = value;
}

void CameraProperties::SetAperture(float value) {
    m_modified = true;
    m_aperture = value;
}

void CameraProperties::SetFocalLength(float value) {
    m_modified = true;
    m_focalLength = value;
}

void CameraProperties::SetMaxDistance(float value) {
    m_maxDistance = value;
    m_modified = true;
}

const char *EnvironmentalLightingTypes[]{"Scene", "Skydome", "SingleLightSource"};

void EnvironmentProperties::OnInspect() {
    static int type = 0;
    if (ImGui::Combo("Environment Lighting", &type, EnvironmentalLightingTypes,
                     IM_ARRAYSIZE(EnvironmentalLightingTypes))) {
        m_environmentalLightingType = static_cast<EnvironmentalLightingType>(type);
    }
    if (m_environmentalLightingType == EnvironmentalLightingType::Skydome) {
        if (ImGui::TreeNodeEx("Atmosphere Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::DragFloat("Earth Radius (km)", &m_atmosphere.m_earthRadius, 1.0f, 0.0f,
                                 m_atmosphere.m_atmosphereRadius - 1.0f)) {
                m_atmosphere.m_earthRadius = glm::clamp(m_atmosphere.m_earthRadius, 1.0f,
                                                        m_atmosphere.m_atmosphereRadius - 1.0f);
            }
            if (ImGui::DragFloat("Atmosphere Radius (km)", &m_atmosphere.m_atmosphereRadius, 1.0f,
                                 m_atmosphere.m_earthRadius + 1.0f, 100000.0f)) {
                m_atmosphere.m_atmosphereRadius = glm::clamp(m_atmosphere.m_atmosphereRadius,
                                                             m_atmosphere.m_earthRadius + 1.0f, 100000.0f);
            }
            if (ImGui::DragFloat("Rayleigh scale height (m)", &m_atmosphere.m_Hr, 1.0f, 0.0f, 100000.0f)) {
                m_atmosphere.m_Hr = glm::clamp(m_atmosphere.m_Hr, 0.0f, 10000.0f);
            }
            if (ImGui::DragFloat("Mie scale height (m)", &m_atmosphere.m_Hm, 1.0f, 0.0f, 100000.0f)) {
                m_atmosphere.m_Hm = glm::clamp(m_atmosphere.m_Hm, 0.0f, 10000.0f);
            }
            if (ImGui::DragFloat("Mie scattering mean cosine", &m_atmosphere.m_g, 0.001f, 0.0f, 0.999f, "%.4f")) {
                m_atmosphere.m_g = glm::clamp(m_atmosphere.m_g, 0.0f, 0.999f);
            }
            if (ImGui::DragInt("Samples", &m_atmosphere.m_numSamples, 1, 128)) {
                m_atmosphere.m_numSamples = glm::clamp(m_atmosphere.m_numSamples, 1, 128);
            }
            if (ImGui::DragInt("Samples light", &m_atmosphere.m_numSamplesLight, 1, 128)) {
                m_atmosphere.m_numSamplesLight = glm::clamp(m_atmosphere.m_numSamplesLight, 1, 128);
            }
            ImGui::TreePop();
        }
        if (ImGui::Button("Reset Atmosphere")) {
            m_atmosphere.m_earthRadius = 6360;      // In the paper this is usually Rg or Re (radius ground, eart)
            m_atmosphere.m_atmosphereRadius = 6420; // In the paper this is usually R or Ra (radius atmosphere)
            m_atmosphere.m_Hr = 7994;               // Thickness of the atmosphere if density was uniform (Hr)
            m_atmosphere.m_Hm = 1200;               // Same as above but for Mie scattering (Hm)
            m_atmosphere.m_g = 0.76f;               // Mean cosine for Mie scattering
            m_atmosphere.m_numSamples = 16;
            m_atmosphere.m_numSamplesLight = 8;
        }
    }else if(m_environmentalLightingType == EnvironmentalLightingType::SingleLightSource){
        if (ImGui::DragFloat("Light Size", &m_lightSize, 0.001f, 0.0f, 1.0f)) {
            m_lightSize = glm::clamp(m_lightSize, 0.0f, 1.0f);
        }
        if (ImGui::DragFloat("Ambient light intensity", &m_ambientLightIntensity, 0.001f, 0.0f, 1.0f)) {
            m_ambientLightIntensity = glm::clamp(m_ambientLightIntensity, 0.0f, 1.0f);
        }
        static glm::vec3 angles = glm::vec3(90, 0, 0);
        if(ImGui::DragFloat3("Sun angle", &angles.x, 1.0f)){
            m_sunDirection = glm::quat(glm::radians(angles)) * glm::vec3(0, 0, -1);
        }
    }
}

void RayProperties::OnInspect() {
    if (ImGui::TreeNode("Ray Properties")) {
        ImGui::DragInt("bounce limit", &m_bounces, 1, 1, 8);
        ImGui::DragInt("pixel samples", &m_samples, 1, 1, 64);
        ImGui::TreePop();
    }
}


void RayTracerProperties::OnInspect() {
    m_environment.OnInspect();
    m_rayProperties.OnInspect();
}

bool RayTracer::RenderToCamera(const EnvironmentProperties &environmentProperties, CameraProperties &cameraProperties,
                               const RayProperties &rayProperties) {
    if (cameraProperties.m_frame.m_size.x == 0 | cameraProperties.m_frame.m_size.y == 0)
        return true;
    if (!m_hasAccelerationStructure)
        return false;
    std::vector<std::pair<unsigned, std::pair<cudaTextureObject_t, int>>> boundTextures;
    std::vector<cudaGraphicsResource_t> boundResources;
    BuildShaderBindingTable(boundTextures, boundResources);
    bool statusChanged = false;
    if (m_requireUpdate) statusChanged = true;
    m_cameraRenderingLaunchParams.m_cameraProperties = cameraProperties;
    statusChanged = statusChanged || cameraProperties.m_modified;
    cameraProperties.m_modified = false;
    if (m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.Changed(environmentProperties)) {
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment = environmentProperties;
        statusChanged = true;
    }
    if (m_cameraRenderingLaunchParams.m_rayTracerProperties.m_rayProperties.Changed(rayProperties)) {
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_rayProperties = rayProperties;
        statusChanged = true;
    }
    if (!m_cameraRenderingLaunchParams.m_cameraProperties.m_accumulate || statusChanged) {
        m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId = 0;
        cameraProperties.m_frame.m_frameId = 0;
    }
#pragma region Bind environmental map as cudaTexture
    struct cudaResourceDesc cudaResourceDesc;
    cudaArray_t environmentalMapPosXArray;
    cudaArray_t environmentalMapNegXArray;
    cudaArray_t environmentalMapPosYArray;
    cudaArray_t environmentalMapNegYArray;
    cudaArray_t environmentalMapPosZArray;
    cudaArray_t environmentalMapNegZArray;
    cudaGraphicsResource_t environmentalMapTexture;
    if (m_cameraRenderingLaunchParams.m_rayTracerProperties
                .m_environment.m_environmentalMapId != 0) {
        CUDA_CHECK(GraphicsGLRegisterImage(
                &environmentalMapTexture,
                m_cameraRenderingLaunchParams.m_rayTracerProperties
                        .m_environment.m_environmentalMapId,
                GL_TEXTURE_CUBE_MAP, cudaGraphicsRegisterFlagsNone));
        CUDA_CHECK(GraphicsMapResources(1, &environmentalMapTexture, nullptr));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapPosXArray, environmentalMapTexture,
                cudaGraphicsCubeFacePositiveX, 0));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapNegXArray, environmentalMapTexture,
                cudaGraphicsCubeFaceNegativeX, 0));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapPosYArray, environmentalMapTexture,
                cudaGraphicsCubeFacePositiveY, 0));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapNegYArray, environmentalMapTexture,
                cudaGraphicsCubeFaceNegativeY, 0));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapPosZArray, environmentalMapTexture,
                cudaGraphicsCubeFacePositiveZ, 0));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapNegZArray, environmentalMapTexture,
                cudaGraphicsCubeFaceNegativeZ, 0));
        memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
        cudaResourceDesc.resType = cudaResourceTypeArray;
        struct cudaTextureDesc cudaTextureDesc;
        memset(&cudaTextureDesc, 0, sizeof(cudaTextureDesc));
        cudaTextureDesc.addressMode[0] = cudaAddressModeWrap;
        cudaTextureDesc.addressMode[1] = cudaAddressModeWrap;
        cudaTextureDesc.filterMode = cudaFilterModeLinear;
        cudaTextureDesc.readMode = cudaReadModeElementType;
        cudaTextureDesc.normalizedCoords = 1;
        // Create texture object
        cudaResourceDesc.res.array.array = environmentalMapPosXArray;
        CUDA_CHECK(CreateTextureObject(
                &m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[0],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
        cudaResourceDesc.res.array.array = environmentalMapNegXArray;
        CUDA_CHECK(CreateTextureObject(
                &m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[1],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
        cudaResourceDesc.res.array.array = environmentalMapPosYArray;
        CUDA_CHECK(CreateTextureObject(
                &m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[2],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
        cudaResourceDesc.res.array.array = environmentalMapNegYArray;
        CUDA_CHECK(CreateTextureObject(
                &m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[3],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
        cudaResourceDesc.res.array.array = environmentalMapPosZArray;
        CUDA_CHECK(CreateTextureObject(
                &m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[4],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
        cudaResourceDesc.res.array.array = environmentalMapNegZArray;
        CUDA_CHECK(CreateTextureObject(
                &m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[5],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
    } else {
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[0] = 0;
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[1] = 0;
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[2] = 0;
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[3] = 0;
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[4] = 0;
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[5] = 0;
    }
#pragma endregion
#pragma region Upload parameters
    m_cameraRenderingPipeline.m_launchParamsBuffer.Upload(
            &m_cameraRenderingLaunchParams, 1);
    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId++;
    cameraProperties.m_frame.m_frameId++;
#pragma endregion
#pragma region Launch rays from camera
    OPTIX_CHECK(
            optixLaunch(/*! pipeline we're launching launch: */
                    m_cameraRenderingPipeline.m_pipeline, m_stream,
                    /*! parameters and SBT */
                    m_cameraRenderingPipeline.m_launchParamsBuffer
                            .DevicePointer(),
                    m_cameraRenderingPipeline.m_launchParamsBuffer.m_sizeInBytes,
                    &m_cameraRenderingPipeline.m_sbt,
                    /*! dimensions of the launch: */
                    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x,
                    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.y,
                    1));
#pragma endregion
    CUDA_SYNC_CHECK();
#pragma region Remove textures binding.
    if (m_cameraRenderingLaunchParams.m_rayTracerProperties
                .m_environment.m_environmentalMapId != 0) {
        CUDA_CHECK(DestroyTextureObject(
                m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[0]));
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[0] = 0;
        CUDA_CHECK(DestroyTextureObject(
                m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[1]));
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[1] = 0;
        CUDA_CHECK(DestroyTextureObject(
                m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[2]));
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[2] = 0;
        CUDA_CHECK(DestroyTextureObject(
                m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[3]));
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[3] = 0;
        CUDA_CHECK(DestroyTextureObject(
                m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[4]));
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[4] = 0;
        CUDA_CHECK(DestroyTextureObject(
                m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[5]));
        m_cameraRenderingLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[5] = 0;

        CUDA_CHECK(GraphicsUnmapResources(1, &environmentalMapTexture, 0));
        CUDA_CHECK(GraphicsUnregisterResource(environmentalMapTexture));
    }
    for (int i = 0; i < boundResources.size(); i++) {
        CUDA_CHECK(DestroySurfaceObject(boundTextures[i].second.first));
        CUDA_CHECK(GraphicsUnmapResources(1, &boundResources[i], 0));
        CUDA_CHECK(GraphicsUnregisterResource(boundResources[i]));
    }
#pragma endregion
#pragma region Bind output texture
    cudaArray_t outputArray;
    cudaGraphicsResource_t outputTexture;
    CUDA_CHECK(GraphicsGLRegisterImage(
            &outputTexture,
            m_cameraRenderingLaunchParams.m_cameraProperties.m_outputTextureId,
            GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    CUDA_CHECK(GraphicsMapResources(1, &outputTexture, nullptr));
    CUDA_CHECK(
            GraphicsSubResourceGetMappedArray(&outputArray, outputTexture, 0, 0));
    /*
    // Specify surface
    memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
    cudaResourceDesc.resType = cudaResourceTypeArray;
    // Create the surface objects
    cudaResourceDesc.res.array.array = outputArray;
    // Create surface object
    cudaSurfaceObject_t outputTextureId;
    CUDA_CHECK(CreateSurfaceObject(&outputTextureId, &cudaResourceDesc));
    */
#pragma endregion
#pragma region Copy results to output texture
    OptixImage2D inputLayer[3];
    inputLayer[0].data = m_cameraRenderingLaunchParams.m_cameraProperties.m_frameBufferColor.DevicePointer();
    /// Width of the image (in pixels)
    inputLayer[0].width =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x;
    /// Height of the image (in pixels)
    inputLayer[0].height =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[0].rowStrideInBytes =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x * sizeof(glm::vec4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels
    /// (no gaps) is supported.
    inputLayer[0].pixelStrideInBytes = sizeof(glm::vec4);
    /// Pixel format.
    inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // ..................................................................
    inputLayer[1].data = m_cameraRenderingLaunchParams.m_cameraProperties.m_frameBufferAlbedo.DevicePointer();
    /// Width of the image (in pixels)
    inputLayer[1].width =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x;
    /// Height of the image (in pixels)
    inputLayer[1].height =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[1].rowStrideInBytes =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x * sizeof(glm::vec4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels
    /// (no gaps) is supported.
    inputLayer[1].pixelStrideInBytes = sizeof(glm::vec4);
    /// Pixel format.
    inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // ..................................................................
    inputLayer[2].data = m_cameraRenderingLaunchParams.m_cameraProperties.m_frameBufferNormal.DevicePointer();
    /// Width of the image (in pixels)
    inputLayer[2].width =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x;
    /// Height of the image (in pixels)
    inputLayer[2].height =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[2].rowStrideInBytes =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x * sizeof(glm::vec4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels
    /// (no gaps) is supported.
    inputLayer[2].pixelStrideInBytes = sizeof(glm::vec4);
    /// Pixel format.
    inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = m_cameraRenderingLaunchParams.m_cameraProperties.m_denoisedBuffer.DevicePointer();
    /// Width of the image (in pixels)
    outputLayer.width =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x;
    /// Height of the image (in pixels)
    outputLayer.height =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.y;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes =
            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x * sizeof(glm::vec4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels
    /// (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(glm::vec4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    switch (m_cameraRenderingLaunchParams.m_cameraProperties.m_outputType) {
        case OutputType::Color: {
            if (cameraProperties.m_denoiserStrength == 0.0f) {
                CUDA_CHECK(MemcpyToArray(
                        outputArray, 0, 0,
                        (void *) m_cameraRenderingLaunchParams
                                .m_cameraProperties.m_frame.m_colorBuffer,
                        sizeof(glm::vec4) *
                        m_cameraRenderingLaunchParams
                                .m_cameraProperties.m_frame.m_size.x *
                        m_cameraRenderingLaunchParams
                                .m_cameraProperties.m_frame.m_size.y,
                        cudaMemcpyDeviceToDevice));
            } else {
                OptixDenoiserParams denoiserParams;
                denoiserParams.denoiseAlpha = 1;
                m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiserIntensity.Resize(sizeof(float));
                if (m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiserIntensity.m_sizeInBytes !=
                    sizeof(float))
                    m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiserIntensity.Resize(sizeof(float));
                denoiserParams.hdrIntensity =
                        m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiserIntensity.DevicePointer();
                if (m_cameraRenderingLaunchParams.m_cameraProperties.m_accumulate &&
                    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId > 1)
                    denoiserParams.blendFactor =
                            (1.0f - cameraProperties.m_denoiserStrength) /
                            m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId;
                else
                    denoiserParams.blendFactor = (1.0f - cameraProperties.m_denoiserStrength);

                OPTIX_CHECK(optixDenoiserComputeIntensity(
                        m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiser,
                        /*stream*/ 0, &inputLayer[0],
                        (CUdeviceptr) m_cameraRenderingLaunchParams
                                .m_cameraProperties.m_denoiserIntensity.DevicePointer(),
                        (CUdeviceptr) m_cameraRenderingLaunchParams
                                .m_cameraProperties.m_denoiserScratch.DevicePointer(),
                        m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiserScratch.m_sizeInBytes));

                OptixDenoiserLayer denoiserLayer = {};
                denoiserLayer.input = inputLayer[0];
                denoiserLayer.output = outputLayer;

                OptixDenoiserGuideLayer denoiserGuideLayer = {};
                denoiserGuideLayer.albedo = inputLayer[1];
                denoiserGuideLayer.normal = inputLayer[2];

                OPTIX_CHECK(optixDenoiserInvoke(
                        m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiser,
                        /*stream*/ 0, &denoiserParams,
                        m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiserState.DevicePointer(),
                        m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiserState.m_sizeInBytes,
                        &denoiserGuideLayer, &denoiserLayer, 1,
                        /*inputOffsetX*/ 0,
                        /*inputOffsetY*/ 0,
                        m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiserScratch.DevicePointer(),
                        m_cameraRenderingLaunchParams.m_cameraProperties.m_denoiserScratch.m_sizeInBytes));
                CUDA_CHECK(MemcpyToArray(outputArray, 0, 0, (void *) outputLayer.data,
                                         sizeof(glm::vec4) *
                                         m_cameraRenderingLaunchParams
                                                 .m_cameraProperties.m_frame.m_size.x *
                                         m_cameraRenderingLaunchParams
                                                 .m_cameraProperties.m_frame.m_size.y,
                                         cudaMemcpyDeviceToDevice));
            }
        }
            break;
        case OutputType::Normal: {
            CUDA_CHECK(MemcpyToArray(
                    outputArray, 0, 0,
                    (void *) m_cameraRenderingLaunchParams
                            .m_cameraProperties.m_frame.m_normalBuffer,
                    sizeof(glm::vec4) *
                    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x *
                    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.y,
                    cudaMemcpyDeviceToDevice));
        }
            break;
        case OutputType::Albedo: {
            CUDA_CHECK(MemcpyToArray(
                    outputArray, 0, 0,
                    (void *) m_cameraRenderingLaunchParams
                            .m_cameraProperties.m_frame.m_albedoBuffer,
                    sizeof(glm::vec4) *
                    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x *
                    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.y,
                    cudaMemcpyDeviceToDevice));
        }
            break;
        case OutputType::Depth: {
            CUDA_CHECK(MemcpyToArray(
                    outputArray, 0, 0,
                    (void *) m_cameraRenderingLaunchParams
                            .m_cameraProperties.m_frame.m_albedoBuffer,
                    sizeof(glm::vec4) *
                    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x *
                    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.y,
                    cudaMemcpyDeviceToDevice));
        }
            break;
    }

#pragma endregion
#pragma region UnBind output texture
    // CUDA_CHECK(DestroySurfaceObject(outputTextureId));
    CUDA_CHECK(GraphicsUnmapResources(1, &outputTexture, 0));
    CUDA_CHECK(GraphicsUnregisterResource(outputTexture));
#pragma endregion

    return true;
}

void RayTracer::EstimateIllumination(const size_t &size,
                                     const EnvironmentProperties &environmentProperties,
                                     const RayProperties &rayProperties,
                                     CudaBuffer &lightProbes, unsigned seed,
                                     float pushNormalDistance) {
    if (!m_hasAccelerationStructure)
        return;
    if (size == 0) {
        std::cout << "Error: Lightprobe is empty" << std::endl;
        return;
    }
    std::vector<std::pair<unsigned, std::pair<cudaTextureObject_t, int>>> boundTextures;
    std::vector<cudaGraphicsResource_t> boundResources;
    BuildShaderBindingTable(boundTextures, boundResources);
#pragma region Bind environmental map as cudaTexture
    struct cudaResourceDesc cudaResourceDesc;
    cudaArray_t environmentalMapPosXArray;
    cudaArray_t environmentalMapNegXArray;
    cudaArray_t environmentalMapPosYArray;
    cudaArray_t environmentalMapNegYArray;
    cudaArray_t environmentalMapPosZArray;
    cudaArray_t environmentalMapNegZArray;
    cudaGraphicsResource_t environmentalMapTexture;
    if (m_illuminationEstimationLaunchParams.m_rayTracerProperties
                .m_environment.m_environmentalMapId != 0) {
        CUDA_CHECK(GraphicsGLRegisterImage(
                &environmentalMapTexture,
                m_illuminationEstimationLaunchParams.m_rayTracerProperties
                        .m_environment.m_environmentalMapId,
                GL_TEXTURE_CUBE_MAP, cudaGraphicsRegisterFlagsNone));
        CUDA_CHECK(GraphicsMapResources(1, &environmentalMapTexture, nullptr));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapPosXArray, environmentalMapTexture,
                cudaGraphicsCubeFacePositiveX, 0));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapNegXArray, environmentalMapTexture,
                cudaGraphicsCubeFaceNegativeX, 0));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapPosYArray, environmentalMapTexture,
                cudaGraphicsCubeFacePositiveY, 0));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapNegYArray, environmentalMapTexture,
                cudaGraphicsCubeFaceNegativeY, 0));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapPosZArray, environmentalMapTexture,
                cudaGraphicsCubeFacePositiveZ, 0));
        CUDA_CHECK(GraphicsSubResourceGetMappedArray(
                &environmentalMapNegZArray, environmentalMapTexture,
                cudaGraphicsCubeFaceNegativeZ, 0));
        memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
        cudaResourceDesc.resType = cudaResourceTypeArray;
        struct cudaTextureDesc cudaTextureDesc;
        memset(&cudaTextureDesc, 0, sizeof(cudaTextureDesc));
        cudaTextureDesc.addressMode[0] = cudaAddressModeWrap;
        cudaTextureDesc.addressMode[1] = cudaAddressModeWrap;
        cudaTextureDesc.filterMode = cudaFilterModeLinear;
        cudaTextureDesc.readMode = cudaReadModeElementType;
        cudaTextureDesc.normalizedCoords = 1;
        // Create texture object
        cudaResourceDesc.res.array.array = environmentalMapPosXArray;
        CUDA_CHECK(CreateTextureObject(
                &m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[0],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
        cudaResourceDesc.res.array.array = environmentalMapNegXArray;
        CUDA_CHECK(CreateTextureObject(
                &m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[1],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
        cudaResourceDesc.res.array.array = environmentalMapPosYArray;
        CUDA_CHECK(CreateTextureObject(
                &m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[2],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
        cudaResourceDesc.res.array.array = environmentalMapNegYArray;
        CUDA_CHECK(CreateTextureObject(
                &m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[3],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
        cudaResourceDesc.res.array.array = environmentalMapPosZArray;
        CUDA_CHECK(CreateTextureObject(
                &m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[4],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
        cudaResourceDesc.res.array.array = environmentalMapNegZArray;
        CUDA_CHECK(CreateTextureObject(
                &m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[5],
                &cudaResourceDesc, &cudaTextureDesc, nullptr));
    } else {
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[0] = 0;
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[1] = 0;
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[2] = 0;
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[3] = 0;
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[4] = 0;
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[5] = 0;
    }
#pragma endregion
#pragma region Upload parameters
    m_illuminationEstimationLaunchParams.m_seed = seed;
    m_illuminationEstimationLaunchParams.m_pushNormalDistance = pushNormalDistance;
    m_illuminationEstimationLaunchParams.m_size = size;
    m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment = environmentProperties;
    m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_rayProperties = rayProperties;
    m_illuminationEstimationLaunchParams.m_lightProbes =
            reinterpret_cast<IlluminationSampler<float> *>(lightProbes.DevicePointer());
    m_illuminationEstimationPipeline.m_launchParamsBuffer.Upload(
            &m_illuminationEstimationLaunchParams, 1);
#pragma endregion
#pragma region Launch rays from light probes
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
            m_illuminationEstimationPipeline.m_pipeline,
            m_stream,
            /*! parameters and SBT */
            m_illuminationEstimationPipeline
                    .m_launchParamsBuffer.DevicePointer(),
            m_illuminationEstimationPipeline
                    .m_launchParamsBuffer.m_sizeInBytes,
            &m_illuminationEstimationPipeline.m_sbt,
            /*! dimensions of the launch: */
            size, 1, 1));
    CUDA_SYNC_CHECK();
#pragma endregion
#pragma region Remove textures binding.
    if (m_illuminationEstimationLaunchParams.m_rayTracerProperties
                .m_environment.m_environmentalMapId != 0) {
        CUDA_CHECK(DestroyTextureObject(
                m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[0]));
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[0] = 0;
        CUDA_CHECK(DestroyTextureObject(
                m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[1]));
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[1] = 0;
        CUDA_CHECK(DestroyTextureObject(
                m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[2]));
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[2] = 0;
        CUDA_CHECK(DestroyTextureObject(
                m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[3]));
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[3] = 0;
        CUDA_CHECK(DestroyTextureObject(
                m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[4]));
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[4] = 0;
        CUDA_CHECK(DestroyTextureObject(
                m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[5]));
        m_illuminationEstimationLaunchParams.m_rayTracerProperties.m_environment.m_environmentalMaps[5] = 0;

        CUDA_CHECK(GraphicsUnmapResources(1, &environmentalMapTexture, 0));
        CUDA_CHECK(GraphicsUnregisterResource(environmentalMapTexture));
    }
    for (int i = 0; i < boundResources.size(); i++) {
        CUDA_CHECK(DestroySurfaceObject(boundTextures[i].second.first));
        CUDA_CHECK(GraphicsUnmapResources(1, &boundResources[i], 0));
        CUDA_CHECK(GraphicsUnregisterResource(boundResources[i]));
    }
#pragma endregion
}

void RayTracer::ScanPointCloud(const size_t &size, const EnvironmentProperties &environmentProperties,
                               CudaBuffer &samples) {
    if (!m_hasAccelerationStructure)
        return;
    if (size == 0) {
        std::cout << "Error: Samples is empty" << std::endl;
        return;
    }
    std::vector<std::pair<unsigned, std::pair<cudaTextureObject_t, int>>> boundTextures;
    std::vector<cudaGraphicsResource_t> boundResources;
    BuildShaderBindingTable(boundTextures, boundResources);
#pragma region Upload parameters
    m_pointCloudScanningLaunchParams.m_size = size;
    m_pointCloudScanningLaunchParams.m_rayTracerProperties.m_environment = environmentProperties;
    m_pointCloudScanningLaunchParams.m_rayTracerProperties.m_rayProperties = RayProperties();
    m_pointCloudScanningLaunchParams.m_samples =
            reinterpret_cast<PointCloudSample *>(samples.DevicePointer());
    m_pointCloudScanningPipeline.m_launchParamsBuffer.Upload(
            &m_pointCloudScanningLaunchParams, 1);
#pragma endregion
#pragma region Launch rays from samples
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
            m_pointCloudScanningPipeline.m_pipeline,
            m_stream,
            /*! parameters and SBT */
            m_pointCloudScanningPipeline.m_launchParamsBuffer.DevicePointer(),
            m_pointCloudScanningPipeline.m_launchParamsBuffer.m_sizeInBytes,
            &m_pointCloudScanningPipeline.m_sbt,
            /*! dimensions of the launch: */
            size, 1, 1));
    CUDA_SYNC_CHECK();
#pragma endregion
#pragma region Remove textures binding.
    for (int i = 0; i < boundResources.size(); i++) {
        CUDA_CHECK(DestroySurfaceObject(boundTextures[i].second.first));
        CUDA_CHECK(GraphicsUnmapResources(1, &boundResources[i], 0));
        CUDA_CHECK(GraphicsUnregisterResource(boundResources[i]));
    }
#pragma endregion
}

RayTracer::RayTracer() {
    m_cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId = 0;
    // std::cout << "#Optix: creating optix context ..." << std::endl;
    CreateContext();
    // std::cout << "#Optix: setting up module ..." << std::endl;
    CreateModules();
    // std::cout << "#Optix: creating raygen programs ..." << std::endl;
    CreateRayGenPrograms();
    // std::cout << "#Optix: creating miss programs ..." << std::endl;
    CreateMissPrograms();
    // std::cout << "#Optix: creating hitgroup programs ..." << std::endl;
    CreateHitGroupPrograms();
    // std::cout << "#Optix: setting up optix pipeline ..." << std::endl;
    AssemblePipelines();

    std::cout << "#Optix: context, module, pipeline, etc, all set up ..."
              << std::endl;

    MLVQMaterialStorage storage;
    storage.m_material = std::make_shared<MLVQMaterial>();
    storage.m_buffer.Upload(storage.m_material.get(), 1);
    m_MLVQMaterialStorage.push_back(storage);
}

static void context_log_cb(const unsigned int level, const char *tag,
                           const char *message, void *) {
    fprintf(stderr, "[%2d][%12s]: %s\n", static_cast<int>(level), tag, message);
}

void RayTracer::CreateContext() {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(StreamCreate(&m_stream));
    CUDA_CHECK(GetDeviceProperties(&m_deviceProps, deviceID));
    std::cout << "#Optix: running on device: " << m_deviceProps.name << std::endl;
    const CUresult cuRes = cuCtxGetCurrent(&m_cudaContext);
    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);
    OPTIX_CHECK(
            optixDeviceContextCreate(m_cudaContext, nullptr, &m_optixDeviceContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(m_optixDeviceContext,
                                                 context_log_cb, nullptr, 4));
}

extern "C" char CAMERA_RENDERING_PTX[];
extern "C" char ILLUMINATION_ESTIMATION_PTX[];
extern "C" char POINT_CLOUD_SCANNING_PTX[];

void RayTracer::CreateModules() {
    CreateModule(m_cameraRenderingPipeline, CAMERA_RENDERING_PTX,
                 "cameraRenderingLaunchParams");
    CreateModule(m_illuminationEstimationPipeline,
                 ILLUMINATION_ESTIMATION_PTX,
                 "illuminationEstimationLaunchParams");
    CreateModule(m_pointCloudScanningPipeline,
                 POINT_CLOUD_SCANNING_PTX,
                 "pointCloudScanningLaunchParams");
}

void RayTracer::CreateRayGenPrograms() {
    CreateRayGenProgram(m_cameraRenderingPipeline, "__raygen__CR");
    CreateRayGenProgram(m_illuminationEstimationPipeline,
                        "__raygen__IE");
    CreateRayGenProgram(m_pointCloudScanningPipeline,
                        "__raygen__pointCloudScanning");
}

void RayTracer::CreateMissPrograms() {
    {
        m_cameraRenderingPipeline.m_missProgramGroups.resize(
                static_cast<int>(RayType::RayTypeCount));
        char log[2048];
        size_t sizeofLog = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module = m_cameraRenderingPipeline.m_module;

        // ------------------------------------------------------------------
        // radiance rays
        // ------------------------------------------------------------------
        pgDesc.miss.entryFunctionName = "__miss__CR_R";

        OPTIX_CHECK(optixProgramGroupCreate(
                m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
                &m_cameraRenderingPipeline.m_missProgramGroups[static_cast<int>(
                        RayType::Radiance)]));
        if (sizeofLog > 1)
            std::cout << log << std::endl;
        // ------------------------------------------------------------------
        // BSSRDF Spatial sampler rays
        // ------------------------------------------------------------------
        pgDesc.miss.entryFunctionName = "__miss__CR_SS";
        OPTIX_CHECK(optixProgramGroupCreate(
                m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
                &m_cameraRenderingPipeline.m_missProgramGroups[static_cast<int>(
                        RayType::SpacialSampling)]));
        if (sizeofLog > 1)
            std::cout << log << std::endl;
    }
    {
        m_illuminationEstimationPipeline.m_missProgramGroups.resize(
                static_cast<int>(RayType::RayTypeCount));
        char log[2048];
        size_t sizeofLog = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module = m_illuminationEstimationPipeline.m_module;

        // ------------------------------------------------------------------
        // radiance rays
        // ------------------------------------------------------------------
        pgDesc.miss.entryFunctionName = "__miss__IE_R";

        OPTIX_CHECK(optixProgramGroupCreate(
                m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
                &m_illuminationEstimationPipeline
                        .m_missProgramGroups[static_cast<int>(
                        RayType::Radiance)]));
        if (sizeofLog > 1)
            std::cout << log << std::endl;
        // ------------------------------------------------------------------
        // BSSRDF Spatial sampler rays
        // ------------------------------------------------------------------
        pgDesc.miss.entryFunctionName = "__miss__IE_SS";
        OPTIX_CHECK(optixProgramGroupCreate(
                m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
                &m_illuminationEstimationPipeline.m_missProgramGroups[static_cast<int>(
                        RayType::SpacialSampling)]));
        if (sizeofLog > 1)
            std::cout << log << std::endl;
    }
    {
        m_pointCloudScanningPipeline.m_missProgramGroups.resize(
                static_cast<int>(RayType::RayTypeCount) - 1);
        char log[2048];
        size_t sizeofLog = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module = m_pointCloudScanningPipeline.m_module;

        // ------------------------------------------------------------------
        // radiance rays
        // ------------------------------------------------------------------
        pgDesc.miss.entryFunctionName = "__miss__pointCloudScanning";

        OPTIX_CHECK(optixProgramGroupCreate(
                m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
                &m_pointCloudScanningPipeline.m_missProgramGroups[static_cast<int>(
                        RayType::Radiance)]));
        if (sizeofLog > 1)
            std::cout << log << std::endl;
    }
}

void RayTracer::CreateHitGroupPrograms() {
    {
        m_cameraRenderingPipeline.m_hitGroupProgramGroups.resize(
                static_cast<int>(RayType::RayTypeCount));
        char log[2048];
        size_t sizeofLog = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = m_cameraRenderingPipeline.m_module;
        pgDesc.hitgroup.moduleAH = m_cameraRenderingPipeline.m_module;
        // -------------------------------------------------------
        // radiance rays
        // -------------------------------------------------------
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__CR_R";
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__CR_R";
        OPTIX_CHECK(optixProgramGroupCreate(
                m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
                &m_cameraRenderingPipeline.m_hitGroupProgramGroups[static_cast<int>(
                        RayType::Radiance)]));
        if (sizeofLog > 1)
            std::cout << log << std::endl;

        // -------------------------------------------------------
        // BSSRDF Sampler ray
        // -------------------------------------------------------
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__CR_SS";
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__CR_SS";

        OPTIX_CHECK(optixProgramGroupCreate(
                m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
                &m_cameraRenderingPipeline.m_hitGroupProgramGroups[static_cast<int>(
                        RayType::SpacialSampling)]));
        if (sizeofLog > 1)
            std::cout << log << std::endl;
    }
    {
        m_illuminationEstimationPipeline.m_hitGroupProgramGroups.resize(
                static_cast<int>(RayType::RayTypeCount));
        char log[2048];
        size_t sizeofLog = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = m_illuminationEstimationPipeline.m_module;
        pgDesc.hitgroup.moduleAH = m_illuminationEstimationPipeline.m_module;
        // -------------------------------------------------------
        // radiance rays
        // -------------------------------------------------------
        pgDesc.hitgroup.entryFunctionNameCH =
                "__closesthit__IE_R";
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__IE_R";
        OPTIX_CHECK(optixProgramGroupCreate(
                m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
                &m_illuminationEstimationPipeline
                        .m_hitGroupProgramGroups[static_cast<int>(
                        RayType::Radiance)]));
        if (sizeofLog > 1)
            std::cout << log << std::endl;
        // -------------------------------------------------------
        // BSSRDF Sampler ray
        // -------------------------------------------------------
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__IE_SS";
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__IE_SS";

        OPTIX_CHECK(optixProgramGroupCreate(
                m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
                &m_illuminationEstimationPipeline.m_hitGroupProgramGroups[static_cast<int>(
                        RayType::SpacialSampling)]));
        if (sizeofLog > 1)
            std::cout << log << std::endl;
    }
    {
        m_pointCloudScanningPipeline.m_hitGroupProgramGroups.resize(
                static_cast<int>(RayType::RayTypeCount) - 1);
        char log[2048];
        size_t sizeofLog = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = m_pointCloudScanningPipeline.m_module;
        pgDesc.hitgroup.moduleAH = m_pointCloudScanningPipeline.m_module;
        // -------------------------------------------------------
        // radiance rays
        // -------------------------------------------------------
        pgDesc.hitgroup.entryFunctionNameCH =
                "__closesthit__pointCloudScanning";
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__pointCloudScanning";
        OPTIX_CHECK(optixProgramGroupCreate(
                m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
                &m_pointCloudScanningPipeline
                        .m_hitGroupProgramGroups[static_cast<int>(
                        RayType::Radiance)]));
        if (sizeofLog > 1)
            std::cout << log << std::endl;
    }
}

__global__ void
ApplyTransformKernelInstanced(int matricesSize, int verticesSize, glm::mat4 globalTransform, glm::mat4 *matrices,
                              Vertex *vertices,
                              glm::vec3 *targetPositions, glm::vec3 *targetNormals, glm::vec4 *targetColors,
                              glm::vec3 *targetTangents, glm::vec2 *targetTexCoords) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < verticesSize * matricesSize) {
        targetPositions[idx] =
                globalTransform * matrices[idx / verticesSize] *
                glm::vec4(vertices[idx % verticesSize].m_position, 1.0f);
        glm::vec3 N = glm::normalize(globalTransform * matrices[idx / verticesSize] *
                                     glm::vec4(vertices[idx % verticesSize].m_normal, 0.0f));
        glm::vec3 T = glm::normalize(globalTransform * matrices[idx / verticesSize] *
                                     glm::vec4(vertices[idx % verticesSize].m_tangent, 0.0f));
        T = glm::normalize(T - dot(T, N) * N);
        targetNormals[idx] = N;
        targetTangents[idx] = T;
        targetTexCoords[idx] = vertices[idx % verticesSize].m_texCoords;
        targetColors[idx] = vertices[idx % verticesSize].m_color;
    }
}

__global__ void
ApplyTransformKernel(int size, glm::mat4 globalTransform, Vertex *vertices,
                     glm::vec3 *targetPositions, glm::vec3 *targetNormals, glm::vec4 *targetColors,
                     glm::vec3 *targetTangents, glm::vec2 *targetTexCoords) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        targetPositions[idx] =
                globalTransform * glm::vec4(vertices[idx].m_position, 1.0f);
        glm::vec3 N = glm::normalize(globalTransform *
                                     glm::vec4(vertices[idx].m_normal, 0.0f));
        glm::vec3 T = glm::normalize(globalTransform *
                                     glm::vec4(vertices[idx].m_tangent, 0.0f));
        T = glm::normalize(T - dot(T, N) * N);
        targetNormals[idx] = N;
        targetTangents[idx] = T;
        targetTexCoords[idx] = vertices[idx].m_texCoords;
        targetColors[idx] = vertices[idx].m_color;
    }
}

__global__ void ApplySkinnedTransformKernel(int size, glm::mat4 globalTransform,
                                            SkinnedVertex *vertices,
                                            glm::mat4 *boneMatrices,
                                            glm::vec3 *targetPositions,
                                            glm::vec3 *targetNormals, glm::vec4 *targetColors,
                                            glm::vec3 *targetTangents,
                                            glm::vec2 *targetTexCoords) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        glm::mat4 boneTransform =
                boneMatrices[vertices[idx].m_bondId[0]] * vertices[idx].m_weight[0];
        if (vertices[idx].m_bondId[1] != -1) {
            boneTransform +=
                    boneMatrices[vertices[idx].m_bondId[1]] * vertices[idx].m_weight[1];
        }
        if (vertices[idx].m_bondId[2] != -1) {
            boneTransform +=
                    boneMatrices[vertices[idx].m_bondId[2]] * vertices[idx].m_weight[2];
        }
        if (vertices[idx].m_bondId[3] != -1) {
            boneTransform +=
                    boneMatrices[vertices[idx].m_bondId[3]] * vertices[idx].m_weight[3];
        }
        if (vertices[idx].m_bondId2[0] != -1) {
            boneTransform +=
                    boneMatrices[vertices[idx].m_bondId2[0]] * vertices[idx].m_weight2[0];
        }
        if (vertices[idx].m_bondId2[1] != -1) {
            boneTransform +=
                    boneMatrices[vertices[idx].m_bondId2[1]] * vertices[idx].m_weight2[1];
        }
        if (vertices[idx].m_bondId2[2] != -1) {
            boneTransform +=
                    boneMatrices[vertices[idx].m_bondId2[2]] * vertices[idx].m_weight2[2];
        }
        if (vertices[idx].m_bondId2[3] != -1) {
            boneTransform +=
                    boneMatrices[vertices[idx].m_bondId2[3]] * vertices[idx].m_weight2[3];
        }
        auto finalTransform = globalTransform * boneTransform;
        targetPositions[idx] =
                finalTransform * glm::vec4(vertices[idx].m_position, 1.0f);
        glm::vec3 N = glm::normalize(finalTransform *
                                     glm::vec4(vertices[idx].m_normal, 0.0f));
        glm::vec3 T = glm::normalize(finalTransform *
                                     glm::vec4(vertices[idx].m_tangent, 0.0f));
        T = glm::normalize(T - dot(T, N) * N);
        targetNormals[idx] = N;
        targetTangents[idx] = T;
        targetTexCoords[idx] = vertices[idx].m_texCoords;
        targetColors[idx] = vertices[idx].m_color;
    }
}

void RayTracer::BuildAccelerationStructure() {
    int instanceSize = m_instances.size() + m_skinnedInstances.size();
    for (auto &i: m_trianglesBuffer)
        i.Free();
    for (auto &i: m_transformedPositionsBuffer)
        i.Free();
    for (auto &i: m_transformedNormalsBuffer)
        i.Free();
    for (auto &i: m_transformedTangentBuffer)
        i.Free();
    for (auto &i: m_texCoordBuffer)
        i.Free();
    for (auto &i: m_vertexColorBuffer)
        i.Free();
    for (auto &i: m_boneMatricesBuffer)
        i.Free();

    m_trianglesBuffer.clear();
    m_transformedPositionsBuffer.clear();
    m_transformedNormalsBuffer.clear();
    m_transformedTangentBuffer.clear();
    m_texCoordBuffer.clear();
    m_vertexColorBuffer.clear();
    m_boneMatricesBuffer.clear();

    m_trianglesBuffer.resize(instanceSize);
    m_transformedPositionsBuffer.resize(instanceSize);
    m_transformedNormalsBuffer.resize(instanceSize);
    m_transformedTangentBuffer.resize(instanceSize);
    m_texCoordBuffer.resize(instanceSize);
    m_vertexColorBuffer.resize(instanceSize);
    m_boneMatricesBuffer.resize(instanceSize);

    OptixTraversableHandle asHandle = 0;

    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(instanceSize);
    std::vector<CUdeviceptr> deviceVertexPositions(instanceSize);
    std::vector<CUdeviceptr> deviceVertexTriangles(instanceSize);
    std::vector<CUdeviceptr> deviceTransforms(instanceSize);
    std::vector<uint32_t> triangleInputFlags(instanceSize);
    int meshID = 0;
    CudaBuffer verticesBuffer;
    CudaBuffer matricesBuffer;
    for (; meshID < m_instances.size(); meshID++) {
        // upload the model to the device: the builder
        MeshInstance &triangleMesh = m_instances[meshID];
        verticesBuffer.Upload(*triangleMesh.m_mesh.m_vertices);
        if (triangleMesh.m_instancing) {
            matricesBuffer.Upload(*triangleMesh.m_matrices);
            m_transformedPositionsBuffer[meshID].Resize(triangleMesh.m_matrices->size() *
                                                        triangleMesh.m_mesh.m_vertices->size() * sizeof(glm::vec3));
            m_transformedNormalsBuffer[meshID].Resize(triangleMesh.m_matrices->size() *
                                                      triangleMesh.m_mesh.m_vertices->size() * sizeof(glm::vec3));
            m_transformedTangentBuffer[meshID].Resize(triangleMesh.m_matrices->size() *
                                                      triangleMesh.m_mesh.m_vertices->size() * sizeof(glm::vec3));
            m_texCoordBuffer[meshID].Resize(triangleMesh.m_matrices->size() *
                                            triangleMesh.m_mesh.m_vertices->size() * sizeof(glm::vec2));
            m_vertexColorBuffer[meshID].Resize(triangleMesh.m_matrices->size() *
                                               triangleMesh.m_mesh.m_vertices->size() * sizeof(glm::vec4));
        } else {
            m_transformedPositionsBuffer[meshID].Resize(
                    triangleMesh.m_mesh.m_vertices->size() * sizeof(glm::vec3));
            m_transformedNormalsBuffer[meshID].Resize(
                    triangleMesh.m_mesh.m_vertices->size() * sizeof(glm::vec3));
            m_transformedTangentBuffer[meshID].Resize(
                    triangleMesh.m_mesh.m_vertices->size() * sizeof(glm::vec3));
            m_texCoordBuffer[meshID].Resize(triangleMesh.m_mesh.m_vertices->size() *
                                            sizeof(glm::vec2));
            m_vertexColorBuffer[meshID].Resize(triangleMesh.m_mesh.m_vertices->size() * sizeof(glm::vec4));
        }

        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the
        // maximum occupancy for a full device launch
        int gridSize = 0;    // The actual grid size needed, based on input size
        if (triangleMesh.m_instancing) {
            int size = triangleMesh.m_matrices->size() * triangleMesh.m_mesh.m_vertices->size();
            int matricesSize = triangleMesh.m_matrices->size();
            int verticesSize = triangleMesh.m_mesh.m_vertices->size();
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                               ApplyTransformKernelInstanced, 0, size);
            gridSize = (size + blockSize - 1) / blockSize;
            ApplyTransformKernelInstanced<<<gridSize, blockSize>>>(
                    matricesSize, verticesSize, triangleMesh.m_globalTransform,
                    static_cast<glm::mat4 *>(matricesBuffer.m_dPtr),
                    static_cast<Vertex *>(verticesBuffer.m_dPtr),
                    static_cast<glm::vec3 *>(m_transformedPositionsBuffer[meshID].m_dPtr),
                    static_cast<glm::vec3 *>(m_transformedNormalsBuffer[meshID].m_dPtr),
                    static_cast<glm::vec4 *>(m_vertexColorBuffer[meshID].m_dPtr),
                    static_cast<glm::vec3 *>(m_transformedTangentBuffer[meshID].m_dPtr),
                    static_cast<glm::vec2 *>(m_texCoordBuffer[meshID].m_dPtr));
        } else {
            int size = triangleMesh.m_mesh.m_vertices->size();
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                               ApplyTransformKernel, 0, size);
            gridSize = (size + blockSize - 1) / blockSize;
            ApplyTransformKernel<<<gridSize, blockSize>>>(
                    size, triangleMesh.m_globalTransform,
                    static_cast<Vertex *>(verticesBuffer.m_dPtr),
                    static_cast<glm::vec3 *>(m_transformedPositionsBuffer[meshID].m_dPtr),
                    static_cast<glm::vec3 *>(m_transformedNormalsBuffer[meshID].m_dPtr),
                    static_cast<glm::vec4 *>(m_vertexColorBuffer[meshID].m_dPtr),
                    static_cast<glm::vec3 *>(m_transformedTangentBuffer[meshID].m_dPtr),
                    static_cast<glm::vec2 *>(m_texCoordBuffer[meshID].m_dPtr));
        }
        CUDA_SYNC_CHECK();

        if (triangleMesh.m_instancing) {
            auto triangles = std::vector<glm::uvec3>();
            triangles.resize(triangleMesh.m_mesh.m_triangles->size() * triangleMesh.m_matrices->size());
            unsigned offset = 0;
            for (const auto &matrix: *triangleMesh.m_matrices) {
                for (const auto &i: *triangleMesh.m_mesh.m_triangles) {
                    triangles.push_back(i);
                    triangles.back() += glm::uvec3(offset);
                }
                offset += triangleMesh.m_mesh.m_vertices->size();
            }
            m_trianglesBuffer[meshID].Upload(triangles);
        } else {
            m_trianglesBuffer[meshID].Upload(*triangleMesh.m_mesh.m_triangles);
        }
        triangleInput[meshID] = {};
        triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        deviceVertexPositions[meshID] =
                m_transformedPositionsBuffer[meshID].DevicePointer();
        deviceVertexTriangles[meshID] = m_trianglesBuffer[meshID].DevicePointer();

        triangleInput[meshID].triangleArray.vertexFormat =
                OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
        triangleInput[meshID].triangleArray.numVertices =
                static_cast<int>(m_transformedPositionsBuffer[meshID].m_sizeInBytes / sizeof(glm::vec3));
        triangleInput[meshID].triangleArray.vertexBuffers =
                &deviceVertexPositions[meshID];

        // triangleInput[meshID].triangleArray.transformFormat =
        // OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
        // triangleInput[meshID].triangleArray.preTransform =
        // deviceTransforms[meshID];

        triangleInput[meshID].triangleArray.indexFormat =
                OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::uvec3);
        triangleInput[meshID].triangleArray.numIndexTriplets =
                static_cast<int>(m_trianglesBuffer[meshID].m_sizeInBytes / sizeof(glm::uvec3));
        triangleInput[meshID].triangleArray.indexBuffer =
                deviceVertexTriangles[meshID];

        triangleInputFlags[meshID] = 0;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }
    for (; meshID < instanceSize; meshID++) {
        // upload the model to the device: the builder
        SkinnedMeshInstance &triangleMesh =
                m_skinnedInstances[meshID - m_instances.size()];

        verticesBuffer.Upload(*triangleMesh.m_skinnedMesh.m_skinnedVertices);
        m_boneMatricesBuffer[meshID].Upload(*triangleMesh.m_skinnedMesh.m_boneMatrices);

        m_transformedPositionsBuffer[meshID].Resize(
                triangleMesh.m_skinnedMesh.m_skinnedVertices->size() * sizeof(glm::vec3));
        m_transformedNormalsBuffer[meshID].Resize(
                triangleMesh.m_skinnedMesh.m_skinnedVertices->size() * sizeof(glm::vec3));
        m_transformedTangentBuffer[meshID].Resize(
                triangleMesh.m_skinnedMesh.m_skinnedVertices->size() * sizeof(glm::vec3));
        m_texCoordBuffer[meshID].Resize(triangleMesh.m_skinnedMesh.m_skinnedVertices->size() *
                                        sizeof(glm::vec2));
        m_vertexColorBuffer[meshID].Resize(triangleMesh.m_skinnedMesh.m_skinnedVertices->size() *
                                           sizeof(glm::vec4));

        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the
        // maximum occupancy for a full device launch
        int gridSize = 0;    // The actual grid size needed, based on input size
        int size = triangleMesh.m_skinnedMesh.m_skinnedVertices->size();
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                           ApplyTransformKernel, 0, size);
        gridSize = (size + blockSize - 1) / blockSize;
        ApplySkinnedTransformKernel<<<gridSize, blockSize>>>(
                size, triangleMesh.m_globalTransform,
                static_cast<SkinnedVertex *>(verticesBuffer.m_dPtr),
                static_cast<glm::mat4 *>(m_boneMatricesBuffer[meshID].m_dPtr),
                static_cast<glm::vec3 *>(m_transformedPositionsBuffer[meshID].m_dPtr),
                static_cast<glm::vec3 *>(m_transformedNormalsBuffer[meshID].m_dPtr),
                static_cast<glm::vec4 *>(m_vertexColorBuffer[meshID].m_dPtr),
                static_cast<glm::vec3 *>(m_transformedTangentBuffer[meshID].m_dPtr),
                static_cast<glm::vec2 *>(m_texCoordBuffer[meshID].m_dPtr));
        CUDA_SYNC_CHECK();

        m_trianglesBuffer[meshID].Upload(*triangleMesh.m_skinnedMesh.m_triangles);
        triangleInput[meshID] = {};
        triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        deviceVertexPositions[meshID] =
                m_transformedPositionsBuffer[meshID].DevicePointer();
        deviceVertexTriangles[meshID] = m_trianglesBuffer[meshID].DevicePointer();

        triangleInput[meshID].triangleArray.vertexFormat =
                OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
        triangleInput[meshID].triangleArray.numVertices =
                static_cast<int>(m_transformedPositionsBuffer[meshID].m_sizeInBytes / sizeof(glm::vec3));
        triangleInput[meshID].triangleArray.vertexBuffers =
                &deviceVertexPositions[meshID];

        // triangleInput[meshID].triangleArray.transformFormat =
        // OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
        // triangleInput[meshID].triangleArray.preTransform =
        // deviceTransforms[meshID];

        triangleInput[meshID].triangleArray.indexFormat =
                OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::uvec3);
        triangleInput[meshID].triangleArray.numIndexTriplets =
                static_cast<int>(m_trianglesBuffer[meshID].m_sizeInBytes / sizeof(glm::uvec3));
        triangleInput[meshID].triangleArray.indexBuffer =
                deviceVertexTriangles[meshID];

        triangleInputFlags[meshID] = 0;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }

    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelerateOptions = {};
    accelerateOptions.buildFlags =
            OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelerateOptions.motionOptions.numKeys = 1;
    accelerateOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            m_optixDeviceContext, &accelerateOptions, triangleInput.data(),
            static_cast<int>(instanceSize), // num_build_inputs
            &blasBufferSizes));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CudaBuffer compactedSizeBuffer;
    compactedSizeBuffer.Resize(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.DevicePointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CudaBuffer tempBuffer;
    tempBuffer.Resize(blasBufferSizes.tempSizeInBytes);

    CudaBuffer outputBuffer;
    outputBuffer.Resize(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(
            optixAccelBuild(m_optixDeviceContext,
                    /* stream */ nullptr, &accelerateOptions,
                            triangleInput.data(), static_cast<int>(instanceSize),
                            tempBuffer.DevicePointer(), tempBuffer.m_sizeInBytes,

                            outputBuffer.DevicePointer(), outputBuffer.m_sizeInBytes,

                            &asHandle,

                            &emitDesc, 1));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.Download(&compactedSize, 1);

    m_acceleratedStructuresBuffer.Resize(compactedSize);
    OPTIX_CHECK(optixAccelCompact(m_optixDeviceContext,
            /*stream:*/ nullptr, asHandle,
                                  m_acceleratedStructuresBuffer.DevicePointer(),
                                  m_acceleratedStructuresBuffer.m_sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.Free(); // << the Uncompacted, temporary output buffer
    tempBuffer.Free();
    compactedSizeBuffer.Free();
    verticesBuffer.Free();
    matricesBuffer.Free();
    m_cameraRenderingLaunchParams.m_traversable = asHandle;
    m_illuminationEstimationLaunchParams.m_traversable = asHandle;
    m_pointCloudScanningLaunchParams.m_traversable = asHandle;
    m_hasAccelerationStructure = true;
}

void RayTracer::AssemblePipelines() {
    AssemblePipeline(m_cameraRenderingPipeline);
    AssemblePipeline(m_illuminationEstimationPipeline);
    AssemblePipeline(m_pointCloudScanningPipeline);
}

void RayTracer::CreateRayGenProgram(RayTracerPipeline &targetPipeline,
                                    char entryFunctionName[]) const {
    targetPipeline.m_rayGenProgramGroups.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = targetPipeline.m_module;
    pgDesc.raygen.entryFunctionName = entryFunctionName;
    char log[2048];
    size_t sizeofLog = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
            m_optixDeviceContext, &pgDesc, 1, &pgOptions, log, &sizeofLog,
            &targetPipeline.m_rayGenProgramGroups[0]));
    if (sizeofLog > 1)
        std::cout << log << std::endl;
}

void RayTracer::CreateModule(RayTracerPipeline &targetPipeline, char ptxCode[],
                             char launchParamsName[]) const {
    targetPipeline.m_launchParamsName = launchParamsName;

    targetPipeline.m_moduleCompileOptions.maxRegisterCount = 50;
    targetPipeline.m_moduleCompileOptions.optLevel =
            OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    targetPipeline.m_moduleCompileOptions.debugLevel =
            OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    targetPipeline.m_pipelineCompileOptions = {};
    targetPipeline.m_pipelineCompileOptions.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    targetPipeline.m_pipelineCompileOptions.usesMotionBlur = false;
    targetPipeline.m_pipelineCompileOptions.numPayloadValues = 2;
    targetPipeline.m_pipelineCompileOptions.numAttributeValues = 2;
    targetPipeline.m_pipelineCompileOptions.exceptionFlags =
            OPTIX_EXCEPTION_FLAG_NONE;
    targetPipeline.m_pipelineCompileOptions.pipelineLaunchParamsVariableName =
            launchParamsName;

    targetPipeline.m_pipelineLinkOptions.maxTraceDepth = 31;

    const std::string code = ptxCode;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
            m_optixDeviceContext, &targetPipeline.m_moduleCompileOptions,
            &targetPipeline.m_pipelineCompileOptions, code.c_str(), code.size(), log,
            &sizeof_log, &targetPipeline.m_module));
    if (sizeof_log > 1)
        std::cout << log << std::endl;
}

void RayTracer::AssemblePipeline(RayTracerPipeline &targetPipeline) const {
    std::vector<OptixProgramGroup> programGroups;
    for (auto *pg: targetPipeline.m_rayGenProgramGroups)
        programGroups.push_back(pg);
    for (auto *pg: targetPipeline.m_missProgramGroups)
        programGroups.push_back(pg);
    for (auto *pg: targetPipeline.m_hitGroupProgramGroups)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeofLog = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
            m_optixDeviceContext, &targetPipeline.m_pipelineCompileOptions,
            &targetPipeline.m_pipelineLinkOptions, programGroups.data(),
            static_cast<int>(programGroups.size()), log, &sizeofLog,
            &targetPipeline.m_pipeline));
    if (sizeofLog > 1)
        std::cout << log << std::endl;

    OPTIX_CHECK(
            optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size
                                   for */
                    targetPipeline.m_pipeline,
                    /* [in] The direct stack size requirement for
                       direct callables invoked from IS or AH. */
                    2 * 1024,
                    /* [in] The direct stack size requirement for
                       direct
                       callables invoked from RG, MS, or CH.  */
                    2 * 1024,
                    /* [in] The continuation stack requirement. */
                    2 * 1024,
                    /* [in] The maximum depth of a traversable graph
                       passed to trace. */
                    1));
    if (sizeofLog > 1)
        std::cout << log << std::endl;
}

void RayTracer::BuildShaderBindingTable(
        std::vector<std::pair<unsigned, std::pair<cudaTextureObject_t, int>>> &boundTextures,
        std::vector<cudaGraphicsResource_t> &boundResources) {
    const int numObjects = m_instances.size() + m_skinnedInstances.size();
#pragma region Materials
    // ------------------------------------------------------------------
    // Prepare surface materials
    // ------------------------------------------------------------------
    for (auto &i: m_surfaceMaterials)
        if (i.m_type == MaterialType::Default || i.m_type == MaterialType::VertexColor)
            i.m_buffer.Free();
    m_surfaceMaterials.clear();
    m_surfaceMaterials.resize(numObjects);
    int i = 0;
    for (; i < m_instances.size(); i++) {
        auto &instance = m_instances[i];
        switch (instance.m_material.m_materialType) {
            case MaterialType::MLVQ: {
                m_surfaceMaterials[i].m_type = MaterialType::MLVQ;
                if (instance.m_material.m_MLVQMaterialIndex >= 0 &&
                    instance.m_material.m_MLVQMaterialIndex < m_MLVQMaterialStorage.size()) {
                    m_surfaceMaterials[i].m_buffer =
                            m_MLVQMaterialStorage[instance.m_material.m_MLVQMaterialIndex].m_buffer;
                } else {
                    m_surfaceMaterials[i].m_buffer = m_MLVQMaterialStorage[0].m_buffer;
                }
                break;
            }
            case MaterialType::VertexColor: {
                m_surfaceMaterials[i].m_type = MaterialType::VertexColor;
                DefaultMaterial material;
                UpdateDefaultMaterial(material, instance.m_material, boundTextures, boundResources);
                m_surfaceMaterials[i].m_buffer.Upload(&material, 1);
                break;
            }
            case MaterialType::Default:
                m_surfaceMaterials[i].m_type = MaterialType::Default;
                DefaultMaterial material;
                UpdateDefaultMaterial(material, instance.m_material, boundTextures, boundResources);
                m_surfaceMaterials[i].m_buffer.Upload(&material, 1);
                break;
        }
    }
    for (; i < numObjects; i++) {
        auto &instance = m_skinnedInstances[i - m_instances.size()];
        switch (instance.m_material.m_materialType) {
            case MaterialType::MLVQ: {
                m_surfaceMaterials[i].m_type = MaterialType::MLVQ;
                if (instance.m_material.m_MLVQMaterialIndex >= 0 &&
                    instance.m_material.m_MLVQMaterialIndex < m_MLVQMaterialStorage.size()) {
                    m_surfaceMaterials[i].m_buffer =
                            m_MLVQMaterialStorage[instance.m_material.m_MLVQMaterialIndex].m_buffer;
                } else {
                    m_surfaceMaterials[i].m_buffer = m_MLVQMaterialStorage[0].m_buffer;
                }
                break;
            }
            case MaterialType::VertexColor: {
                m_surfaceMaterials[i].m_type = MaterialType::VertexColor;
                DefaultMaterial material;
                UpdateDefaultMaterial(material, instance.m_material, boundTextures, boundResources);
                m_surfaceMaterials[i].m_buffer.Upload(&material, 1);
                break;
            }
            case MaterialType::Default:
                m_surfaceMaterials[i].m_type = MaterialType::Default;
                DefaultMaterial material;
                UpdateDefaultMaterial(material, instance.m_material, boundTextures, boundResources);
                m_surfaceMaterials[i].m_buffer.Upload(&material, 1);
                break;
        }
    }
#pragma endregion
#pragma region Prepare SBTs
    std::vector<SBT> sBTs;
    sBTs.resize(numObjects);
    i = 0;
    for (; i < m_instances.size(); i++) {
        auto &instance = m_instances[i];
        sBTs[i].m_handle = instance.m_handle;
        sBTs[i].m_mesh.m_positions = reinterpret_cast<glm::vec3 *>(
                m_transformedPositionsBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_normals = reinterpret_cast<glm::vec3 *>(
                m_transformedNormalsBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_tangents = reinterpret_cast<glm::vec3 *>(
                m_transformedTangentBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_texCoords =
                reinterpret_cast<glm::vec2 *>(m_texCoordBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_colors =
                reinterpret_cast<glm::vec4 *>(m_vertexColorBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_triangles = reinterpret_cast<glm::uvec3 *>(
                m_trianglesBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_transform = instance.m_globalTransform;

        sBTs[i].m_materialType = instance.m_material.m_materialType;
        sBTs[i].m_material = reinterpret_cast<void *>(
                m_surfaceMaterials[i].m_buffer.DevicePointer());
    }
    for (; i < numObjects; i++) {
        auto &instance = m_skinnedInstances[i - m_instances.size()];
        sBTs[i].m_handle = instance.m_handle;
        sBTs[i].m_mesh.m_positions = reinterpret_cast<glm::vec3 *>(
                m_transformedPositionsBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_normals = reinterpret_cast<glm::vec3 *>(
                m_transformedNormalsBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_tangents = reinterpret_cast<glm::vec3 *>(
                m_transformedTangentBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_texCoords =
                reinterpret_cast<glm::vec2 *>(m_texCoordBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_colors =
                reinterpret_cast<glm::vec4 *>(m_vertexColorBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_triangles = reinterpret_cast<glm::uvec3 *>(
                m_trianglesBuffer[i].DevicePointer());
        sBTs[i].m_mesh.m_transform = instance.m_globalTransform;

        sBTs[i].m_materialType = instance.m_material.m_materialType;
        sBTs[i].m_material = reinterpret_cast<void *>(
                m_surfaceMaterials[i].m_buffer.DevicePointer());
    }
#pragma endregion
    {
        // ------------------------------------------------------------------
        // build raygen records
        // ------------------------------------------------------------------
        std::vector<CameraRenderingRayGenRecord> raygenRecords;
        for (i = 0; i < m_cameraRenderingPipeline.m_rayGenProgramGroups.size();
             i++) {
            CameraRenderingRayGenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(
                    m_cameraRenderingPipeline.m_rayGenProgramGroups[i], &rec));
            rec.m_data = nullptr; /* for now ... */
            raygenRecords.push_back(rec);
        }
        m_cameraRenderingPipeline.m_rayGenRecordsBuffer.Upload(raygenRecords);
        m_cameraRenderingPipeline.m_sbt.raygenRecord =
                m_cameraRenderingPipeline.m_rayGenRecordsBuffer.DevicePointer();

        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        std::vector<CameraRenderingRayMissRecord> missRecords;
        for (i = 0; i < m_cameraRenderingPipeline.m_missProgramGroups.size();
             i++) {
            CameraRenderingRayMissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(
                    m_cameraRenderingPipeline.m_missProgramGroups[i], &rec));
            rec.m_data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        m_cameraRenderingPipeline.m_missRecordsBuffer.Upload(missRecords);
        m_cameraRenderingPipeline.m_sbt.missRecordBase =
                m_cameraRenderingPipeline.m_missRecordsBuffer.DevicePointer();
        m_cameraRenderingPipeline.m_sbt.missRecordStrideInBytes =
                sizeof(CameraRenderingRayMissRecord);
        m_cameraRenderingPipeline.m_sbt.missRecordCount =
                static_cast<int>(missRecords.size());

        // ------------------------------------------------------------------
        // build hit records
        // ------------------------------------------------------------------

        // we don't actually have any objects in this example, but let's
        // create a dummy one so the SBT doesn't have any null pointers
        // (which the sanity checks in compilation would complain about)

        std::vector<CameraRenderingRayHitRecord> hitGroupRecords;
        for (i = 0; i < m_instances.size(); i++) {
            auto &instance = m_instances[i];
            for (int rayID = 0;
                 rayID < static_cast<int>(RayType::RayTypeCount);
                 rayID++) {
                CameraRenderingRayHitRecord rec;
                OPTIX_CHECK(optixSbtRecordPackHeader(
                        m_cameraRenderingPipeline.m_hitGroupProgramGroups[rayID], &rec));
                rec.m_data = sBTs[i];
                hitGroupRecords.push_back(rec);
            }
        }
        for (; i < numObjects; i++) {
            auto &instance = m_skinnedInstances[i - m_instances.size()];
            for (int rayID = 0;
                 rayID < static_cast<int>(RayType::RayTypeCount);
                 rayID++) {
                CameraRenderingRayHitRecord rec;
                OPTIX_CHECK(optixSbtRecordPackHeader(
                        m_cameraRenderingPipeline.m_hitGroupProgramGroups[rayID], &rec));
                rec.m_data = sBTs[i];
                hitGroupRecords.push_back(rec);
            }
        }
        m_cameraRenderingPipeline.m_hitGroupRecordsBuffer.Upload(hitGroupRecords);
        m_cameraRenderingPipeline.m_sbt.hitgroupRecordBase =
                m_cameraRenderingPipeline.m_hitGroupRecordsBuffer.DevicePointer();
        m_cameraRenderingPipeline.m_sbt.hitgroupRecordStrideInBytes =
                sizeof(CameraRenderingRayHitRecord);
        m_cameraRenderingPipeline.m_sbt.hitgroupRecordCount =
                static_cast<int>(hitGroupRecords.size());
    }
    {
        // ------------------------------------------------------------------
        // build raygen records
        // ------------------------------------------------------------------
        std::vector<IlluminationEstimationRayGenRecord> raygenRecords;
        for (i = 0;
             i < m_illuminationEstimationPipeline.m_rayGenProgramGroups.size();
             i++) {
            IlluminationEstimationRayGenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(
                    m_illuminationEstimationPipeline.m_rayGenProgramGroups[i],
                    &rec));
            rec.m_data = nullptr; /* for now ... */
            raygenRecords.push_back(rec);
        }
        m_illuminationEstimationPipeline.m_rayGenRecordsBuffer.Upload(
                raygenRecords);
        m_illuminationEstimationPipeline.m_sbt.raygenRecord =
                m_illuminationEstimationPipeline.m_rayGenRecordsBuffer
                        .DevicePointer();

        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        std::vector<IlluminationEstimationRayMissRecord> missRecords;
        for (i = 0;
             i < m_illuminationEstimationPipeline.m_missProgramGroups.size();
             i++) {
            IlluminationEstimationRayMissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(
                    m_illuminationEstimationPipeline.m_missProgramGroups[i],
                    &rec));
            rec.m_data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        m_illuminationEstimationPipeline.m_missRecordsBuffer.Upload(
                missRecords);
        m_illuminationEstimationPipeline.m_sbt.missRecordBase =
                m_illuminationEstimationPipeline.m_missRecordsBuffer
                        .DevicePointer();
        m_illuminationEstimationPipeline.m_sbt.missRecordStrideInBytes =
                sizeof(IlluminationEstimationRayMissRecord);
        m_illuminationEstimationPipeline.m_sbt.missRecordCount =
                static_cast<int>(missRecords.size());

        // ------------------------------------------------------------------
        // build hit records
        // ------------------------------------------------------------------

        // we don't actually have any objects in this example, but let's
        // create a dummy one so the SBT doesn't have any null pointers
        // (which the sanity checks in compilation would complain about)
        std::vector<IlluminationEstimationRayHitRecord> hitGroupRecords;
        for (i = 0; i < m_instances.size(); i++) {
            auto &instance = m_instances[i];
            for (int rayID = 0;
                 rayID <
                 static_cast<int>(RayType::RayTypeCount);
                 rayID++) {
                IlluminationEstimationRayHitRecord rec;
                OPTIX_CHECK(
                        optixSbtRecordPackHeader(
                                m_illuminationEstimationPipeline
                                        .m_hitGroupProgramGroups[rayID],
                                &rec));
                rec.m_data = sBTs[i];
                hitGroupRecords.push_back(rec);
            }
        }
        for (; i < numObjects; i++) {
            auto &instance = m_skinnedInstances[i - m_instances.size()];
            for (int rayID = 0;
                 rayID <
                 static_cast<int>(RayType::RayTypeCount);
                 rayID++) {
                IlluminationEstimationRayHitRecord rec;
                OPTIX_CHECK(
                        optixSbtRecordPackHeader(
                                m_illuminationEstimationPipeline
                                        .m_hitGroupProgramGroups[rayID],
                                &rec));
                rec.m_data = sBTs[i];
                hitGroupRecords.push_back(rec);
            }
        }
        m_illuminationEstimationPipeline.m_hitGroupRecordsBuffer.Upload(
                hitGroupRecords);
        m_illuminationEstimationPipeline.m_sbt.hitgroupRecordBase =
                m_illuminationEstimationPipeline.m_hitGroupRecordsBuffer
                        .DevicePointer();
        m_illuminationEstimationPipeline.m_sbt.hitgroupRecordStrideInBytes =
                sizeof(IlluminationEstimationRayHitRecord);
        m_illuminationEstimationPipeline.m_sbt.hitgroupRecordCount =
                static_cast<int>(hitGroupRecords.size());
    }

    {
        // ------------------------------------------------------------------
        // build raygen records
        // ------------------------------------------------------------------
        std::vector<PointCloudScanningRayGenRecord> raygenRecords;
        for (int i = 0;
             i < m_pointCloudScanningPipeline.m_rayGenProgramGroups.size();
             i++) {
            PointCloudScanningRayGenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(
                    m_pointCloudScanningPipeline.m_rayGenProgramGroups[i],
                    &rec));
            rec.m_data = nullptr; /* for now ... */
            raygenRecords.push_back(rec);
        }
        m_pointCloudScanningPipeline.m_rayGenRecordsBuffer.Upload(
                raygenRecords);
        m_pointCloudScanningPipeline.m_sbt.raygenRecord =
                m_pointCloudScanningPipeline.m_rayGenRecordsBuffer
                        .DevicePointer();

        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        std::vector<PointCloudScanningRayMissRecord> missRecords;
        for (int i = 0;
             i < m_pointCloudScanningPipeline.m_missProgramGroups.size();
             i++) {
            PointCloudScanningRayMissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(
                    m_pointCloudScanningPipeline.m_missProgramGroups[i],
                    &rec));
            rec.m_data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        m_pointCloudScanningPipeline.m_missRecordsBuffer.Upload(
                missRecords);
        m_pointCloudScanningPipeline.m_sbt.missRecordBase =
                m_pointCloudScanningPipeline.m_missRecordsBuffer
                        .DevicePointer();
        m_pointCloudScanningPipeline.m_sbt.missRecordStrideInBytes =
                sizeof(PointCloudScanningRayMissRecord);
        m_pointCloudScanningPipeline.m_sbt.missRecordCount =
                static_cast<int>(missRecords.size());

        // ------------------------------------------------------------------
        // build hit records
        // ------------------------------------------------------------------

        // we don't actually have any objects in this example, but let's
        // create a dummy one so the SBT doesn't have any null pointers
        // (which the sanity checks in compilation would complain about)
        std::vector<PointCloudScanningRayHitRecord> hitGroupRecords;
        for (i = 0; i < m_instances.size(); i++) {
            auto &instance = m_instances[i];
            for (int rayID = 0;
                 rayID <
                 static_cast<int>(RayType::RayTypeCount) - 1;
                 rayID++) {
                PointCloudScanningRayHitRecord rec;
                OPTIX_CHECK(
                        optixSbtRecordPackHeader(
                                m_pointCloudScanningPipeline.m_hitGroupProgramGroups[rayID],
                                &rec));
                rec.m_data = sBTs[i];
                hitGroupRecords.push_back(rec);
            }
        }
        for (; i < numObjects; i++) {
            auto &instance = m_skinnedInstances[i - m_instances.size()];
            for (int rayID = 0;
                 rayID <
                 static_cast<int>(RayType::RayTypeCount) - 1;
                 rayID++) {
                PointCloudScanningRayHitRecord rec;
                OPTIX_CHECK(
                        optixSbtRecordPackHeader(
                                m_pointCloudScanningPipeline.m_hitGroupProgramGroups[rayID],
                                &rec));
                rec.m_data = sBTs[i];
                hitGroupRecords.push_back(rec);
            }
        }
        m_pointCloudScanningPipeline.m_hitGroupRecordsBuffer.Upload(
                hitGroupRecords);
        m_pointCloudScanningPipeline.m_sbt.hitgroupRecordBase =
                m_pointCloudScanningPipeline.m_hitGroupRecordsBuffer
                        .DevicePointer();
        m_pointCloudScanningPipeline.m_sbt.hitgroupRecordStrideInBytes =
                sizeof(PointCloudScanningRayHitRecord);
        m_pointCloudScanningPipeline.m_sbt.hitgroupRecordCount =
                static_cast<int>(hitGroupRecords.size());
    }
}


void RayTracer::LoadBtfMaterials(const std::vector<std::string> &folderPathes) {
    for (const auto &entry: folderPathes) {
        MLVQMaterialStorage storage;
        storage.m_material = std::make_shared<MLVQMaterial>();
        storage.m_material->m_btf.Init(entry);
        storage.m_buffer.Upload(storage.m_material.get(), 1);
        m_MLVQMaterialStorage.push_back(storage);
    }
}

void RayTracer::UpdateDefaultMaterial(DefaultMaterial &material, RayTracerMaterial &rayTracerMaterial,
                                      std::vector<std::pair<unsigned, std::pair<cudaTextureObject_t, int>>> &boundTextures,
                                      std::vector<cudaGraphicsResource_t> &boundResources) {
#pragma region Material Settings
    material.m_materialProperties = rayTracerMaterial.m_materialProperties;
    material.m_albedoTexture.m_texture = 0;
    material.m_normalTexture.m_texture = 0;
    material.m_roughnessTexture.m_texture = 0;
    material.m_metallicTexture.m_texture = 0;
    if (rayTracerMaterial.m_albedoTexture.m_textureId != 0) {
        bool duplicate = false;
        for (auto &boundTexture: boundTextures) {
            if (boundTexture.first == rayTracerMaterial.m_albedoTexture.m_textureId) {
                material.m_albedoTexture.m_texture = boundTexture.second.first;
                material.m_albedoTexture.m_channel = boundTexture.second.second;
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            cudaGraphicsResource_t graphicsResource;
            BindTexture(rayTracerMaterial.m_albedoTexture.m_textureId, graphicsResource, material.m_albedoTexture.m_texture);
            boundResources.push_back(graphicsResource);
            boundTextures.emplace_back(rayTracerMaterial.m_albedoTexture.m_textureId,
                                       std::make_pair(material.m_albedoTexture.m_texture,
                                                      material.m_albedoTexture.m_channel));
        }
    }
    if (rayTracerMaterial.m_normalTexture.m_textureId != 0) {
        bool duplicate = false;
        for (auto &boundTexture: boundTextures) {
            if (boundTexture.first == rayTracerMaterial.m_normalTexture.m_textureId) {
                material.m_normalTexture.m_texture = boundTexture.second.first;
                material.m_normalTexture.m_channel = boundTexture.second.second;
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            cudaGraphicsResource_t graphicsResource;
            BindTexture(rayTracerMaterial.m_normalTexture.m_textureId, graphicsResource, material.m_normalTexture.m_texture);
            boundResources.push_back(graphicsResource);
            boundTextures.emplace_back(rayTracerMaterial.m_normalTexture.m_textureId,
                                       std::make_pair(material.m_normalTexture.m_texture,
                                                      material.m_normalTexture.m_channel));
        }
    }
    if (rayTracerMaterial.m_roughnessTexture.m_textureId != 0) {
        bool duplicate = false;
        for (auto &boundTexture: boundTextures) {
            if (boundTexture.first == rayTracerMaterial.m_roughnessTexture.m_textureId) {
                material.m_roughnessTexture.m_texture = boundTexture.second.first;
                material.m_roughnessTexture.m_channel = boundTexture.second.second;
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            cudaGraphicsResource_t graphicsResource;
            BindTexture(rayTracerMaterial.m_roughnessTexture.m_textureId, graphicsResource, material.m_roughnessTexture.m_texture);
            boundResources.push_back(graphicsResource);
            boundTextures.emplace_back(rayTracerMaterial.m_roughnessTexture.m_textureId,
                                       std::make_pair(material.m_roughnessTexture.m_texture,
                                                      material.m_roughnessTexture.m_channel));
        }
    }
    if (rayTracerMaterial.m_metallicTexture.m_textureId != 0) {
        bool duplicate = false;
        for (auto &boundTexture: boundTextures) {
            if (boundTexture.first == rayTracerMaterial.m_metallicTexture.m_textureId) {
                material.m_metallicTexture.m_texture = boundTexture.second.first;
                material.m_metallicTexture.m_channel = boundTexture.second.second;
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            cudaGraphicsResource_t graphicsResource;
            BindTexture(rayTracerMaterial.m_metallicTexture.m_textureId, graphicsResource, material.m_metallicTexture.m_texture);
            boundResources.push_back(graphicsResource);
            boundTextures.emplace_back(rayTracerMaterial.m_metallicTexture.m_textureId,
                                       std::make_pair(material.m_metallicTexture.m_texture,
                                                      material.m_metallicTexture.m_channel));
        }
    }
#pragma endregion


}

void RayTracer::BindTexture(unsigned int id, cudaGraphicsResource_t &graphicsResource, cudaTextureObject_t& textureObject) {
    cudaArray_t textureArray;
    CUDA_CHECK(GraphicsGLRegisterImage(
            &graphicsResource, id, GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsReadOnly));
    CUDA_CHECK(GraphicsMapResources(1, &graphicsResource, nullptr));
    CUDA_CHECK(GraphicsSubResourceGetMappedArray(&textureArray,
                                                 graphicsResource, 0, 0));
    struct cudaResourceDesc cudaResourceDesc;
    memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
    cudaResourceDesc.resType = cudaResourceTypeArray;
    cudaResourceDesc.res.array.array = textureArray;
    struct cudaTextureDesc cudaTextureDesc;
    memset(&cudaTextureDesc, 0, sizeof(cudaTextureDesc));
    cudaTextureDesc.addressMode[0] = cudaAddressModeWrap;
    cudaTextureDesc.addressMode[1] = cudaAddressModeWrap;
    cudaTextureDesc.filterMode = cudaFilterModeLinear;
    cudaTextureDesc.readMode = cudaReadModeElementType;
    cudaTextureDesc.normalizedCoords = 1;
    CUDA_CHECK(CreateTextureObject(&textureObject,
                                   &cudaResourceDesc, &cudaTextureDesc,
                                   nullptr));
}


