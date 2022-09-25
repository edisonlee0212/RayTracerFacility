#pragma once

#include <BTFBase.cuh>

#include <CUDABuffer.hpp>

#include <Optix7.hpp>

#include <Vertex.hpp>

#include <cuda.h>

#include "string"

#include "vector"

#include "map"

#include "MaterialProperties.hpp"

#include "filesystem"

namespace RayTracerFacility {
    enum class OutputType {
        Color, Normal, Albedo, Depth
    };

    struct RAY_TRACER_FACILITY_API CameraProperties {
#pragma region FrameBuffer
        /*! the color buffer we use during _rendering_, which is a bit
        larger than the actual displayed frame buffer (to account for
        the border), and in float4 format (the denoiser requires
        floats) */
        CudaBuffer m_frameBufferColor;
        CudaBuffer m_frameBufferNormal;
        CudaBuffer m_frameBufferAlbedo;
#pragma endregion
#pragma region Denoiser
        /*! output of the denoiser pass, in float4 */
        CudaBuffer m_denoisedBuffer;
        OptixDenoiser m_denoiser = nullptr;
        CudaBuffer m_denoiserScratch;
        CudaBuffer m_denoiserState;
        CudaBuffer m_denoiserIntensity;
#pragma endregion
        bool m_accumulate = true;

        float m_fov = 120;
        /*! camera position - *from* where we are looking */
        glm::vec3 m_from = glm::vec3(0.0f);
        /*! which direction we are looking *at* */
        glm::vec3 m_direction = glm::vec3(0.0f);
        /*! which direction we are looking *at* */
        glm::vec3 m_up = glm::vec3(0.0f);
        /*! general up-vector */
        glm::vec3 m_horizontal;
        glm::vec3 m_vertical;

        float m_denoiserStrength = 1.0f;
        float m_maxDistance = 50.0f;
        unsigned m_outputTextureId = 0;
        OutputType m_outputType = OutputType::Color;
        float m_gamma = 2.2f;
        struct {
            glm::vec4 *m_colorBuffer;
            glm::vec4 *m_normalBuffer;
            glm::vec4 *m_albedoBuffer;
            /*! the size of the frame buffer to render */
            glm::ivec2 m_size;
            size_t m_frameId;
        } m_frame;

        bool m_modified = false;

        float m_aperture = 0.0f;
        float m_focalLength = 1.0f;

        void SetAperture(float value);

        void SetFocalLength(float value);

        void SetFov(float value);

        void SetGamma(float value);

        void SetMaxDistance(float value);

        void SetOutputType(OutputType value);

        void SetDenoiserStrength(float value);

        void Resize(const glm::ivec2 &newSize);

        void Set(const glm::vec3 &position, const glm::quat &rotation);

        void OnInspect();
    };

#pragma region MyRegion
    enum class EnvironmentalLightingType {
        Scene, Skydome, SingleLightSource
    };

    struct RAY_TRACER_FACILITY_API EnvironmentProperties {
        EnvironmentalLightingType m_environmentalLightingType =
                EnvironmentalLightingType::Scene;
        float m_skylightIntensity = 1.0f;
        float m_ambientLightIntensity = 0.1f;
        float m_lightSize = 0.0f;
        float m_gamma = 1.0f;
        glm::vec3 m_sunDirection = glm::vec3(0, 1, 0);
        glm::vec3 m_color = glm::vec3(1, 1, 1);
        unsigned m_environmentalMapId = 0;
        cudaTextureObject_t m_environmentalMaps[6];

        struct {
            float m_earthRadius =
                    6360; // In the paper this is usually Rg or Re (radius ground, eart)
            float m_atmosphereRadius =
                    6420; // In the paper this is usually R or Ra (radius atmosphere)
            float m_Hr =
                    7994; // Thickness of the atmosphere if density was uniform (Hr)
            float m_Hm = 1200; // Same as above but for Mie scattering (Hm)
            float m_g = 0.76f; // Mean cosine for Mie scattering
            int m_numSamples = 16;
            int m_numSamplesLight = 8;
        } m_atmosphere;

        [[nodiscard]] bool Changed(const EnvironmentProperties &properties) const {
            return properties.m_environmentalLightingType !=
                   m_environmentalLightingType ||
                   properties.m_lightSize != m_lightSize ||
                   properties.m_ambientLightIntensity != m_ambientLightIntensity ||
                   properties.m_skylightIntensity != m_skylightIntensity ||
                   properties.m_gamma != m_gamma ||
                   properties.m_sunDirection != m_sunDirection ||
                   properties.m_environmentalMapId != m_environmentalMapId ||
                   properties.m_color != m_color ||
                   properties.m_atmosphere.m_earthRadius !=
                   m_atmosphere.m_earthRadius ||
                   properties.m_atmosphere.m_atmosphereRadius !=
                   m_atmosphere.m_atmosphereRadius ||
                   properties.m_atmosphere.m_Hr != m_atmosphere.m_Hr ||
                   properties.m_atmosphere.m_Hm != m_atmosphere.m_Hm ||
                   properties.m_atmosphere.m_g != m_atmosphere.m_g ||
                   properties.m_atmosphere.m_numSamples != m_atmosphere.m_numSamples ||
                   properties.m_atmosphere.m_numSamplesLight !=
                   m_atmosphere.m_numSamplesLight;
        }

        void OnInspect();
    };

    struct RAY_TRACER_FACILITY_API RayProperties {
        int m_bounces = 4;
        int m_samples = 1;

        [[nodiscard]] bool Changed(const RayProperties &properties) const {
            return properties.m_bounces != m_bounces ||
                   properties.m_samples != m_samples;
        }

        void OnInspect();
    };

    struct RAY_TRACER_FACILITY_API RayTracerProperties {
        EnvironmentProperties m_environment;
        RayProperties m_rayProperties;

        [[nodiscard]] bool Changed(const RayTracerProperties &properties) const {
            return m_environment.Changed(properties.m_environment) ||
                   m_rayProperties.Changed(properties.m_rayProperties);
        }

        void OnInspect();
    };

    enum class RayType {
        Radiance, SpacialSampling, RayTypeCount
    };

    struct CameraRenderingLaunchParams {
        CameraProperties m_cameraProperties;
        RayTracerProperties m_rayTracerProperties;
        OptixTraversableHandle m_traversable;
    };

    template<typename T>
    struct RAY_TRACER_FACILITY_API IlluminationSampler {
        glm::vec3 m_surfaceNormal;
        /**
         * \brief The position of the light probe.
         */
        glm::vec3 m_position;
        /**
         * \brief The calculated overall direction where the point received most
         * light.
         */
        glm::vec3 m_direction;
        /**
         * \brief The total energy received at this point.
         */
        T m_energy = 0;
        /*
         * If the lightprobe covers entire sphere or just a hemisphere.
         */
        bool m_doubleFace = false;
    };

    struct IlluminationEstimationLaunchParams {
        unsigned m_seed = 0;
        float m_pushNormalDistance = 0.001f;
        size_t m_size;
        RayTracerProperties m_rayTracerProperties;
        IlluminationSampler<float> *m_lightProbes;
        OptixTraversableHandle m_traversable;
    };

    struct RAY_TRACER_FACILITY_API PointCloudSample {
        // Input
        glm::vec3 m_direction;
        glm::vec3 m_start;

        // Output
        uint64_t m_handle;
        bool m_hit = false;
        glm::vec3 m_end;
        glm::vec3 m_albedo;
    };

    struct PointCloudScanningLaunchParams {
        size_t m_size;
        RayTracerProperties m_rayTracerProperties;
        PointCloudSample *m_samples;
        OptixTraversableHandle m_traversable;
    };

#pragma endregion

    struct RAY_TRACER_FACILITY_API RayTracerTexture {
        unsigned m_textureId = 0;
        int m_channel = 0;
    };

    struct RAY_TRACER_FACILITY_API RayTracerMaterial {
        MaterialType m_materialType = MaterialType::Default;

        int m_MLVQMaterialIndex;
        MaterialProperties m_materialProperties;

        RayTracerTexture m_albedoTexture;
        RayTracerTexture m_normalTexture;
        RayTracerTexture m_metallicTexture;
        RayTracerTexture m_roughnessTexture;

        size_t m_version;
        uint64_t m_handle = 0;
    };

    enum class RayTracerMeshType {
        Default,
        Instanced,
        Skinned
    };

    struct RAY_TRACER_FACILITY_API RayTracedGeometry {
        RayTracerMeshType m_meshType = RayTracerMeshType::Default;
        union {
            std::vector<UniEngine::Vertex> *m_vertices;
            std::vector<UniEngine::SkinnedVertex> *m_skinnedVertices;
        };
        std::vector<glm::mat4> *m_boneMatrices = nullptr;
        std::vector<glm::mat4> *m_instanceMatrices = nullptr;
        std::vector<glm::uvec3> *m_triangles;

        OptixTraversableHandle m_traversableHandle = 0;
        CudaBuffer m_positionBuffer;
        CudaBuffer m_normalBuffer;
        CudaBuffer m_tangentBuffer;
        CudaBuffer m_colorBuffer;
        CudaBuffer m_texCoordBuffer;
        CudaBuffer m_triangleBuffer;
        CudaBuffer m_acceleratedStructureBuffer;
        size_t m_version = 0;
        uint64_t m_handle = 0;
        bool m_updateFlag = false;
        bool m_removeFlag = true;

        void BuildGAS(const OptixDeviceContext &context);
    };

    struct SurfaceMaterial {
        MaterialType m_type;
        CudaBuffer m_buffer;
    };
    struct DefaultMaterial;

    struct RAY_TRACER_FACILITY_API RayTracedInstance {
        uint64_t m_entityHandle = 0;
        size_t m_version = 0;
        uint64_t m_privateComponentHandle = 0;

        uint64_t m_geometryMapKey = 0;
        RayTracerMaterial m_material;
        glm::mat4 m_globalTransform;
        bool m_removeFlag = true;
    };

    struct RayTracerPipeline {
        std::string m_launchParamsName;
        OptixModule m_module;
        OptixModuleCompileOptions m_moduleCompileOptions = {};

        OptixPipeline m_pipeline;
        OptixPipelineCompileOptions m_pipelineCompileOptions = {};
        OptixPipelineLinkOptions m_pipelineLinkOptions = {};

        std::vector<OptixProgramGroup> m_rayGenProgramGroups;
        CudaBuffer m_rayGenRecordsBuffer;
        std::vector<OptixProgramGroup> m_missProgramGroups;
        CudaBuffer m_missRecordsBuffer;
        std::vector<OptixProgramGroup> m_hitGroupProgramGroups;
        CudaBuffer m_hitGroupRecordsBuffer;
        OptixShaderBindingTable m_sbt = {};
        CudaBuffer m_launchParamsBuffer;
    };
    struct MLVQMaterial;
    struct MLVQMaterialStorage {
        std::shared_ptr<MLVQMaterial> m_material;
        CudaBuffer m_buffer;
    };


    class RayTracer {
    public:
        bool m_requireUpdate = false;
        std::map<uint64_t, SurfaceMaterial> m_materials;
        std::map<uint64_t, RayTracedGeometry> m_geometries;
        std::map<uint64_t, RayTracedInstance> m_instances;

        // ------------------------------------------------------------------
        // internal helper functions
        // ------------------------------------------------------------------
        [[nodiscard]] bool
        RenderToCamera(const EnvironmentProperties &environmentProperties,
                       CameraProperties &cameraProperties,
                       const RayProperties &rayProperties);

        void EstimateIllumination(const size_t &size,
                                  const EnvironmentProperties &environmentProperties,
                                  const RayProperties &rayProperties,
                                  CudaBuffer &lightProbes, unsigned seed,
                                  float pushNormalDistance);

        void ScanPointCloud(const size_t &size,
                            const EnvironmentProperties &environmentProperties,
                            CudaBuffer &samples);

        RayTracer();

        /*! build an acceleration structure for the given triangle mesh */
        void BuildIAS();

        /*! constructs the shader binding table */
        void BuildShaderBindingTable(
                std::vector<std::pair<unsigned, std::pair<cudaTextureObject_t, int>>>
                &boundTextures,
                std::vector<cudaGraphicsResource_t> &boundResources);

        void LoadBtfMaterials(const std::vector<std::string> &folderPathes);

    protected:
        void BindTexture(unsigned int id, cudaGraphicsResource_t &graphicsResource,
                         cudaTextureObject_t &textureObject);

        void UpdateDefaultMaterial(
                DefaultMaterial &material, RayTracerMaterial &rayTracerMaterial,
                std::vector<std::pair<unsigned, std::pair<cudaTextureObject_t, int>>>
                &boundTextures,
                std::vector<cudaGraphicsResource_t> &boundResources);

#pragma region MLVQ
        std::vector<MLVQMaterialStorage> m_MLVQMaterialStorage;
#pragma endregion

#pragma region Device and context
        /*! @{ CUDA device context and stream that optix pipeline will run
                on, as well as device properties for this device */
        CUcontext m_cudaContext;
        CUstream m_stream;
        cudaDeviceProp m_deviceProps;
        /*! @} */
        //! the optix context that our pipeline will run in.
        OptixDeviceContext m_optixDeviceContext;

        friend class CameraProperties;

        /*! creates and configures a optix device context (in this simple
          example, only for the primary GPU device) */
        void CreateContext();

#pragma endregion
#pragma region Pipeline setup

        CameraRenderingLaunchParams m_cameraRenderingLaunchParams;
        IlluminationEstimationLaunchParams m_illuminationEstimationLaunchParams;
        PointCloudScanningLaunchParams m_pointCloudScanningLaunchParams;

        RayTracerPipeline m_cameraRenderingPipeline;
        RayTracerPipeline m_illuminationEstimationPipeline;
        RayTracerPipeline m_pointCloudScanningPipeline;

        /*! creates the module that contains all the programs we are going
          to use. in this simple example, we use a single module from a
          single .cu file, using a single embedded ptx string */
        void CreateModules();

        /*! does all setup for the rayGen program(s) we are going to use */
        void CreateRayGenPrograms();

        /*! does all setup for the miss program(s) we are going to use */
        void CreateMissPrograms();

        /*! does all setup for the hitGroup program(s) we are going to use */
        void CreateHitGroupPrograms();

        /*! assembles the full pipeline of all programs */
        void AssemblePipelines();

        void CreateRayGenProgram(RayTracerPipeline &targetPipeline,
                                 char entryFunctionName[]) const;

        void CreateModule(RayTracerPipeline &targetPipeline, char ptxCode[],
                          char launchParamsName[]) const;

        void AssemblePipeline(RayTracerPipeline &targetPipeline) const;

#pragma endregion

#pragma region Accleration structure
        /*! check if we have build the acceleration structure. */
        bool m_hasAccelerationStructure = false;
        //! buffer that keeps the (final, compacted) acceleration structure
        CudaBuffer m_iASBuffer;
#pragma endregion

        friend class RayTracerCamera;
    };

} // namespace RayTracerFacility
