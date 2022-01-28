#pragma once

#include <BTFBase.cuh>

#include <CUDABuffer.hpp>

#include <Optix7.hpp>

#include <Vertex.hpp>

#include <cuda.h>

#include <glm/glm.hpp>

#include <string>

#include <vector>

#include <MaterialType.hpp>

#include <filesystem>


namespace RayTracerFacility {
    enum class OutputType {
        Color,
        Normal,
        Albedo
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

        void SetFov(float value);

        void SetGamma(float value);

        void SetOutputType(OutputType value);

        void SetDenoiserStrength(float value);

        void Resize(const glm::ivec2 &newSize);

        void Set(const glm::vec3 &position, const glm::quat &rotation);

        void OnInspect();
    };

#pragma region MyRegion
    enum class EnvironmentalLightingType {
        Skydome, EnvironmentalMap, Color
    };


    struct RAY_TRACER_FACILITY_API EnvironmentProperties {
        EnvironmentalLightingType m_environmentalLightingType =
                EnvironmentalLightingType::Skydome;
        float m_skylightIntensity = 1.0f;
        glm::vec3 m_sunDirection = glm::vec3(0, 1, 0);
        glm::vec3 m_sunColor = glm::vec3(1, 1, 1);
        unsigned m_environmentalMapId = 0;
        cudaTextureObject_t m_environmentalMaps[6];

        struct {
            float m_earthRadius = 6360;      // In the paper this is usually Rg or Re (radius ground, eart)
            float m_atmosphereRadius = 6420; // In the paper this is usually R or Ra (radius atmosphere)
            float m_Hr = 7994;               // Thickness of the atmosphere if density was uniform (Hr)
            float m_Hm = 1200;               // Same as above but for Mie scattering (Hm)
            float m_g = 0.76f;               // Mean cosine for Mie scattering
            int m_numSamples = 16;
            int m_numSamplesLight = 8;
        } m_atmosphere;

        [[nodiscard]] bool
        Changed(const EnvironmentProperties &properties) const {
            return properties.m_environmentalLightingType !=
                   m_environmentalLightingType ||
                   properties.m_skylightIntensity != m_skylightIntensity ||
                   properties.m_sunDirection != m_sunDirection ||
                   properties.m_environmentalMapId != m_environmentalMapId ||
                   properties.m_sunColor != m_sunColor ||
                   properties.m_atmosphere.m_earthRadius != m_atmosphere.m_earthRadius ||
                   properties.m_atmosphere.m_atmosphereRadius != m_atmosphere.m_atmosphereRadius ||
                   properties.m_atmosphere.m_Hr != m_atmosphere.m_Hr ||
                   properties.m_atmosphere.m_Hm != m_atmosphere.m_Hm ||
                   properties.m_atmosphere.m_g != m_atmosphere.m_g ||
                   properties.m_atmosphere.m_numSamples != m_atmosphere.m_numSamples ||
                   properties.m_atmosphere.m_numSamplesLight != m_atmosphere.m_numSamplesLight;
        }

        void OnInspect();
    };

    struct RAY_TRACER_FACILITY_API RayProperties {
        int m_bounces = 4;
        int m_samples = 1;

        [[nodiscard]] bool
        Changed(const RayProperties &properties) const {
            return properties.m_bounces != m_bounces ||
                   properties.m_samples != m_samples;
        }

        void OnInspect();
    };

    struct RAY_TRACER_FACILITY_API RayTracerProperties {
        EnvironmentProperties m_environment;
        RayProperties m_rayProperties;

        [[nodiscard]] bool
        Changed(const RayTracerProperties &properties) const {
            return m_environment.Changed(properties.m_environment) ||
                   m_rayProperties.Changed(properties.m_rayProperties);
        }

        void OnInspect();
    };

    enum class RayType {
        Radiance,
        SpacialSampling,
        RayTypeCount
    };

    struct VertexInfo;

    struct CameraRenderingLaunchParams {
        CameraProperties m_cameraProperties;
        RayTracerProperties m_rayTracerProperties;
        OptixTraversableHandle m_traversable;
    };

    template<typename T>
    struct RAY_TRACER_FACILITY_API LightProbe {
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
        LightProbe<float> *m_lightProbes;
        OptixTraversableHandle m_traversable;
    };

    struct RAY_TRACER_FACILITY_API PointCloudSample {
        //Input
        glm::vec3 m_direction;
        glm::vec3 m_start;

        //Output
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

    struct RAY_TRACER_FACILITY_API RayTracerMaterial{

        MaterialType m_materialType = MaterialType::Default;
        int m_MLVQMaterialIndex;
        glm::vec3 m_surfaceColor;
        glm::vec3 m_subsurfaceColor;
        float m_subsurfaceRadius;
        float m_roughness;
        float m_metallic;
        float m_emission = 0;

        unsigned m_albedoTexture = 0;
        unsigned m_normalTexture = 0;
        unsigned m_metallicTexture = 0;
        unsigned m_roughnessTexture = 0;


    };

    struct RAY_TRACER_FACILITY_API RayTracerInstance {
        bool m_instancing = false;
        std::vector<Vertex> *m_vertices;
        std::vector<glm::uvec3> *m_triangles;
        std::vector<glm::mat4> *m_matrices;
        size_t m_version;
        size_t m_matricesVersion;
        uint64_t m_handle = 0;

        RayTracerMaterial m_material;

        bool m_removeTag = false;
        glm::mat4 m_globalTransform;
    };

    struct RAY_TRACER_FACILITY_API SkinnedRayTracerInstance {
        MaterialType m_materialType = MaterialType::Default;
        int m_MLVQMaterialIndex;

        std::vector<SkinnedVertex> *m_skinnedVertices;
        std::vector<glm::uvec3> *m_triangles;
        std::vector<glm::mat4> *m_boneMatrices;

        size_t m_version;
        uint64_t m_handle = 0;

        RayTracerMaterial m_material;

        bool m_removeTag = false;
        glm::mat4 m_globalTransform;
    };

    enum PipelineType {
        DefaultRendering,
        IlluminationEstimation,

        PipelineSize
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

    struct SurfaceMaterial {
        MaterialType m_type;
        CudaBuffer m_buffer;
    };

    class RayTracer {
    public:
        bool m_requireUpdate = false;
        std::vector<RayTracerInstance> m_instances;
        std::vector<SkinnedRayTracerInstance> m_skinnedInstances;

        // ------------------------------------------------------------------
        // internal helper functions
        // ------------------------------------------------------------------
        [[nodiscard]] bool
        RenderToCamera(const EnvironmentProperties &environmentProperties, CameraProperties &cameraProperties,
                       const RayProperties &rayProperties);

        void EstimateIllumination(const size_t &size,
                                  const EnvironmentProperties &environmentProperties,
                                  const RayProperties &rayProperties,
                                  CudaBuffer &lightProbes, unsigned seed,
                                  float pushNormalDistance);

        void ScanPointCloud(const size_t &size, const EnvironmentProperties &environmentProperties,
                            CudaBuffer &samples);

        RayTracer();

        /*! build an acceleration structure for the given triangle mesh */
        void BuildAccelerationStructure();

        /*! constructs the shader binding table */
        void BuildShaderBindingTable(
                std::vector<std::pair<unsigned, cudaTextureObject_t>> &boundTextures,
                std::vector<cudaGraphicsResource_t> &boundResources);


        void LoadBtfMaterials(const std::vector<std::string> &folderPathes);

    protected:
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

        /*! one buffer per input mesh */
        std::vector<CudaBuffer> m_transformedPositionsBuffer;
        std::vector<CudaBuffer> m_transformedNormalsBuffer;
        std::vector<CudaBuffer> m_transformedTangentBuffer;
        std::vector<CudaBuffer> m_vertexColorBuffer;
        std::vector<CudaBuffer> m_texCoordBuffer;

        std::vector<CudaBuffer> m_boneMatricesBuffer;

        std::vector<SurfaceMaterial> m_surfaceMaterials;

        /*! one buffer per input mesh */
        std::vector<CudaBuffer> m_trianglesBuffer;
        //! buffer that keeps the (final, compacted) acceleration structure
        CudaBuffer m_acceleratedStructuresBuffer;
#pragma endregion

        friend class RayTracerCamera;
    };

} // namespace RayTracerFacility
