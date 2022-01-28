#include <RayTracerUtilities.cuh>

namespace RayTracerFacility {
extern "C" __constant__ CameraRenderingLaunchParams cameraRenderingLaunchParams;

struct DefaultRenderingSpSamplerRayData {
  unsigned m_instanceId;
  glm::vec3 m_p0;
  glm::vec3 m_p1;
  glm::vec3 m_n1;
  bool m_found = false;
};
#pragma region Closest hit functions
extern "C" __global__ void __closesthit__radiance() {
  const float3 rayDirectionInternal = optixGetWorldRayDirection();
  glm::vec3 rayDirection = glm::vec3(
      rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
#pragma region Retrive information
  const auto &sbtData = *(const DefaultSbtData *)optixGetSbtDataPointer();
  const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
  const int primitiveId = optixGetPrimitiveIndex();

  auto indices = sbtData.m_mesh.GetIndices(primitiveId);
  auto texCoord =
      sbtData.m_mesh.GetTexCoord(triangleBarycentricsInternal, indices);
  auto normal = sbtData.m_mesh.GetNormal(triangleBarycentricsInternal, indices);
  if (glm::dot(rayDirection, normal) > 0.0f) {
    normal = -normal;
  }
  auto tangent =
      sbtData.m_mesh.GetTangent(triangleBarycentricsInternal, indices);
  auto hitPoint =
      sbtData.m_mesh.GetPosition(triangleBarycentricsInternal, indices);
#pragma endregion
  PerRayData<glm::vec3> &perRayData =
      *GetRayDataPointer<PerRayData<glm::vec3>>();
  unsigned hitCount = perRayData.m_hitCount + 1;

  // start with some ambient term
  auto energy = glm::vec3(0.0f);
  uint32_t u0, u1;
  PackRayDataPointer(&perRayData, u0, u1);
  perRayData.m_hitCount = hitCount;
  perRayData.m_energy = glm::vec3(0.0f);

  switch (sbtData.m_materialType) {
  case MaterialType::VertexColor: {
    perRayData.m_energy =
        sbtData.m_mesh.GetColor(triangleBarycentricsInternal, indices);
  } break;
  case MaterialType::Default: {
    static_cast<DefaultMaterial *>(sbtData.m_material)
        ->ApplyNormalTexture(normal, texCoord, tangent);
    float metallic =
        static_cast<DefaultMaterial *>(sbtData.m_material)->GetMetallic(texCoord);
    float roughness =
        static_cast<DefaultMaterial *>(sbtData.m_material)->GetRoughness(texCoord);
    glm::vec3 albedoColor =
        static_cast<DefaultMaterial *>(sbtData.m_material)->GetAlbedo(texCoord);
    if (perRayData.m_hitCount <=
        cameraRenderingLaunchParams.m_rayTracerProperties.m_rayProperties
            .m_bounces) {
      energy = glm::vec3(0.0f);
      float f = 1.0f;
      if (metallic >= 0.0f)
        f = (metallic + 2) / (metallic + 1);
      float3 incidentRayOrigin;
      float3 newRayDirectionInternal;
      BRDF(metallic, perRayData.m_random, normal, hitPoint, rayDirection,
           incidentRayOrigin, newRayDirectionInternal);
      optixTrace(
          cameraRenderingLaunchParams.m_traversable, incidentRayOrigin,
          newRayDirectionInternal,
          1e-3f, // tmin
          1e20f, // tmax
          0.0f,  // rayTime
          static_cast<OptixVisibilityMask>(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
          static_cast<int>(
              CameraRenderingRayType::RadianceRayType),           // SBT offset
          static_cast<int>(CameraRenderingRayType::RayTypeCount), // SBT stride
          static_cast<int>(
              CameraRenderingRayType::RadianceRayType), // missSBTIndex
          u0, u1);
      energy +=
          albedoColor *
          glm::clamp(
              glm::abs(glm::dot(normal, glm::vec3(newRayDirectionInternal.x,
                                                  newRayDirectionInternal.y,
                                                  newRayDirectionInternal.z))) *
                      roughness +
                  (1.0f - roughness) * f,
              0.0f, 1.0f) *
          perRayData.m_energy;
    }
    if (hitCount == 1) {
      perRayData.m_normal = normal;
      perRayData.m_albedo = albedoColor;
    }
    perRayData.m_energy =
        energy +
        static_cast<DefaultMaterial *>(sbtData.m_material)->m_diffuseIntensity *
            albedoColor;

  } break;
  case MaterialType::MLVQ: {
    glm::vec3 btfColor;
    if (perRayData.m_hitCount <=
        cameraRenderingLaunchParams.m_rayTracerProperties.m_rayProperties
            .m_bounces) {
      energy = glm::vec3(0.0f);
      float f = 1.0f;
      glm::vec3 reflected = Reflect(rayDirection, normal);
      glm::vec3 newRayDirection =
          RandomSampleHemisphere(perRayData.m_random, reflected, 1.0f);
      static_cast<MLVQMaterial *>(sbtData.m_material)
          ->GetValue(texCoord, rayDirection, newRayDirection, normal, tangent,
                     btfColor,
                     false /*(perRayData.m_printInfo && sampleID == 0)*/);
      auto origin = hitPoint;
      origin += normal * 1e-3f;
      float3 incidentRayOrigin = make_float3(origin.x, origin.y, origin.z);
      float3 newRayDirectionInternal =
          make_float3(newRayDirection.x, newRayDirection.y, newRayDirection.z);
      optixTrace(
          cameraRenderingLaunchParams.m_traversable, incidentRayOrigin,
          newRayDirectionInternal,
          1e-3f, // tmin
          1e20f, // tmax
          0.0f,  // rayTime
          static_cast<OptixVisibilityMask>(255),
          OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
          static_cast<int>(
              CameraRenderingRayType::RadianceRayType),           // SBT offset
          static_cast<int>(CameraRenderingRayType::RayTypeCount), // SBT stride
          static_cast<int>(
              CameraRenderingRayType::RadianceRayType), // missSBTIndex
          u0, u1);
      energy += btfColor * perRayData.m_energy;
    }
    if (hitCount == 1) {
      perRayData.m_normal = normal;
      perRayData.m_albedo = btfColor;
    }
    perRayData.m_energy = energy;
  } break;
  }
}
extern "C" __global__ void __closesthit__sampleSp() {
  DefaultRenderingSpSamplerRayData &perRayData =
      *GetRayDataPointer<DefaultRenderingSpSamplerRayData>();
  assert(perRayData.m_instanceId == optixGetInstanceId());
  const auto &sbtData = *(const DefaultSbtData *)optixGetSbtDataPointer();
  const auto indices = sbtData.m_mesh.GetIndices(optixGetPrimitiveIndex());
  const auto centrics = optixGetTriangleBarycentrics();
  perRayData.m_p1 = sbtData.m_mesh.GetPosition(centrics, indices);
  perRayData.m_n1 = sbtData.m_mesh.GetNormal(centrics, indices);
  perRayData.m_found = true;
}
#pragma endregion
#pragma region Any hit functions
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __anyhit__sampleSp() {
  DefaultRenderingSpSamplerRayData &perRayData =
      *GetRayDataPointer<DefaultRenderingSpSamplerRayData>();
  if (perRayData.m_instanceId != optixGetInstanceId())
    optixIgnoreIntersection();
  const auto &sbtData = *(const DefaultSbtData *)optixGetSbtDataPointer();
  const auto indices = sbtData.m_mesh.GetIndices(optixGetPrimitiveIndex());
  const auto hitPoint =
      sbtData.m_mesh.GetPosition(optixGetTriangleBarycentrics(), indices);
  const auto origin = optixGetWorldRayOrigin();
  const float distance =
      glm::distance(hitPoint, glm::vec3(origin.x, origin.y, origin.z));
  // if (distance > sbtData.m_material.GetRadiusMax())
  // optixIgnoreIntersection();
}
#pragma endregion
#pragma region Miss functions
extern "C" __global__ void __miss__radiance() {
  PerRayData<glm::vec3> &perRayData =
      *GetRayDataPointer<PerRayData<glm::vec3>>();
  const float3 rayDir = optixGetWorldRayDirection();
  float3 rayOrigin = optixGetWorldRayOrigin();
  glm::vec3 rayOrig = glm::vec3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
  glm::vec3 rayDirection = glm::vec3(rayDir.x, rayDir.y, rayDir.z);

  glm::vec3 environmentalLightColor = CalculateEnvironmentalLight(
      rayOrig, rayDirection,
      cameraRenderingLaunchParams.m_rayTracerProperties.m_environment);
  perRayData.m_albedo = perRayData.m_energy = environmentalLightColor;
}
extern "C" __global__ void __miss__sampleSp() {}
#pragma endregion
#pragma region Main ray generation
extern "C" __global__ void __raygen__renderFrame() {
  float ix = optixGetLaunchIndex().x;
  float iy = optixGetLaunchIndex().y;
  const uint32_t fbIndex =
      ix + iy * cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x;

  // compute a test pattern based on pixel ID

  PerRayData<glm::vec3> cameraRayData;
  cameraRayData.m_hitCount = 0;
  cameraRayData.m_random.Init(
      ix + cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size.x * iy,
      cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId);
  cameraRayData.m_energy = glm::vec3(0);
  cameraRayData.m_normal = glm::vec3(0);
  cameraRayData.m_albedo = glm::vec3(0);
  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  PackRayDataPointer(&cameraRayData, u0, u1);

  const auto numPixelSamples = cameraRenderingLaunchParams.m_rayTracerProperties
                                   .m_rayProperties.m_samples;
  auto pixelColor = glm::vec3(0.f);
  auto pixelNormal = glm::vec3(0.f);
  auto pixelAlbedo = glm::vec3(0.f);

  for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
    // normalized screen plane position, in [0,1]^2
    // iw: note for de-noising that's not actually correct - if we
    // assume that the camera should only(!) cover the de-noised
    // screen then the actual screen plane we should be using during
    // rendering is slightly larger than [0,1]^2
    glm::vec2 screen;
    screen = glm::vec2(ix + cameraRayData.m_random(),
                       iy + cameraRayData.m_random()) /
             glm::vec2(
                 cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_size);
    glm::vec3 rayDir = glm::normalize(
        cameraRenderingLaunchParams.m_cameraProperties.m_direction +
        (screen.x - 0.5f) *
            cameraRenderingLaunchParams.m_cameraProperties.m_horizontal +
        (screen.y - 0.5f) *
            cameraRenderingLaunchParams.m_cameraProperties.m_vertical);
    float3 rayOrigin =
        make_float3(cameraRenderingLaunchParams.m_cameraProperties.m_from.x,
                    cameraRenderingLaunchParams.m_cameraProperties.m_from.y,
                    cameraRenderingLaunchParams.m_cameraProperties.m_from.z);
    float3 rayDirection = make_float3(rayDir.x, rayDir.y, rayDir.z);

    optixTrace(
        cameraRenderingLaunchParams.m_traversable, rayOrigin, rayDirection,
        0.f,   // tmin
        1e20f, // tmax
        0.0f,  // rayTime
        static_cast<OptixVisibilityMask>(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
        static_cast<int>(CameraRenderingRayType::RadianceRayType), // SBT offset
        static_cast<int>(CameraRenderingRayType::RayTypeCount),    // SBT stride
        static_cast<int>(
            CameraRenderingRayType::RadianceRayType), // missSBTIndex
        u0, u1);
    pixelColor += cameraRayData.m_energy / static_cast<float>(numPixelSamples);
    pixelNormal += cameraRayData.m_normal / static_cast<float>(numPixelSamples);
    pixelAlbedo += cameraRayData.m_albedo / static_cast<float>(numPixelSamples);
    cameraRayData.m_energy = glm::vec3(0.0f);
    cameraRayData.m_normal = glm::vec3(0.0f);
    cameraRayData.m_albedo = glm::vec3(0.0f);
    cameraRayData.m_hitCount = 0;
  }

  // and write/accumulate to frame buffer ...
  if (cameraRenderingLaunchParams.m_cameraProperties.m_accumulate) {
    if (cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId > 1) {
      glm::vec3 currentGammaCorrectedColor =
          cameraRenderingLaunchParams.m_cameraProperties.m_frame
              .m_colorBuffer[fbIndex];
      glm::vec3 accumulatedColor = glm::vec3(glm::pow(
          currentGammaCorrectedColor,
          glm::vec3(cameraRenderingLaunchParams.m_cameraProperties.m_gamma)));
      pixelColor +=
          static_cast<float>(cameraRenderingLaunchParams.m_cameraProperties
                                 .m_frame.m_frameId) *
          accumulatedColor;
      pixelColor /= static_cast<float>(
          cameraRenderingLaunchParams.m_cameraProperties.m_frame.m_frameId + 1);
    }
  }
  auto gammaCorrectedColor = glm::pow(
      pixelColor,
      glm::vec3(1.0 / cameraRenderingLaunchParams.m_cameraProperties.m_gamma));
  // and write to frame buffer ...
  cameraRenderingLaunchParams.m_cameraProperties.m_frame
      .m_colorBuffer[fbIndex] = glm::vec4(gammaCorrectedColor, 1.0f);
  cameraRenderingLaunchParams.m_cameraProperties.m_frame
      .m_albedoBuffer[fbIndex] = glm::vec4(pixelAlbedo, 1.0f);
  cameraRenderingLaunchParams.m_cameraProperties.m_frame
      .m_normalBuffer[fbIndex] = glm::vec4(pixelNormal, 1.0f);
}
#pragma endregion
} // namespace RayTracerFacility
