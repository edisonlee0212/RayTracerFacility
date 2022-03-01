#include "MLVQRenderer.hpp"
#include <RayTracerLayer.hpp>
#include <ProjectManager.hpp>
#include <RayTracer.hpp>
#include "Windows.hpp"
#include "EditorLayer.hpp"
#include "RayTracerCamera.hpp"
#include "TriangleIlluminationEstimator.hpp"
#include "PointCloudScanner.hpp"

using namespace RayTracerFacility;

std::shared_ptr<RayTracerCamera> RayTracerLayer::m_rayTracerCamera;

void RayTracerLayer::UpdateMeshesStorage(
        std::vector<MeshInstance> &meshesStorage,
        bool &rebuildAccelerationStructure, bool &updateShaderBindingTable) const {
    for (auto &i: meshesStorage) {
        i.m_removeTag = true;
    }
    if (const auto *rayTracedEntities =
                Entities::UnsafeGetPrivateComponentOwnersList<MeshRenderer>(
                        Entities::GetCurrentScene());
            rayTracedEntities && m_renderMeshRenderer) {
        for (auto entity: *rayTracedEntities) {
            if (!entity.IsEnabled())
                continue;
            auto meshRenderer =
                    entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
            if (!meshRenderer->IsEnabled())
                continue;
            auto mesh = meshRenderer->m_mesh.Get<Mesh>();
            auto material = meshRenderer->m_material.Get<Material>();
            if (!material || !mesh || mesh->UnsafeGetVertices().empty())
                continue;
            auto globalTransform = entity.GetDataComponent<GlobalTransform>().m_value;
            MeshInstance newRayTracerInstance;
            MeshInstance *rayTracerInstance = &newRayTracerInstance;
            bool needVerticesUpdate = false;
            bool needTransformUpdate = false;
            bool fromNew = true;
            bool needMaterialUpdate = false;
            for (auto &currentRayTracerInstance: meshesStorage) {
                if (currentRayTracerInstance.m_handle == meshRenderer->GetHandle().GetValue()) {
                    fromNew = false;
                    rayTracerInstance = &currentRayTracerInstance;
                    currentRayTracerInstance.m_removeTag = false;
                    if (globalTransform != currentRayTracerInstance.m_globalTransform) {
                        needTransformUpdate = true;
                    }
                    if (rayTracerInstance->m_version != mesh->GetVersion())
                        needVerticesUpdate = true;
                    if(CheckMaterial(rayTracerInstance->m_material, material)) needMaterialUpdate = true;
                }
            }
            rayTracerInstance->m_material.m_materialType = MaterialType::Default;
            rayTracerInstance->m_version = mesh->GetVersion();
            rayTracerInstance->m_vertices =
                reinterpret_cast<std::vector<Vertex> *>(&mesh->UnsafeGetVertices());
            rayTracerInstance->m_triangles = &mesh->UnsafeGetTriangles();
            if (fromNew || needVerticesUpdate || needTransformUpdate ||
                needMaterialUpdate) {
                updateShaderBindingTable = true;
                rayTracerInstance->m_handle = meshRenderer->GetHandle().GetValue();
            }
            if (fromNew || needVerticesUpdate) {
                rebuildAccelerationStructure = true;
                rayTracerInstance->m_instancing = false;
                if (fromNew) {
                    rayTracerInstance->m_globalTransform = globalTransform;
                }

            } else if (needTransformUpdate) {
                rebuildAccelerationStructure = true;
                rayTracerInstance->m_globalTransform = globalTransform;
            }
            if (fromNew)
                meshesStorage.push_back(newRayTracerInstance);
        }
    }
    if (const auto *rayTracedEntities =
                Entities::UnsafeGetPrivateComponentOwnersList<Particles>(
                        Entities::GetCurrentScene());
            rayTracedEntities && m_renderParticles) {
        for (auto entity: *rayTracedEntities) {
            if (!entity.IsEnabled())
                continue;
            auto particles =
                    entity.GetOrSetPrivateComponent<Particles>().lock();
            if (!particles->IsEnabled())
                continue;
            auto mesh = particles->m_mesh.Get<Mesh>();
            auto material = particles->m_material.Get<Material>();
            auto matrices = particles->m_matrices;
            if (!material || !mesh || mesh->UnsafeGetVertices().empty() || matrices->m_value.empty())
                continue;
            auto globalTransform = entity.GetDataComponent<GlobalTransform>().m_value;
            MeshInstance newRayTracerInstance;
            MeshInstance *rayTracerInstance = &newRayTracerInstance;
            bool needVerticesUpdate = false;
            bool needTransformUpdate = false;
            bool fromNew = true;
            bool needMaterialUpdate = false;
            for (auto &currentRayTracerInstance: meshesStorage) {
                if (currentRayTracerInstance.m_handle == particles->GetHandle().GetValue()) {
                    fromNew = false;
                    rayTracerInstance = &currentRayTracerInstance;
                    currentRayTracerInstance.m_removeTag = false;
                    if (globalTransform != currentRayTracerInstance.m_globalTransform) {
                        needTransformUpdate = true;
                    }
                    if (rayTracerInstance->m_version != mesh->GetVersion() ||
                        matrices->GetVersion() != rayTracerInstance->m_matricesVersion)
                        needVerticesUpdate = true;
                    if(CheckMaterial(rayTracerInstance->m_material, material)) needMaterialUpdate = true;
                }
            }
            rayTracerInstance->m_material.m_materialType = MaterialType::Default;
            rayTracerInstance->m_version = mesh->GetVersion();
            rayTracerInstance->m_matricesVersion = matrices->GetVersion();
            rayTracerInstance->m_vertices =
                reinterpret_cast<std::vector<Vertex> *>(&mesh->UnsafeGetVertices());
            rayTracerInstance->m_triangles = &mesh->UnsafeGetTriangles();
            rayTracerInstance->m_matrices = &matrices->m_value;
            if (fromNew || needVerticesUpdate || needTransformUpdate ||
                needMaterialUpdate) {
                updateShaderBindingTable = true;
                rayTracerInstance->m_handle = particles->GetHandle().GetValue();
            }
            if (fromNew || needVerticesUpdate) {
                rebuildAccelerationStructure = true;
                rayTracerInstance->m_instancing = true;
                if (fromNew) {
                    rayTracerInstance->m_globalTransform = globalTransform;
                }

            } else if (needTransformUpdate) {
                rebuildAccelerationStructure = true;
                rayTracerInstance->m_globalTransform = globalTransform;
            }
            if (fromNew)
                meshesStorage.push_back(newRayTracerInstance);
        }
    }
    if (const auto *rayTracedEntities =
                Entities::UnsafeGetPrivateComponentOwnersList<MLVQRenderer>(
                        Entities::GetCurrentScene());
            rayTracedEntities && m_renderMLVQRenderer) {
        for (auto entity: *rayTracedEntities) {
            if (!entity.IsEnabled())
                continue;
            auto mLVQRenderer =
                    entity.GetOrSetPrivateComponent<MLVQRenderer>().lock();
            if (!mLVQRenderer->IsEnabled())
                continue;
            auto mesh = mLVQRenderer->m_mesh.Get<Mesh>();
            if (!mesh || mesh->UnsafeGetVertices().empty())
                continue;
            auto globalTransform = entity.GetDataComponent<GlobalTransform>().m_value;
            MeshInstance newRayTracerInstance;
            MeshInstance *rayTracerInstance = &newRayTracerInstance;
            bool needVerticesUpdate = false;
            bool needTransformUpdate = false;
            bool fromNew = true;
            bool needMaterialUpdate = false;
            for (auto &currentRayTracerInstance: meshesStorage) {
                if (currentRayTracerInstance.m_handle == mLVQRenderer->GetHandle().GetValue()) {
                    fromNew = false;
                    rayTracerInstance = &currentRayTracerInstance;
                    currentRayTracerInstance.m_removeTag = false;
                    if (globalTransform != currentRayTracerInstance.m_globalTransform) {
                        needTransformUpdate = true;
                    }
                    if (rayTracerInstance->m_version != mesh->GetVersion())
                        needVerticesUpdate = true;
                    if (rayTracerInstance->m_material.m_MLVQMaterialIndex !=
                        mLVQRenderer->m_materialIndex) {
                        needMaterialUpdate = true;
                    }
                }
            }
            rayTracerInstance->m_material.m_materialType = MaterialType::MLVQ;
            rayTracerInstance->m_version = mesh->GetVersion();
            rayTracerInstance->m_vertices =
                reinterpret_cast<std::vector<Vertex> *>(&mesh->UnsafeGetVertices());
            rayTracerInstance->m_triangles = &mesh->UnsafeGetTriangles();
            if (fromNew || needVerticesUpdate || needTransformUpdate ||
                needMaterialUpdate) {
                updateShaderBindingTable = true;
                rayTracerInstance->m_material.m_MLVQMaterialIndex = mLVQRenderer->m_materialIndex;
                rayTracerInstance->m_material.m_normalTexture.m_textureId = 0;
                rayTracerInstance->m_material.m_albedoTexture.m_textureId = 0;
                rayTracerInstance->m_material.m_metallicTexture.m_textureId = 0;
                rayTracerInstance->m_material.m_roughnessTexture.m_textureId = 0;
                rayTracerInstance->m_handle = mLVQRenderer->GetHandle().GetValue();
            }
            if (fromNew || needVerticesUpdate) {
                rebuildAccelerationStructure = true;
                rayTracerInstance->m_instancing = false;
                if (fromNew) {
                    rayTracerInstance->m_globalTransform = globalTransform;
                }

            } else if (needTransformUpdate) {
                rebuildAccelerationStructure = true;
                rayTracerInstance->m_globalTransform = globalTransform;
            }
            if (fromNew)
                meshesStorage.push_back(newRayTracerInstance);
        }
    }
    for (int i = 0; i < meshesStorage.size(); i++) {
        if (meshesStorage[i].m_removeTag) {
            meshesStorage.erase(meshesStorage.begin() + i);
            i--;
            rebuildAccelerationStructure = true;
        }
    }
}

void RayTracerLayer::UpdateSkinnedMeshesStorage(
        std::vector<SkinnedMeshInstance> &meshesStorage,
        bool &rebuildAccelerationStructure, bool &updateShaderBindingTable) const {
    for (auto &i: meshesStorage) {
        i.m_removeTag = true;
    }
    if (const auto *rayTracedEntities =
                Entities::UnsafeGetPrivateComponentOwnersList<
                        SkinnedMeshRenderer>(Entities::GetCurrentScene());
            rayTracedEntities && m_renderSkinnedMeshRenderer) {
        for (auto entity: *rayTracedEntities) {
            if (!entity.IsEnabled())
                continue;
            auto skinnedMeshRenderer =
                    entity.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
            if (!skinnedMeshRenderer->IsEnabled())
                continue;
            auto mesh = skinnedMeshRenderer->m_skinnedMesh.Get<SkinnedMesh>();
            auto material = skinnedMeshRenderer->m_material.Get<Material>();
            if (!material || !mesh || mesh->UnsafeGetSkinnedVertices().empty() ||
                skinnedMeshRenderer->m_finalResults->m_value.empty())
                continue;
            auto globalTransform =
                    skinnedMeshRenderer->RagDoll()
                    ? glm::mat4(1.0f)
                    : entity.GetDataComponent<GlobalTransform>().m_value;
            SkinnedMeshInstance newRayTracerInstance;
            SkinnedMeshInstance *rayTracerInstance = &newRayTracerInstance;
            bool needVerticesUpdate = false;
            bool needTransformUpdate = false;
            bool fromNew = true;
            bool needMaterialUpdate = false;
            for (auto &currentRayTracerInstance: meshesStorage) {
                if (currentRayTracerInstance.m_handle == skinnedMeshRenderer->GetHandle().GetValue()) {
                    fromNew = false;
                    rayTracerInstance = &currentRayTracerInstance;
                    currentRayTracerInstance.m_removeTag = false;
                    if (globalTransform != currentRayTracerInstance.m_globalTransform) {
                        needTransformUpdate = true;
                    }
                    if (rayTracerInstance->m_version != mesh->GetVersion() ||
                        (skinnedMeshRenderer->RagDoll() &&
                         !skinnedMeshRenderer->m_ragDollFreeze) ||
                        skinnedMeshRenderer->m_animator.Get<Animator>()
                                ->AnimatedCurrentFrame())
                        needVerticesUpdate = true;
                    if(CheckMaterial(rayTracerInstance->m_material, material)) needMaterialUpdate = true;
                }
            }
            rayTracerInstance->m_version = mesh->GetVersion();
            rayTracerInstance->m_skinnedVertices =
                reinterpret_cast<std::vector<SkinnedVertex> *>(
                    &mesh->UnsafeGetSkinnedVertices());
            rayTracerInstance->m_boneMatrices =
                reinterpret_cast<std::vector<glm::mat4> *>(
                    &skinnedMeshRenderer->m_finalResults.get()->m_value);
            rayTracerInstance->m_triangles = &mesh->UnsafeGetTriangles();
            if (fromNew || needVerticesUpdate || needTransformUpdate ||
                needMaterialUpdate) {
                updateShaderBindingTable = true;
                rayTracerInstance->m_handle = skinnedMeshRenderer->GetHandle().GetValue();
            }
            if (fromNew || needVerticesUpdate) {
                rebuildAccelerationStructure = true;
                if (fromNew) {
                    rayTracerInstance->m_globalTransform = globalTransform;
                }

            } else if (needTransformUpdate) {
                rebuildAccelerationStructure = true;
                rayTracerInstance->m_globalTransform = globalTransform;
            }
            if (fromNew)
                meshesStorage.push_back(newRayTracerInstance);
        }
    }
    for (int i = 0; i < meshesStorage.size(); i++) {
        if (meshesStorage[i].m_removeTag) {
            meshesStorage.erase(meshesStorage.begin() + i);
            i--;
            rebuildAccelerationStructure = true;
        }
    }
}

void RayTracerLayer::UpdateScene() {
    bool rebuildAccelerationStructure = false;
    bool updateShaderBindingTable = false;
    auto &meshesStorage = CudaModule::GetRayTracer()->m_instances;
    auto &skinnedMeshesStorage = CudaModule::GetRayTracer()->m_skinnedInstances;
    UpdateMeshesStorage(meshesStorage, rebuildAccelerationStructure,
                        updateShaderBindingTable);
    UpdateSkinnedMeshesStorage(skinnedMeshesStorage, rebuildAccelerationStructure,
                               updateShaderBindingTable);
    unsigned int envMapId = 0;
    auto& envSettings = Entities::GetCurrentScene()->m_environmentSettings;
    if(envSettings.m_environmentType == UniEngine::EnvironmentType::EnvironmentalMap) {
        auto environmentalMap = envSettings.m_environmentalMap.Get<EnvironmentalMap>();
        if (environmentalMap) {
            auto cubeMap = environmentalMap->GetCubemap().Get<Cubemap>();
            if (cubeMap) envMapId = cubeMap->Texture()->Id();
        }
    }else if(envSettings.m_backgroundColor != m_environmentProperties.m_color){
        m_environmentProperties.m_color = envSettings.m_backgroundColor;
        updateShaderBindingTable = true;
    }
    if(m_environmentProperties.m_skylightIntensity != envSettings.m_ambientLightIntensity){
        m_environmentProperties.m_skylightIntensity = envSettings.m_ambientLightIntensity;
        updateShaderBindingTable = true;
    }
    if(m_environmentProperties.m_gamma != envSettings.m_environmentGamma){
        m_environmentProperties.m_gamma = envSettings.m_environmentGamma;
        updateShaderBindingTable = true;
    }
    if(m_environmentProperties.m_environmentalMapId != envMapId){
        m_environmentProperties.m_environmentalMapId = envMapId;
        updateShaderBindingTable = true;
    }

    CudaModule::GetRayTracer()->m_requireUpdate = false;
    if (rebuildAccelerationStructure &&
        (!meshesStorage.empty() || !skinnedMeshesStorage.empty())) {
        CudaModule::GetRayTracer()->BuildAccelerationStructure();
        CudaModule::GetRayTracer()->m_requireUpdate = true;
    } else if (updateShaderBindingTable) {
        CudaModule::GetRayTracer()->m_requireUpdate = true;
    }
}

void RayTracerLayer::OnCreate() {
    CudaModule::Init();
    ClassRegistry::RegisterPrivateComponent<MLVQRenderer>(
            "MLVQRenderer");
    ClassRegistry::RegisterPrivateComponent<TriangleIlluminationEstimator>(
            "TriangleIlluminationEstimator");
    ClassRegistry::RegisterPrivateComponent<RayTracerCamera>(
            "RayTracerCamera");
    ClassRegistry::RegisterPrivateComponent<PointCloudScanner>(
            "PointCloudScanner");

    m_sceneCamera = Serialization::ProduceSerializable<RayTracerCamera>();
    m_sceneCamera->OnCreate();

    Application::RegisterPostAttachSceneFunction([&](const std::shared_ptr<Scene> &scene) {
        m_rayTracerCamera.reset();
    });
}


void RayTracerLayer::LateUpdate() {
    UpdateScene();

    if (!CudaModule::GetRayTracer()->m_instances.empty() ||
        !CudaModule::GetRayTracer()->m_skinnedInstances.empty()) {
        auto editorLayer = Application::GetLayer<EditorLayer>();
        if (m_enableSceneCamera && editorLayer && m_renderingEnabled) {
            m_sceneCamera->Ready(editorLayer->m_sceneCameraPosition, editorLayer->m_sceneCameraRotation);
            m_sceneCamera->m_rendered = CudaModule::GetRayTracer()->RenderToCamera(m_environmentProperties,
                                                                                   m_sceneCamera->m_cameraProperties,
                                                                                   m_sceneCamera->m_rayProperties);
        }
        auto *entities = Entities::UnsafeGetPrivateComponentOwnersList<RayTracerCamera>(
                Entities::GetCurrentScene());
        m_rayTracerCamera.reset();
        if (entities) {
            bool check = false;
            for (const auto &entity: *entities) {
                if (!entity.IsEnabled()) continue;
                auto rayTracerCamera = entity.GetOrSetPrivateComponent<RayTracerCamera>().lock();
                if (!rayTracerCamera->IsEnabled()) continue;
                auto globalTransform = rayTracerCamera->GetOwner().GetDataComponent<GlobalTransform>().m_value;
                rayTracerCamera->Ready(globalTransform[3], glm::quat_cast(globalTransform));
                rayTracerCamera->m_rendered = CudaModule::GetRayTracer()->RenderToCamera(m_environmentProperties,
                                                                                         rayTracerCamera->m_cameraProperties,
                                                                                         rayTracerCamera->m_rayProperties);
                if (!check) {
                    if (rayTracerCamera->m_mainCamera) {
                        m_rayTracerCamera = rayTracerCamera;
                        check = true;
                    }
                } else {
                    rayTracerCamera->m_mainCamera = false;
                }
            }
        }
    }

}

void RayTracerLayer::OnInspect() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::Checkbox("Ray Tracer Manager", &m_enableMenus);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    if (ImGui::Begin("Ray Tracer Manager")) {
        ImGui::Checkbox("Mesh Renderer", &m_renderMeshRenderer);
        ImGui::Checkbox("Particles", &m_renderParticles);
        ImGui::Checkbox("Skinned Mesh Renderer", &m_renderSkinnedMeshRenderer);
        ImGui::Checkbox("MLVQ Renderer", &m_renderSkinnedMeshRenderer);
        ImGui::Checkbox("Scene Camera", &m_enableSceneCamera);
        if (ImGui::TreeNode("Scene Camera Settings")) {
            m_sceneCamera->OnInspect();
            ImGui::TreePop();
        }
        if (ImGui::TreeNodeEx("Environment Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
            m_environmentProperties.OnInspect();
            ImGui::TreePop();
        }
        if (ImGui::Button("Load all MLVQ Materials")) {
            std::vector<std::string> pathes;
            std::filesystem::path folder("../Resources/btfs");
            for (auto &entry: std::filesystem::directory_iterator(folder)) {
                pathes.push_back(entry.path().string());
            }
            CudaModule::GetRayTracer()->LoadBtfMaterials(pathes);
        }
    }
    ImGui::End();
    RayCameraWindow();
    if (m_enableSceneCamera) SceneCameraWindow();
}

void RayTracerLayer::OnDestroy() { CudaModule::Terminate(); }

void RayTracerLayer::SceneCameraWindow() {
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    if (m_rightMouseButtonHold &&
        !Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT, Windows::GetWindow())) {
        m_rightMouseButtonHold = false;
        m_startMouse = false;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0});
    if (ImGui::Begin("Scene (Ray)")) {
        if (ImGui::BeginChild("RaySceneRenderer", ImVec2(0, 0), false,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{5, 5});
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Settings")) {
                    ImGui::DragFloat("Resolution multiplier", &m_resolutionMultiplier,
                                     0.01f, 0.1f, 1.0f);
                    m_sceneCamera->m_cameraProperties.OnInspect();
                    m_sceneCamera->m_rayProperties.OnInspect();
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }
            ImGui::PopStyleVar();
            ImVec2 viewPortSize = ImGui::GetWindowSize();
            viewPortSize.y -= 20;
            if (viewPortSize.y < 0)
                viewPortSize.y = 0;
            if (m_sceneCamera->m_allowAutoResize)
                m_sceneCamera->m_frameSize =
                        glm::vec2(viewPortSize.x, viewPortSize.y) *
                        m_resolutionMultiplier;
            if (m_sceneCamera->m_rendered) {
                ImGui::Image(reinterpret_cast<ImTextureID>(m_sceneCamera->m_cameraProperties.m_outputTextureId),
                             viewPortSize, ImVec2(0, 1), ImVec2(1, 0));
                editorLayer->CameraWindowDragAndDrop();
            } else
                ImGui::Text("No mesh in the scene!");
            if (ImGui::IsWindowFocused()) {
                const bool valid = true;
                const glm::vec2 mousePosition =
                        Inputs::GetMouseAbsolutePositionInternal(
                                Windows::GetWindow());
                if (valid) {
                    if (!m_startMouse) {
                        m_lastX = mousePosition.x;
                        m_lastY = mousePosition.y;
                        m_startMouse = true;
                    }
                    const float xOffset = mousePosition.x - m_lastX;
                    const float yOffset = -mousePosition.y + m_lastY;
                    m_lastX = mousePosition.x;
                    m_lastY = mousePosition.y;
#pragma region Scene Camera Controller
                    if (!m_rightMouseButtonHold &&
                        Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT,
                                                 Windows::GetWindow())) {
                        m_rightMouseButtonHold = true;
                    }
                    if (m_rightMouseButtonHold &&
                        !editorLayer->m_lockCamera) {
                        const glm::vec3 front =
                                editorLayer->m_sceneCameraRotation *
                                glm::vec3(0, 0, -1);
                        const glm::vec3 right =
                                editorLayer->m_sceneCameraRotation *
                                glm::vec3(1, 0, 0);
                        if (Inputs::GetKeyInternal(GLFW_KEY_W,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition +=
                                    front * static_cast<float>(Application::Time().DeltaTime()) *
                                    editorLayer->m_velocity;
                        }
                        if (Inputs::GetKeyInternal(GLFW_KEY_S,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition -=
                                    front * static_cast<float>(Application::Time().DeltaTime()) *
                                    editorLayer->m_velocity;
                        }
                        if (Inputs::GetKeyInternal(GLFW_KEY_A,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition -=
                                    right * static_cast<float>(Application::Time().DeltaTime()) *
                                    editorLayer->m_velocity;
                        }
                        if (Inputs::GetKeyInternal(GLFW_KEY_D,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition +=
                                    right * static_cast<float>(Application::Time().DeltaTime()) *
                                    editorLayer->m_velocity;
                        }
                        if (Inputs::GetKeyInternal(GLFW_KEY_LEFT_SHIFT,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition.y +=
                                    editorLayer->m_velocity *
                                    static_cast<float>(Application::Time().DeltaTime());
                        }
                        if (Inputs::GetKeyInternal(GLFW_KEY_LEFT_CONTROL,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition.y -=
                                    editorLayer->m_velocity *
                                    static_cast<float>(Application::Time().DeltaTime());
                        }
                        if (xOffset != 0.0f || yOffset != 0.0f) {
                            editorLayer->m_sceneCameraYawAngle +=
                                    xOffset * editorLayer->m_sensitivity;
                            editorLayer->m_sceneCameraPitchAngle +=
                                    yOffset * editorLayer->m_sensitivity;
                            if (editorLayer->m_sceneCameraPitchAngle > 89.0f)
                                editorLayer->m_sceneCameraPitchAngle = 89.0f;
                            if (editorLayer->m_sceneCameraPitchAngle < -89.0f)
                                editorLayer->m_sceneCameraPitchAngle = -89.0f;

                            editorLayer->m_sceneCameraRotation =
                                    UniEngine::Camera::ProcessMouseMovement(
                                            editorLayer->m_sceneCameraYawAngle,
                                            editorLayer->m_sceneCameraPitchAngle,
                                            false);
                        }
                    }
#pragma endregion
                }
            }
        }
        ImGui::EndChild();
        auto *window = ImGui::FindWindowByName("Scene (Ray)");
        m_renderingEnabled = !(window->Hidden && !window->Collapsed);
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void RayTracerLayer::RayCameraWindow() {
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0});
    if (ImGui::Begin("Camera (Ray)")) {
        if (ImGui::BeginChild("RayCameraRenderer", ImVec2(0, 0), false,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
            ImVec2 viewPortSize = ImGui::GetWindowSize();
            viewPortSize.y -= 20;
            if (viewPortSize.y < 0)
                viewPortSize.y = 0;
            if (m_rayTracerCamera) {
                if (m_rayTracerCamera->m_allowAutoResize)
                    m_rayTracerCamera->m_frameSize = glm::vec2(viewPortSize.x, viewPortSize.y);
                if (m_rayTracerCamera->m_rendered) {
                    ImGui::Image(reinterpret_cast<ImTextureID>(m_rayTracerCamera->m_cameraProperties.m_outputTextureId),
                                 viewPortSize, ImVec2(0, 1), ImVec2(1, 0));
                    editorLayer->CameraWindowDragAndDrop();
                } else
                    ImGui::Text("No mesh in the scene!");
            } else {
                ImGui::Text("No camera attached!");
            }
        }
        ImGui::EndChild();
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void RayTracerLayer::Update() {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0});
    if (ImGui::Begin("Camera (Ray)")) {
        if (ImGui::BeginChild("RayCameraRenderer", ImVec2(0, 0), false,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
            if (m_rayTracerCamera && m_rayTracerCamera->m_rendered && ImGui::IsWindowFocused()) {
                Application::GetLayer<EditorLayer>()->m_mainCameraFocusOverride = true;
            }
        }
        ImGui::EndChild();
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

bool
RayTracerLayer::CheckMaterial(RayTracerMaterial& rayTracerMaterial, const std::shared_ptr<Material> &material) const {
    bool changed = false;
    if (rayTracerMaterial.m_materialProperties.m_surfaceColor != material->m_albedoColor){
        changed = true;
        rayTracerMaterial.m_materialProperties.m_surfaceColor = material->m_albedoColor;
    }
    if (rayTracerMaterial.m_materialProperties.m_subsurfaceColor != material->m_subsurfaceColor){
        changed = true;
        rayTracerMaterial.m_materialProperties.m_subsurfaceColor = material->m_subsurfaceColor;
    }
    if (rayTracerMaterial.m_materialProperties.m_subsurfaceRadius != material->m_subsurfaceRadius){
        changed = true;
        rayTracerMaterial.m_materialProperties.m_subsurfaceRadius = material->m_subsurfaceRadius;
    }
    if (rayTracerMaterial.m_materialProperties.m_subsurfaceFactor != material->m_subsurfaceFactor){
        changed = true;
        rayTracerMaterial.m_materialProperties.m_subsurfaceFactor = glm::clamp(material->m_subsurfaceFactor, 0.0f, 1.0f);
    }
    if (rayTracerMaterial.m_materialProperties.m_roughness != material->m_roughness){
        changed = true;
        rayTracerMaterial.m_materialProperties.m_roughness = material->m_roughness;
    }
    if (rayTracerMaterial.m_materialProperties.m_emission != material->m_emission){
        changed = true;
        rayTracerMaterial.m_materialProperties.m_emission = material->m_emission;
    }
    if (rayTracerMaterial.m_materialProperties.m_metallic != material->m_metallic){
        changed = true;
        rayTracerMaterial.m_materialProperties.m_metallic = material->m_metallic;
    }
    auto albedoTexture = material->m_albedoTexture.Get<Texture2D>();
    if (albedoTexture &&
            albedoTexture->UnsafeGetGLTexture()) {
        if (albedoTexture
                    ->UnsafeGetGLTexture()
                    ->Id() != rayTracerMaterial.m_albedoTexture.m_textureId) {
            changed = true;
            rayTracerMaterial.m_albedoTexture.m_textureId =
                    albedoTexture
                            ->UnsafeGetGLTexture()
                            ->Id();
            rayTracerMaterial.m_albedoTexture.m_channel = (int)albedoTexture->m_textureColorType;
        }
    } else if (rayTracerMaterial.m_albedoTexture.m_textureId != 0) {
        changed = true;
        rayTracerMaterial.m_albedoTexture.m_textureId = 0;
    }
    auto normalTexture = material->m_normalTexture.Get<Texture2D>();
    if (normalTexture &&
            normalTexture->UnsafeGetGLTexture()) {
        if (normalTexture
                    ->UnsafeGetGLTexture()
                    ->Id() != rayTracerMaterial.m_normalTexture.m_textureId) {
            changed = true;
            rayTracerMaterial.m_normalTexture.m_textureId =
                    normalTexture
                            ->UnsafeGetGLTexture()
                            ->Id();
            rayTracerMaterial.m_normalTexture.m_channel = (int)normalTexture->m_textureColorType;
        }
    } else if (rayTracerMaterial.m_normalTexture.m_textureId != 0) {
        changed = true;
        rayTracerMaterial.m_normalTexture.m_textureId = 0;
    }
    auto roughnessTexture = material->m_normalTexture.Get<Texture2D>();
    if (roughnessTexture &&
            roughnessTexture->UnsafeGetGLTexture()) {
        if (roughnessTexture
                    ->UnsafeGetGLTexture()
                    ->Id() != rayTracerMaterial.m_roughnessTexture.m_textureId) {
            changed = true;
            rayTracerMaterial.m_roughnessTexture.m_textureId =
                    normalTexture
                            ->UnsafeGetGLTexture()
                            ->Id();
            rayTracerMaterial.m_roughnessTexture.m_channel = (int)roughnessTexture->m_textureColorType;
        }
    } else if (rayTracerMaterial.m_roughnessTexture.m_textureId != 0) {
        changed = true;
        rayTracerMaterial.m_roughnessTexture.m_textureId = 0;
    }
    auto metallicTexture = material->m_metallicTexture.Get<Texture2D>();
    if (metallicTexture &&
            metallicTexture->UnsafeGetGLTexture()) {
        if (metallicTexture
                    ->UnsafeGetGLTexture()
                    ->Id() != rayTracerMaterial.m_metallicTexture.m_textureId) {
            changed = true;
            rayTracerMaterial.m_metallicTexture.m_textureId =
                    metallicTexture
                            ->UnsafeGetGLTexture()
                            ->Id();
            rayTracerMaterial.m_metallicTexture.m_channel = (int)metallicTexture->m_textureColorType;
        }
    } else if (rayTracerMaterial.m_metallicTexture.m_textureId != 0) {
        changed = true;
        rayTracerMaterial.m_metallicTexture.m_textureId = 0;
    }
    return changed;
}

