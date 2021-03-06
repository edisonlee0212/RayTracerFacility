//
// Created by lllll on 9/3/2021.
//

#include "MLVQRenderer.hpp"

using namespace RayTracerFacility;

#include <Editor.hpp>
#include <MeshRenderer.hpp>

using namespace UniEngine;

const char *MLVQMaterials[]{"None", "alu", "corduroy", "fabric01", "impalla",
                            "proposte", "pulli", "wallpaper", "wool"};

void MLVQRenderer::OnInspect() {
    if(ImGui::Button("Sync")) Sync();
    Editor::DragAndDropButton<Mesh>(m_mesh, "Mesh");
    ImGui::Text("Material: ");
    ImGui::SameLine();
    ImGui::Combo("MLVQ Material", &m_materialIndex, MLVQMaterials,
                 IM_ARRAYSIZE(MLVQMaterials));
}

void MLVQRenderer::Sync() {
    Entity owner = GetOwner();
    auto scene = GetScene();
    if (scene->HasPrivateComponent<MeshRenderer>(owner)) {
        auto mmr = scene->GetOrSetPrivateComponent<MeshRenderer>(owner).lock();
        m_mesh = mmr->m_mesh;
    }
}

void MLVQRenderer::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_materialIndex" << YAML::Value << m_materialIndex;
    m_mesh.Save("m_mesh", out);
}

void MLVQRenderer::Deserialize(const YAML::Node &in) {
    m_materialIndex = in["m_materialIndex"].as<int>();
    m_mesh.Load("m_mesh", in);
}

void MLVQRenderer::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_mesh);
}