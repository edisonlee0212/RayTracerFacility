//
// Created by lllll on 10/13/2022.
//

#include "CompressedBTF.hpp"

using namespace RayTracerFacility;

void CompressedBTF::OnInspect() {
    bool changed = false;
    FileUtils::OpenFolder("Import Database", [&](const std::filesystem::path &path) {
        try{
            bool succeed = m_btfBase.Init(path.string());
            if(succeed) changed = true;
            UNIENGINE_LOG((std::string("BTF Material import ") + (succeed ? "succeed" : "failed")))
        } catch (const std::exception& e){
            UNIENGINE_ERROR(std::string(e.what()))
        }
    }, false);

    if(m_btfBase.m_hasData) {
        if (ImGui::DragFloat("Multiplier", &m_btfBase.m_multiplier, 1.0f)) {
            changed = true;
        }

        if (ImGui::DragFloat("TexCoord Multiplier", &m_btfBase.m_texCoordMultiplier, 0.1f)) {
            changed = true;
        }

        if (ImGui::Checkbox("HDR", &m_btfBase.m_hdr)) {
            changed = true;
        }

        if (ImGui::DragFloat("HDR Value", &m_btfBase.m_hdrValue, 0.01f)){
            changed = true;
        }

    }
    if(changed) {
        m_saved = false;
        m_version++;
    }
}
