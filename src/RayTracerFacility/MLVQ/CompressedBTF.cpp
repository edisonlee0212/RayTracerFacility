//
// Created by lllll on 10/13/2022.
//

#include "CompressedBTF.hpp"

using namespace RayTracerFacility;

void CompressedBTF::OnInspect() {
    FileUtils::OpenFolder("Import Database", [&](const std::filesystem::path &path) {
        try{
            bool succeed = m_btfBase.Init(path.string());
            if(succeed) m_version++;
            UNIENGINE_LOG((std::string("BTF Material import ") + (succeed ? "succeed" : "failed")))
        } catch (const std::exception& e){
            UNIENGINE_ERROR(std::string(e.what()))
        }
    }, false);
}
