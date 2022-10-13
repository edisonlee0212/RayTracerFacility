#pragma once
#include <UniEngine-pch.hpp>
#include <memory>
#include <ray_tracer_facility_export.h>
#include "OpenGLUtils.hpp"
#include "IAsset.hpp"
#include "Scene.hpp"
#include "Transform.hpp"
#include "Vertex.hpp"
#include "filesystem"
#include "BTFBase.cuh"
using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API CompressedBTF : public IAsset {
    public:
        size_t m_version = 0;
        BtfBase m_btfBase;
        void OnInspect() override;
    };
}