// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#ifdef RAYTRACERFACILITY
#include <RayTracerManager.hpp>
#include "MLVQRenderer.hpp"
#endif


#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif

int main() {
#ifdef RAYTRACERFACILITY
    ClassRegistry::RegisterPrivateComponent<MLVQRenderer>(
            "MLVQRenderer");
#endif
    const bool enableRayTracing = true;
    ApplicationConfigs applicationConfigs;
    Application::Create(applicationConfigs);
#ifdef RAYTRACERFACILITY
    Application::PushLayer<RayTracerManager>();
#endif
#pragma region Engine Loop
    Application::Start();
#pragma endregion
    Application::End();
}
