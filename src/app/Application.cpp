// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#ifdef RAYTRACERFACILITY
#include <RayTracerManager.hpp>
#include "MLVQRenderer.hpp"
#endif
#include <PhysicsManager.hpp>


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
    Application::Init(applicationConfigs);
#ifdef RAYTRACERFACILITY
    if (enableRayTracing)
        RayTracerManager::Init();
    Application::RegisterUpdateFunction([]() {
        RayTracerManager::Update();
        RayTracerManager::OnGui();
    });
#endif
#pragma region Engine Loop
    Application::Run();
#pragma endregion
#ifdef RAYTRACERFACILITY
    if (enableRayTracing)
        RayTracerManager::End();
#endif
    Application::End();
}
