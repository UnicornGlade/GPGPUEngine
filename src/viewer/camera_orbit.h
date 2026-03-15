#pragma once

#include "../kernels/shared_structs/camera_gpu_shared.h"

#include <libbase/point.h>

#include <vector>

namespace viewer {

struct PrimaryRay {
    point3f origin;
    point3f direction;
};

struct OrbitCameraState {
    point3f focus = point3f(0.0f, 0.0f, 0.0f);
    float distance = 1.0f;
    float yaw = 0.0f;
    float pitch = 0.0f;
    bool auto_orbit_enabled = true;
};

PrimaryRay makePrimaryRay(const CameraViewGPU &camera, float u, float v);
point3f reconstructWorldPoint(const CameraViewGPU &camera, float u, float v, float t_hit);

point3f computeSceneCenter(const std::vector<point3f> &vertices);
float computeSceneRadius(const std::vector<point3f> &vertices, const point3f &center);

OrbitCameraState makeOrbitStateFromCamera(const CameraViewGPU &camera, const point3f &focus);
void orbitAroundFocus(OrbitCameraState &state, float delta_yaw, float delta_pitch);
void zoomOrbit(OrbitCameraState &state, float zoom_factor);
CameraViewGPU makeCameraFromOrbit(const CameraViewGPU &base_camera, const OrbitCameraState &state);

} // namespace viewer
