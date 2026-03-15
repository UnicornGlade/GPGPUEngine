#include "camera_orbit.h"

#include <libbase/runtime_assert.h>

#include <algorithm>
#include <cmath>

namespace viewer {

namespace {

float dot(const point3f &a, const point3f &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

point3f cross(const point3f &a, const point3f &b)
{
    return point3f(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

float length(const point3f &v)
{
    return std::sqrt(dot(v, v));
}

point3f normalize(const point3f &v)
{
    float len = length(v);
    rassert(len > 0.0f, 2026031601242400001, len);
    return v / len;
}

void computeTranslation(const float R[9], const point3f &camera_center, float t[3])
{
    t[0] = -(R[0] * camera_center.x + R[1] * camera_center.y + R[2] * camera_center.z);
    t[1] = -(R[3] * camera_center.x + R[4] * camera_center.y + R[5] * camera_center.z);
    t[2] = -(R[6] * camera_center.x + R[7] * camera_center.y + R[8] * camera_center.z);
}

} // namespace

PrimaryRay makePrimaryRay(const CameraViewGPU &camera, float u, float v)
{
    float x_cam = (u - camera.K.cx) / camera.K.fx;
    float y_cam = -(v - camera.K.cy) / camera.K.fy;
    float z_cam = -1.0f;

    float dx = camera.E.R[0] * x_cam + camera.E.R[3] * y_cam + camera.E.R[6] * z_cam;
    float dy = camera.E.R[1] * x_cam + camera.E.R[4] * y_cam + camera.E.R[7] * z_cam;
    float dz = camera.E.R[2] * x_cam + camera.E.R[5] * y_cam + camera.E.R[8] * z_cam;

    PrimaryRay ray;
    ray.origin = point3f(camera.E.C[0], camera.E.C[1], camera.E.C[2]);
    ray.direction = normalize(point3f(dx, dy, dz));
    return ray;
}

point3f reconstructWorldPoint(const CameraViewGPU &camera, float u, float v, float t_hit)
{
    PrimaryRay ray = makePrimaryRay(camera, u, v);
    return ray.origin + ray.direction * t_hit;
}

point3f computeSceneCenter(const std::vector<point3f> &vertices)
{
    rassert(!vertices.empty(), 2026031601242400002);
    point3f min_v = vertices[0];
    point3f max_v = vertices[0];
    for (const point3f &v: vertices) {
        min_v.x = std::min(min_v.x, v.x);
        min_v.y = std::min(min_v.y, v.y);
        min_v.z = std::min(min_v.z, v.z);
        max_v.x = std::max(max_v.x, v.x);
        max_v.y = std::max(max_v.y, v.y);
        max_v.z = std::max(max_v.z, v.z);
    }
    return (min_v + max_v) * 0.5f;
}

float computeSceneRadius(const std::vector<point3f> &vertices, const point3f &center)
{
    rassert(!vertices.empty(), 2026031601242400003);
    float radius = 0.0f;
    for (const point3f &v: vertices) {
        radius = std::max(radius, length(v - center));
    }
    return radius;
}

OrbitCameraState makeOrbitStateFromCamera(const CameraViewGPU &camera, const point3f &focus)
{
    point3f camera_center(camera.E.C[0], camera.E.C[1], camera.E.C[2]);
    point3f camera_offset = camera_center - focus;
    float distance = std::max(length(camera_offset), 1e-3f);
    point3f dir = camera_offset / distance;

    OrbitCameraState state;
    state.focus = focus;
    state.distance = distance;
    state.yaw = std::atan2(dir.y, dir.x);
    state.pitch = std::asin(std::clamp(dir.z, -1.0f, 1.0f));
    state.auto_orbit_enabled = true;
    return state;
}

void orbitAroundFocus(OrbitCameraState &state, float delta_yaw, float delta_pitch)
{
    state.yaw += delta_yaw;
    state.pitch = std::clamp(state.pitch + delta_pitch, -1.45f, 1.45f);
}

void zoomOrbit(OrbitCameraState &state, float zoom_factor)
{
    rassert(zoom_factor > 0.0f, 2026031601242400004, zoom_factor);
    state.distance = std::clamp(state.distance * zoom_factor, 0.05f, 1e6f);
}

CameraViewGPU makeCameraFromOrbit(const CameraViewGPU &base_camera, const OrbitCameraState &state)
{
    CameraViewGPU camera = base_camera;

    point3f camera_offset(
        state.distance * std::cos(state.pitch) * std::cos(state.yaw),
        state.distance * std::cos(state.pitch) * std::sin(state.yaw),
        state.distance * std::sin(state.pitch));
    point3f camera_center = state.focus + camera_offset;
    point3f forward = normalize(state.focus - camera_center);

    point3f world_up(0.0f, 0.0f, 1.0f);
    if (std::abs(dot(forward, world_up)) > 0.98f) {
        world_up = point3f(0.0f, 1.0f, 0.0f);
    }
    point3f right = normalize(cross(forward, world_up));
    point3f up = normalize(cross(right, forward));
    point3f backward = forward * -1.0f;

    camera.E.C[0] = camera_center.x;
    camera.E.C[1] = camera_center.y;
    camera.E.C[2] = camera_center.z;

    camera.E.R[0] = right.x;
    camera.E.R[1] = right.y;
    camera.E.R[2] = right.z;
    camera.E.R[3] = up.x;
    camera.E.R[4] = up.y;
    camera.E.R[5] = up.z;
    camera.E.R[6] = backward.x;
    camera.E.R[7] = backward.y;
    camera.E.R[8] = backward.z;

    computeTranslation(camera.E.R, camera_center, camera.E.t);
    camera.magic_bits_guard = CAMERA_VIEW_GPU_MAGIC_BITS_GUARD;
    return camera;
}

} // namespace viewer
