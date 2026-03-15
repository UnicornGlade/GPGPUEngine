#include <gtest/gtest.h>

#include "camera_orbit.h"

namespace {

CameraViewGPU makeTestCamera()
{
    CameraViewGPU camera = {};
    camera.K.fx = 100.0f;
    camera.K.fy = 100.0f;
    camera.K.cx = 50.0f;
    camera.K.cy = 40.0f;
    camera.K.width = 100;
    camera.K.height = 80;
    camera.E.R[0] = 1.0f;
    camera.E.R[1] = 0.0f;
    camera.E.R[2] = 0.0f;
    camera.E.R[3] = 0.0f;
    camera.E.R[4] = 1.0f;
    camera.E.R[5] = 0.0f;
    camera.E.R[6] = 0.0f;
    camera.E.R[7] = 0.0f;
    camera.E.R[8] = 1.0f;
    camera.E.C[0] = 0.0f;
    camera.E.C[1] = 0.0f;
    camera.E.C[2] = 5.0f;
    camera.E.t[0] = 0.0f;
    camera.E.t[1] = 0.0f;
    camera.E.t[2] = -5.0f;
    camera.magic_bits_guard = CAMERA_VIEW_GPU_MAGIC_BITS_GUARD;
    return camera;
}

} // namespace

TEST(ao_viewer, makePrimaryRayThroughCenterLooksAlongMinusZ)
{
    CameraViewGPU camera = makeTestCamera();
    viewer::PrimaryRay ray = viewer::makePrimaryRay(camera, 50.0f, 40.0f);
    EXPECT_NEAR(ray.origin.x, 0.0f, 1e-6f);
    EXPECT_NEAR(ray.origin.y, 0.0f, 1e-6f);
    EXPECT_NEAR(ray.origin.z, 5.0f, 1e-6f);
    EXPECT_NEAR(ray.direction.x, 0.0f, 1e-6f);
    EXPECT_NEAR(ray.direction.y, 0.0f, 1e-6f);
    EXPECT_NEAR(ray.direction.z, -1.0f, 1e-6f);
}

TEST(ao_viewer, reconstructWorldPointUsesTHitAlongPrimaryRay)
{
    CameraViewGPU camera = makeTestCamera();
    point3f p = viewer::reconstructWorldPoint(camera, 50.0f, 40.0f, 2.5f);
    EXPECT_NEAR(p.x, 0.0f, 1e-6f);
    EXPECT_NEAR(p.y, 0.0f, 1e-6f);
    EXPECT_NEAR(p.z, 2.5f, 1e-6f);
}

TEST(ao_viewer, makeCameraFromOrbitKeepsRequestedFocus)
{
    CameraViewGPU base = makeTestCamera();
    viewer::OrbitCameraState state;
    state.focus = point3f(1.0f, 2.0f, 3.0f);
    state.distance = 4.0f;
    state.yaw = 0.0f;
    state.pitch = 0.0f;

    CameraViewGPU camera = viewer::makeCameraFromOrbit(base, state);
    viewer::PrimaryRay center_ray = viewer::makePrimaryRay(camera, base.K.cx, base.K.cy);
    point3f hit = center_ray.origin + center_ray.direction * state.distance;
    EXPECT_NEAR(hit.x, state.focus.x, 1e-4f);
    EXPECT_NEAR(hit.y, state.focus.y, 1e-4f);
    EXPECT_NEAR(hit.z, state.focus.z, 1e-4f);
}

TEST(ao_viewer, zoomOrbitClampsToPositiveDistance)
{
    viewer::OrbitCameraState state;
    state.distance = 2.0f;
    viewer::zoomOrbit(state, 0.5f);
    EXPECT_NEAR(state.distance, 1.0f, 1e-6f);
    viewer::zoomOrbit(state, 1e-9f);
    EXPECT_GT(state.distance, 0.0f);
}
