#include <optix_types.h>

#ifndef RCS_PARAMS_
#define RCS_PARAMS_

// theta polarization corresponds to vertical polarization
// while phi polarization corresponds to horizontal polarization
// H wave, [x,y,0]*ray_direction == 0
 //
 // V wave
enum PolarizationTypes { HH,VV };

struct Result {
    float au_real;
    float au_img;
    float ar_real;
    float ar_img;
    int refCount;
};

struct Params {
    Result* result;
    unsigned int rays_per_dimension;
    // r, phi, theta
    float3 observer_pos;
    //float3 box_center;
    float3 polarization;

    float3 dirN;
    float3* out_normals;
    float3 rayPosStepU;
    float3 rayPosStepR;
    float3 rayPosBegin;

    float freq;
    PolarizationTypes type;
    OptixTraversableHandle handle;
};

struct RayGenData {
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
};

struct MissData {
    float3 bg_color;
};

struct SphereData {
    float3 center;
    float radius;
};

struct MeshData {
    float3* vertices;

    uint3* indices;
};

struct HitGroupData {
    void* shape_data;
};
#endif  // RCS_PARAMS_