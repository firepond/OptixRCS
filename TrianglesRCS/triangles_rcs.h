struct Result {
    unsigned int rid;
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
    float3 cam_eye;
    float3 box_center;

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