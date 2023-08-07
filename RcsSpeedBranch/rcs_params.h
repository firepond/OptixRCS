#include <optix_types.h>

#ifndef RCS_PARAMS_
#define RCS_PARAMS_

// theta polarization corresponds to vertical polarization
// while phi polarization corresponds to horizontal polarization
// H wave, [x,y,0]*ray_direction == 0
 //
 // V wave
enum PolarizationType { HH, VV };

typedef float Result;

struct Params {
	float wave_num;
	float t_value;
	float reflectance;

	// phi, theta
	float2 observer_angle;
	float3 polarization;
	float3 rayDir;
	float3 rayPosStepU;
	float3 rayPosStepR;
	float3 rayPosBegin;

	float3* out_normals;
	Result* result;

	OptixTraversableHandle handle;
};

struct RayGenData {
	float3 cam_eye;
	float3 camera_u, camera_v, camera_w;
};

struct MissData {
	float3 bg_color;
};

struct MeshData {
	float3* vertices;
	uint3* indices;
};

struct HitGroupData {
	void* shape_data;
};



#endif  // RCS_PARAMS_