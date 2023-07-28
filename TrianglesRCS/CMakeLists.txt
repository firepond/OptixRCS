OPTIX_add_sample_executable( TrianglesRCS target_name
  triangles_rcs.cpp
  triangles_rcs.cu
  triangles_rcs.h
  complex_vector.cu
  tiny_obj_loader.h
  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )
