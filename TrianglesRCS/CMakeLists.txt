OPTIX_add_sample_executable( TrianglesRCS target_name

  rcs_params.h
  triangles_rcs.h
  triangles_rcs.cpp

  tiny_obj_loader.h

  triangles_rcs.cu
  complex_vector.cu

  reduce.cu

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )
